import torch
import wandb
import json
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from model.generation import process_prompts,compute_ppl
from utils import parameters_to_prune
from transformers import BloomForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, BloomTokenizerFast, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM
from transformers_pruning import BloomForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import torch.nn.utils.prune as prune


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are using. We normally run every experiment for 5 seeds.",
    )     

    parser.add_argument(
        "--head_knockout",
        type=int,
        default=None,
        help="the id of the attention head to be knocked out in the language generation models",
    )                       
    parser.add_argument(
        "--model",
        choices=[
            "bert-base-cased",
            "bert-base-uncased",
            "bert-large-cased",
            "bert-large-uncased",
            "roberta-base",
            "roberta-large",
            "distilroberta-base",
            "distilbert-base-cased",
            "distilbert-base-uncased",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "gpt2-xl",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
            "EleutherAI/gpt-j-6B",
            "facebook/opt-350m",
            "facebook/opt-1.3b",
            "facebook/opt-2.7b",
            "facebook/opt-6.7b",
            "bigscience/bloom-560m",
            "bigscience/bloom-1b1",
            "bigscience/bloom-3b",
            "bigscience/bloom-7b1",
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m",
            "EleutherAI/pythia-410m",
            "EleutherAI/pythia-1b",
            "EleutherAI/pythia-1.4b",
            "EleutherAI/pythia-2.8b",
            "EleutherAI/pythia-6.9b",
            "EleutherAI/pythia-12b",     
            "EleutherAI/gpt-j-6B"       
        ],
        default="EleutherAI/gpt-neo-125M",
        help="Type of language generation model used",
    )
    parser.add_argument(
        "--method",
        choices=[
            "magnitude_l1_structured",
            "magnitude_l2_structured", 
            "magnitude_linf_structured",
            "mask_gradient_l2_structured",
            "random_structured",
            "magnitude_unstructured",
            "random_unstructured",
            "FASP",
            "bias_only",
            "ppl_only",
            None,
        ],
        default=None,
        help="Method for pruning the attention heads",
    )
    parser.add_argument(
        "--pruned_heads_ratio",
        type=float,
        default=0,
        help="The ratio of the pruned attention heads",
    )       
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The hyperparameter controling the percentage of examples that are considered important for performance",
    )            
    parser.add_argument(
        "--prompting",
        choices=[
            "rtp",
            "PANDA",
            "BOLD",
            "holistic",
        ],
        default="holistic",
        help="Type of prompt used for the language model",
    )   
    parser.add_argument(
        "--targeted_group_PANDA",
        choices=[
            "woman",
            "non-binary",
            "asian",
            "black",
            "man",
            "middle-aged",
            "white",
            "adult",
            "hispanic",
            "young",
            "child",
            "pacific-islander",
            "native-american",
            "senior",            
        ],
        default=None,
        help="The group for which biased is assessed using the PANDA framework",
    )        
    parser.add_argument(
        "--targeted_bias_BOLD",
        choices=[
            "gender",
            "political_ideology",
            "profession",
            "race",
            "religious_ideology",        
        ],
        default=None,
        help="The group for which biased is assessed using the BOLD framework",
    )    
    parser.add_argument(
        "--targeted_holistic_bias",
        choices=[
            "characteristics",
            "ability",
            "gender_and_sex",
            "socioeconomic_class",
            "race_ethnicity",  
            "body_type",
            "cultural",
            "religion",
            "age",
            "nonce",
            "sexual_orientation",  
            "political_ideologies",  
            "nationality",
            "NaN",         
        ],
        default="gender_and_sex",
        help="The group for which biased is assessed using the holistic bias framework",
    )            
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for the language model.",
    )
    #1024
    parser.add_argument(
        "--stride",
        type=int,
        default=512,
        help="Stride used for computing the model preplexity. This corresponds to the number of tokens the model conditions on at each step.",
    )    
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=None,
        help="We partition the dataset into chunks to avoid memory issues. This is the id of the chunk that we are processing.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2125,
        help="We partition the dataset into chunks to avoid memory issues. This is the size of each chunk.",
    )
    parser.add_argument(
        "--max_continuation_length",
        type=int,
        default=40,
        help="The maximum length of the continuation for the language generation model",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=22,
        help="The maximum length of the prompt for the language generation model",
    )    
    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to the output",
    )
    parser.add_argument(
        "--use_gender_scores",
        type=bool,
        default=True,
        help="Whether or not to the head scores for gender bias when reducing other biases",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Whether or not to use wandb to visualize the results",
    )
              
                  
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.use_wandb:
        wandb_mode = "online"
    else:
        wandb_mode = "offline"

    wandb.init(
        name=str(args.model),
        project="Text generation bias in LLM",
        config=args,
        mode=wandb_mode,
    )

    torch.manual_seed(args.seed)

    path_to_prompts= "./prompts/" + str(args.prompting) + "/"

    if args.model in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")   # Initialize tokenizer
        # number of heads per layer, and number of layers
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 

    elif args.model in ["distilroberta-base", "distilbert-base-cased", "bert-base-cased",  "bert-large-cased", "roberta-base","roberta-large"]:
        model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")   # Initialize tokenizer
        # number of heads per layer, and number of layers
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 
    

    elif args.model in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
        num_heads, num_layers = model.config.num_heads, model.config.num_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif args.model in ["EleutherAI/gpt-j-6B"]:
        model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 

    elif args.model in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
        model = OPTForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif args.model in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = BloomTokenizerFast.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.n_layer
        head_dim, max_length = int(model.config.hidden_size/num_heads), 1024

    elif args.model in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:
        model = GPTNeoXForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model).to(device)
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

        
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if args.method == None:
        # this is the case when we are trying to knock out only one head to know the effect pf removing it on fairness and performance
        idx_pruned_heads = [args.head_knockout]

    elif args.method == "magnitude_unstructured":

        idx_pruned_heads = []
        prune.global_unstructured(
            parameters_to_prune(model_name = args.model, model = model),
            pruning_method = prune.L1Unstructured,
            amount = args.pruned_heads_ratio,
        )

    elif args.method == "random_unstructured":

        idx_pruned_heads = []
        prune.global_unstructured(
            parameters_to_prune(model_name = args.model, model = model),
            pruning_method = prune.RandomUnstructured,
            amount = args.pruned_heads_ratio,
        )


    else:
        # this is the other case when we are systemetically pruning some percentage of the total heads
        head_contributions = json.load(open("./model/head_contributions.json", "r"))
        idx_pruned_heads = []
        num_pruned_heads  = int(num_heads * num_layers * args.pruned_heads_ratio)
        if num_pruned_heads != 0:
            # We execlude the case where there is nothing to prune
            if args.method == "random_structured":
                # random pruning chooses a random subset of heads to prune
                idx_pruned_heads = list(np.random.choice(num_heads * num_layers, num_pruned_heads, replace=False))  
            elif args.method in ["mask_gradient_l2_structured", "magnitude_l1_structured", "magnitude_l2_structured", "magnitude_linf_structured", "magnitude_l1_only_wo_structured", "magnitude_l2_only_wo_structured","magnitude_linf_only_wo_structured"]:
                # magnitude pruning chooses the heads with the lowest magnitude to prune
                magnitude_scores = head_contributions[args.model][args.method]
                threshold = np.sort(np.array(magnitude_scores))[num_pruned_heads-1]
                idx_pruned_heads = [index for index, item in enumerate(magnitude_scores) if item <= threshold]

            elif args.method == "ppl_only":
                ppl_scores = head_contributions[args.model]["cont_to_ppl"]
                threshold = np.sort(ppl_scores)[num_pruned_heads-1]
                idx_pruned_heads = [index for index, item in enumerate(ppl_scores) if item <= threshold]

            elif args.method == "bias_only":
                if args.use_gender_scores:
                    ours_scores = head_contributions[args.model]["cont_to_gender_and_sex" + "_bias"]
                else:
                    ours_scores = head_contributions[args.model]["cont_to_" + args.targeted_holistic_bias + "_bias"]
                threshold = np.sort(ours_scores)[num_pruned_heads-1]
                idx_pruned_heads = [index for index, item in enumerate(ours_scores) if item <= threshold]


            elif args.method == "FASP":
                if args.use_gender_scores:
                    ours_scores = head_contributions[args.model]["cont_to_gender_and_sex" + "_bias"]
                else:
                    ours_scores = head_contributions[args.model]["cont_to_" + args.targeted_holistic_bias + "_bias"]
                ppl_scores = head_contributions[args.model]["cont_to_ppl"]

                num_non_imp_heads_perf = int(num_heads * num_layers) - int(num_heads * num_layers * args.gamma)
                threshold_perf = np.sort(ppl_scores)[num_non_imp_heads_perf-1]

                our_scores_modified = [ours_scores[index] for index, item in enumerate(ppl_scores) if item <= threshold_perf]
                threshold_bias = np.sort(our_scores_modified)[num_pruned_heads-1]

                for index, item in enumerate(ours_scores):
                    if ((ppl_scores[index] <= threshold_perf) and (ours_scores[index] <= threshold_bias)):
                        idx_pruned_heads += [index]   

    idx_pruned_heads_relative = {}
    idx_pruned_layers = [int(x / num_heads) for x in idx_pruned_heads]

    for layer in list(set(idx_pruned_layers)):
        idx_pruned_heads_relative[layer] = [idx_pruned_heads[i]%num_heads for i,x in enumerate(idx_pruned_layers) if x == layer]


    if args.model in ["distilroberta-base", "distilbert-base-cased","gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl","bert-base-cased",  "bert-large-cased", "roberta-base","roberta-large"]:
        model.prune_heads(idx_pruned_heads_relative)


    elif args.model in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:

        for head in idx_pruned_heads: 
            num_heads_after_pruning = num_heads
            layer_id = int(head/num_heads)
            num_pruned_heads_same_layer = 0
            for idx_pruned_head_relative in idx_pruned_heads_relative[layer_id]:
              if idx_pruned_head_relative < head%num_heads:
                num_pruned_heads_same_layer += 1
                # the idea is to see if some other heads are pruned in the samer layer before the current head. If so, we shift the current head index by the number of heads that are pruned before it in the same layer.
                # Also, the number of heads after pruning is reduced by the number of heads that are pruned before it in the same layer.

            head -= num_pruned_heads_same_layer
            num_heads_after_pruning -= num_pruned_heads_same_layer

            start = head_dim*(head - layer_id*num_heads)
            end = head_dim*(head - (layer_id*num_heads) + 1)
            for name, para in model.named_parameters():
                if "transformer.h." + str(layer_id) + ".attn.attention.k_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)
                elif "transformer.h." + str(layer_id) + ".attn.attention.v_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)
                elif "transformer.h." + str(layer_id) + ".attn.attention.q_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)
                elif "transformer.h." + str(layer_id) + ".attn.attention.out_proj.weight" in name:
                    para.data =  torch.cat((para.data[:,0:start], para.data[:,end:]), dim = 1)


    elif args.model in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:

        for head in idx_pruned_heads: 
            num_heads_after_pruning = num_heads
            layer_id = int(head/num_heads)
            num_pruned_heads_same_layer = 0
            for idx_pruned_head_relative in idx_pruned_heads_relative[layer_id]:
              if idx_pruned_head_relative < head%num_heads:
                num_pruned_heads_same_layer += 1

            head -= num_pruned_heads_same_layer
            num_heads_after_pruning -= num_pruned_heads_same_layer

            start = head_dim*(head - layer_id*num_heads)
            end = head_dim*(head - (layer_id*num_heads) + 1)
            for name, para in model.named_parameters():
                if "model.decoder.layers." + str(layer_id) + ".self_attn.k_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)

                elif "model.decoder.layers." + str(layer_id) + ".self_attn.k_proj.bias" in name:
                    para.data =  torch.cat((para.data[0:start], para.data[end:]))         

                elif "model.decoder.layers." + str(layer_id) + ".self_attn.v_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)

                elif "model.decoder.layers." + str(layer_id) + ".self_attn.v_proj.bias" in name:
                    para.data =  torch.cat((para.data[0:start], para.data[end:]))         

                elif "model.decoder.layers." + str(layer_id) + ".self_attn.q_proj.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:,:]), dim = 0)

                elif "model.decoder.layers." + str(layer_id) + ".self_attn.q_proj.bias" in name:
                    para.data =  torch.cat((para.data[0:start], para.data[end:]))   
                          
                elif "model.decoder.layers." + str(layer_id) + ".self_attn.out_proj.weight" in name:
                    para.data =  torch.cat((para.data[:,0:start], para.data[:,end:]), dim = 1)

    elif args.model in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        for head in idx_pruned_heads: 
            num_heads_after_pruning = num_heads
            layer_id = int(head/num_heads)
            num_pruned_heads_same_layer = 0
            for idx_pruned_head_relative in idx_pruned_heads_relative[layer_id]:
              if idx_pruned_head_relative < head%num_heads:
                num_pruned_heads_same_layer += 1

            head -= num_pruned_heads_same_layer
            num_heads_after_pruning -= num_pruned_heads_same_layer
            start = head_dim*(head - layer_id*num_heads)
            end = head_dim*(head - (layer_id*num_heads) + 1)
            for name, para in model.named_parameters():
                if "transformer.h." + str(layer_id) + ".self_attention.query_key_value.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:start + head_dim*num_heads_after_pruning,:],para.data[end + head_dim*num_heads_after_pruning:start + 2*head_dim*num_heads_after_pruning,:], para.data[end + 2*head_dim*num_heads_after_pruning: ,:]), dim = 0)
                
                elif "transformer.h" + str(layer_id) + ".self_attention.query_key_value.bias" in name:
                    para.data =  torch.cat((para.data[0:start], para.data[end:start + head_dim*num_heads_after_pruning],para.data[end + head_dim*num_heads_after_pruning:start + 2*head_dim*num_heads_after_pruning], para.data[end + 2*head_dim*num_heads_after_pruning: ]))         
                
                elif "transformer.h." + str(layer_id) + ".self_attention.dense.weight" in name:
                    para.data =  torch.cat((para.data[:,0:start], para.data[:,end:start + head_dim*num_heads_after_pruning],para.data[:,end + head_dim*num_heads_after_pruning:start + 2*head_dim*num_heads_after_pruning], para.data[:,end + 2*head_dim*num_heads_after_pruning:]), dim = 1)

    elif args.model in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:

        for head in idx_pruned_heads: 
            num_heads_after_pruning = num_heads
            layer_id = int(head/num_heads)
            num_pruned_heads_same_layer = 0
            for idx_pruned_head_relative in idx_pruned_heads_relative[layer_id]:
              if idx_pruned_head_relative < head%num_heads:
                num_pruned_heads_same_layer += 1

            head -= num_pruned_heads_same_layer
            num_heads_after_pruning -= num_pruned_heads_same_layer

            start = head_dim*(head - layer_id*num_heads)
            end = head_dim*(head - (layer_id*num_heads) + 1)
            for name, para in model.named_parameters():
                if "gpt_neox.layers." + str(layer_id) + ".attention.query_key_value.weight" in name:
                    para.data =  torch.cat((para.data[0:start,:], para.data[end:start + head_dim*num_heads_after_pruning,:],para.data[end + head_dim*num_heads_after_pruning:start + 2*head_dim*num_heads_after_pruning,:], para.data[end + 2*head_dim*num_heads_after_pruning:,:]), dim = 0)
                elif "gpt_neox.layers." + str(layer_id) + ".attention.query_key_value.bias" in name:
                    para.data =  torch.cat((para.data[0:start], para.data[end:start + head_dim*num_heads_after_pruning],para.data[end + head_dim*num_heads_after_pruning:start + 2*head_dim*num_heads_after_pruning], para.data[end + 2*head_dim*num_heads_after_pruning: ]))         
                elif "gpt_neox.layers." + str(layer_id) + ".attention.dense.weight" in name:
                    para.data =  torch.cat((para.data[:,0:start], para.data[:,end:]), dim = 1)


    ppl = compute_ppl(model,tokenizer, args.stride, int(max_length/2))  
    tox_model = torch.load("./saved_models/unbiased/unbiased.pt")
    tox_model.device = device 
    sentiment_analyzer = SentimentIntensityAnalyzer()   
    model_name = args.model.replace("/", "_")

    for split in ["test","valid"]: 
        output_dir = args.output_dir + "/prompt_" + str(args.prompting) + "_h" + str(args.head_knockout) 
        prompts_file = json.load(open(path_to_prompts + "social_biases_" + split + ".json", "r"))
        output_dir += "_" + split + "/"
        output_dir +=  str(args.method) + "_" + str(args.pruned_heads_ratio) + "_gamma" + str(args.gamma)  + "/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        process_prompts(model_name, model, tokenizer, tox_model, sentiment_analyzer, wandb, ppl[split], args.batch_size, args.max_continuation_length, args.max_prompt_length, args.prompting, output_dir, prompts_file, args.chunk_id, args.chunk_size, args.targeted_holistic_bias, split)
        



            
        






        




            
