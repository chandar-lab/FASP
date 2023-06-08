import torch
import random
import re
import os
import pandas as pd
import wandb
import json
import numpy as np
from argparse import ArgumentParser
from tqdm.notebook import tqdm
from pathlib import Path
from detoxify import Detoxify
from scipy.stats import anderson_ksamp
from model.metrics import calculate_significance,evaluate_fairness_disparity
from model.generation import gen_prompt,process_group_scores,process_prompts,compute_ppl
from utils import get_head_dim_idx
from transformers import AutoModelForCausalLM, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=0).to(device)

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()
    # choosing between our work and the baselines
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="The seed that we are using. We normally run every experiment for 5 seeds.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=4e-6,
        help="learning rate for the language generation models",
    )     
    parser.add_argument(
        "--attention_dropout",
        type=float,
        default=None,
        help="attention dropout in the language generation models",
    )
    parser.add_argument(
        "--head_knockout",
        type=int,
        default=None,
        help="the id of the attention head to be knocked out in the language generation models",
    )    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=None,
        help="number of layers in the language generation models",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="number of heads in the language generation models",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=None,
        help="hidden size in the language generation models",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="attention modulation factor in the language generation models",
    )                    
    parser.add_argument(
        "--model",
        choices=[
            "bert-base-cased",
            "bert-base-uncased",
            "bert-large-cased",
            "bert-large-uncased",
            "roberta-base",
            "distilroberta-base",
            "distilbert-base-cased",
            "distilbert-base-uncased",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "distilgpt2",
            "EleutherAI/gpt-neo-125M",
            "EleutherAI/gpt-neo-1.3B",
            "EleutherAI/gpt-neo-2.7B",
        ],
        default="EleutherAI/gpt-neo-125M",
        help="Type of model used",
    )
    parser.add_argument(
        "--method",
        choices=[
            "random",
            "magntude",
            "ours_ppl",
            "ours_magnitude",
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
        "--alpha",
        type=float,
        default=0.5,
        help="The hyperparameter controling the trade-off between the importance of heads for performance and fairness",
    )        
    parser.add_argument(
        "--prompting",
        choices=[
            "M_CM",
            "CF_F",
            "MCF_FCM",
            "M_F",
            "rtp",
            "PANDA",
            "BOLD",
        ],
        default="PANDA",
        help="Type of model used",
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
        help="The group for which biased is assessed",
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
        help="The targeted bias that is assessed in BOLD",
    )            
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for the classifier.",
    )
    parser.add_argument(
        "--prompt_chunk_id",
        type=int,
        default=0,
        help="The id of the prompt chunk that we work on. To make the experiments faster, we split the prompts into chunks, and process them in parallel.",
    )  
    parser.add_argument(
        "--prompt_chunk_size",
        type=int,
        default=5000,
        help="The number of examples in one prompt chunk",
    )        
    parser.add_argument(
        "--max_continuation_length",
        type=int,
        default=50,
        help="The maximum length of the continuation that we choose to give to the language generation model",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=75,
        help="The maximum length of the prompt for the language generation model",
    )    
    parser.add_argument(
        "--model_checkpoint_path",
        default="./saved_models/checkpoint-",
        help="Path to the saved classifier checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        default="output/",
        help="Directory to the output",
    )
    
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=False,
        help="Whether or not to use wandb to visualize the results",
    )
    parser.add_argument(
        "--random_perturbation",
        type=bool,
        default=False,
        help="Whether or not to add random noise to the model weights as a debiasing method, which is described in https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf",
    ) 
    parser.add_argument(
        "--random_perturbation_mean",
        type=float,
        default=1,
        help="The mean of the noise in the random perturbation, as the original paper https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf",
    )       
    parser.add_argument(
        "--random_perturbation_std",
        type=float,
        default=0.1,
        help="The standard deviation of the noise in the random perturbation, as the original paper https://proceedings.neurips.cc/paper/2020/file/1d8d70dddf147d2d92a634817f01b239-Paper.pdf",
    )            
    parser.add_argument(
        "--famous_names",
        type=bool,
        default=False,
        help="Whether or not to use names of famous actors/actresses",
    )    
    parser.add_argument(
        "--use_amulet",
        type=bool,
        default=False,
        help="Whether or not to run the code on Amulet, which is the cluster used at Microsoft research",
    )      
    parser.add_argument(
        "--analyze_attention",
        type=bool,
        default=False,
        help="Whether or not to analyze the attention map",
    )                    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.seed != None:
        my_seed = args.seed
    else:
        my_seed = np.random.randint(10000, size=1)[0]

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

    # output_dir = args.output_dir

    # if args.use_amulet:
    #     output_dir = f"{os.environ['AMLT_OUTPUT_DIR']}/" + output_dir

    path_to_prompts= "./prompts/" + str(args.prompting) + "/"

    if args.model in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]:
        model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + args.model)
    if args.attention_dropout:
        model.config.attention_dropout=args.attention_dropout

    if args.num_layers:
        model.config.num_layers=args.num_layers

    if args.num_heads:
        model.config.num_heads=args.num_heads

    if args.hidden_size:
        model.config.hidden_size=args.hidden_size

    if args.method == None:
        # this is the case when we are trying to knock out only one head to know the effect pf removing it on fairness and performance
        idx_pruned_heads = [args.head_knockout]
    else:
        # this is when we are systemetically pruning a large percentage of heads
        head_contributions = json.load(open("./model/head_contributions.json", "r"))
        idx_pruned_heads = []
        num_heads = model.config.num_hidden_layers * model.config.num_heads
        if int(num_heads * args.pruned_heads_ratio) != 0:
            # We execlude the case where there is nothing to prune
            if args.method == "random":
                idx_pruned_heads = list(np.random.choice(num_heads, int(num_heads * args.pruned_heads_ratio), replace=False))  
            elif args.method == "magnitude":
                magnitude_scores = head_contributions[args.model]["weight_magnitude"]
                threshold = np.sort(np.array(magnitude_scores))[int(num_heads * args.pruned_heads_ratio)-1]
                idx_pruned_heads = [index for index, item in enumerate(magnitude_scores) if item <= threshold]

            elif args.method == "ours_magnitude":
                ours_scores = head_contributions[args.model]["cont_to_bias"]
                magnitude_scores = head_contributions[args.model]["weight_magnitude"]
                ours_magnitude_scores = args.alpha * np.array(ours_scores) + (1-args.alpha) * np.array(magnitude_scores)
                threshold = np.sort(ours_magnitude_scores)[int(num_heads * args.pruned_heads_ratio)-1]
                idx_pruned_heads = [index for index, item in enumerate(ours_magnitude_scores) if item <= threshold]

            elif args.method == "ours_ppl":
                ours_scores = head_contributions[args.model]["cont_to_bias"]
                ppl_scores = head_contributions[args.model]["cont_to_ppl"]
                ours_ppl_scores = args.alpha * np.array(ours_scores) + (1-args.alpha) * np.array(ppl_scores)
                threshold = np.sort(ours_ppl_scores)[int(num_heads * args.pruned_heads_ratio)-1]
                idx_pruned_heads = [index for index, item in enumerate(ours_ppl_scores) if item <= threshold]


    for head_id in idx_pruned_heads:
        if head_id != None:
            start_idx, end_idx, layer_id = get_head_dim_idx(head_id, model)  
            for name, para in model.named_parameters():
                if "transformer.h." + str(layer_id) + ".attn.attention.out_proj.weight" in name:
                    print(name)
                    para.data[start_idx: end_idx] *= 0    


    if args.beta != None:
        for layer_id in range(model.config.num_hidden_layers):
            for name, para in model.named_parameters():
                if "transformer.h." + str(layer_id) + ".attn.attention.q_proj.weight" in name:
                    para.data *= args.beta


    if args.random_perturbation:
        for name, para in model.named_parameters():
            para.data *= (torch.randn_like(para) * args.random_perturbation_std + args.random_perturbation_mean)

    tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + args.model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model=model.to(device)
    # print("yaaaa rb")
    # print("argoook")
    ppl = compute_ppl(model,tokenizer)
    # tox_model = Detoxify("unbiased")
    # tox_model = torch.load("./saved_models/detoxify/pytorch_model.bin").to(device)
    tox_model = torch.load("./saved_models/unbiased/unbiased.pt")
    tox_model.device = device 
    sentiment_analyzer = SentimentIntensityAnalyzer()   
    model_name = args.model.replace("/", "_")

    if args.prompting=="PANDA":
        for split in ["valid"]: 
            output_dir = args.output_dir + "/prompt_" + str(args.prompting) + "_" + str(args.beta) + "_" + str(args.random_perturbation) + "_h" + str(args.head_knockout) 
            domain_pd = json.load(open(path_to_prompts + "social_biases_" + split + ".json", "r"))
            output_dir += "_" + split + "/"
            output_dir +=  str(args.method) + "_" + str(args.pruned_heads_ratio) + "_alpha" + str(args.alpha)  + "/"
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            process_prompts(model_name, model, tokenizer, tox_model, sentiment_analyzer, wandb, ppl, args.batch_size, args.max_continuation_length, args.max_prompt_length, args.prompting, output_dir, domain_pd, split)
            
    elif args.prompting=="BOLD":
        output_dir = args.output_dir + "/prompt_" + str(args.prompting) + "_" + str(args.targeted_bias_BOLD) + "_" + str(args.beta) + "_" + str(args.random_perturbation) + "_h" + str(args.head_knockout) + "/"
        output_dir +=  str(args.method) + "_" + str(args.pruned_heads_ratio) + "_alpha" + str(args.alpha)  + "/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        domain_pd = json.load(open(path_to_prompts + args.targeted_bias_BOLD + "_prompt.json", "r"))
        process_prompts(model_name, model, tokenizer, tox_model, sentiment_analyzer, wandb, ppl, args.batch_size, args.max_continuation_length, args.max_prompt_length, args.prompting, output_dir, domain_pd, split=None)

    


            
        






        




            
