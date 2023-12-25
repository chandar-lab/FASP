import pandas as pd
from tqdm.notebook import tqdm
import json, torch
from transformers import AutoModelWithLMHead, OPTForCausalLM, AutoTokenizer
#, 
from transformers_pruning_new import AutoTokenizer, BloomTokenizerFast, GPT2Tokenizer, LlamaForCausalLM, LlamaTokenizerFast
from transformers_pruning_new import BloomForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM, GPTJForCausalLM, AutoModelForCausalLM, AutoModelWithLMHead
import os
from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# It is used to compute the effects of attention heads on perplexity using both magnitude and gradient as baselines.

def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--account",
        choices=[
            "abdel1", "olewicki"
        ],
        default="olewicki",
        help="The Compute Canada account that we work on",
    )

    parser.add_argument("-", "--model_list", nargs="+", default=[])

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_names = args.model_list
    weights_norms = ["magnitude_l1_structured", "magnitude_l2_structured", "magnitude_linf_structured"]

    head_contributions_file  = "./model/head_contributions.json"
    # the file containing the importance scores for the heads of the models
    if os.path.exists(head_contributions_file):
      head_contributions = json.load(open(head_contributions_file, "r"))
    else:
      head_contributions = {}

    for model_name in model_names:
      print(model_name)
      biases_scores = "./output/compute_scores_" + str(args.model_list[0].replace("/", "_")) + ".csv"
      biases = None
      if os.path.exists(biases_scores):
        df = pd.read_csv(biases_scores)
        biases = []
        for bias in ["gender_and_sex", "race_ethnicity", "religion", "sexual_orientation", "nationality"]:
          if bias in df["Group"].unique():
            biases += [bias]

      if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
          model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

      elif model_name in ["distilroberta-base", "distilbert-base-cased","bert-base-cased",  "bert-large-cased", "roberta-base","roberta-large"]:
          model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

      elif model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m"]:
          model = GPTNeoXForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

      elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
          model = GPTNeoForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)

      elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
          model = OPTForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)
      
      elif model_name in ["EleutherAI/gpt-j-6B"]:
          model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name, revision="float16",torch_dtype=torch.float16,).to(device)
          # num_heads, num_layers = model.config.n_head, model.config.n_layer
          # head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 
          # num_heads = 16
          # num_layers = 28
          # max_length = 1024

      elif model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
          model = LlamaForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name, revision="float16",torch_dtype=torch.float16,).bfloat16().to(device)

      model_configs = json.load(open("./model/models_config.json", "r"))
      num_heads, num_layers = model_configs[model_name]["num_heads"], model_configs[model_name]["num_layers"] 
      head_dim, max_length = model_configs[model_name]["head_dim"], model_configs[model_name]["max_length"] 

      if model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        tokenizer = GPT2Tokenizer.from_pretrained("./saved_models/cached_tokenizers/" + model_name, padding_side="left")
      else:
        tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + model_name, padding_side="left")

      if model_name in head_contributions.keys():
        # if the json files already has the scores for a model, we load them and add the new scores to them
        model_dict = head_contributions[model_name]
      else:
        model_dict = {}  

      max_length=int(max_length/4)
      # this is just to make make sure to make the code run for different models

      if biases is not None:
        for group in biases:
          model_dict["cont_to_" + group + "_bias"], model_dict["cont_to_ppl"] = [], []

          for head_id in range(num_layers * num_heads):
            # the effects on bias and perplexity are computed for each head of each layer based on the validation dataset
            model_dict["cont_to_" + group + "_bias"].append(df[(df["Head id"] == head_id + 1) & (df["Group"] == group) & (df["Split"] == "valid")]["Bias"].mean())
            model_dict["cont_to_ppl"].append(df[(df["Head id"] == head_id + 1) & (df["Split"] == "valid")]["PPL"].mean())

      for weights_norm in weights_norms:
        # for different norms, we compute the scores for each head of each layer
        if weights_norm not in model_dict.keys():
          model_dict[weights_norm] = []
          for head_id in range(num_layers * num_heads):
            layer_id = int(head_id/num_heads)
            start_idx = head_dim*(head_id - layer_id*num_heads)
            end_idx = head_dim*(head_id - (layer_id*num_heads) + 1)

            head_weights = []

            if model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m"]:
              for name, para in model.named_parameters():
                if name in ["gpt_neox.layers." + str(layer_id) + ".attention.query_key_value.weight", "gpt_neox.layers." + str(layer_id) + ".attention.query_key_value.bias", "gpt_neox.layers." + str(layer_id) + ".attention.dense.weight"]:
                  head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][start_idx:end_idx,:]),torch.flatten(head_weights[0][start_idx + head_dim*num_heads: end_idx + head_dim*num_heads,:]), torch.flatten(head_weights[0][start_idx + 2*head_dim*num_heads : end_idx + 2*head_dim*num_heads,:])))
              head_attn_bias = torch.cat((torch.flatten(head_weights[1][start_idx:end_idx]),torch.flatten(head_weights[1][start_idx + head_dim*num_heads : end_idx + head_dim*num_heads]), torch.flatten(head_weights[1][start_idx + 2*head_dim*num_heads : end_idx + 2*head_dim*num_heads])))
              head_proj_weight = torch.flatten(head_weights[2][:,start_idx:end_idx])

            if model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
              for name, para in model.named_parameters():
                if name in ["transformer.h." + str(layer_id) + ".attn.attention.k_proj.weight", "transformer.h." + str(layer_id) + ".attn.attention.v_proj.weight", "transformer.h." + str(layer_id) + ".attn.attention.q_proj.weight","transformer.h." + str(layer_id) + ".attn.attention.out_proj.weight"]:
                  head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][start_idx:end_idx,:]),torch.flatten(head_weights[1][start_idx:end_idx,:]),torch.flatten(head_weights[2][start_idx:end_idx,:])))
              head_attn_bias = torch.randn(0).to(device) #any empty tensor because this model does not have a bias parameter for the attention heads
              head_proj_weight = torch.flatten(head_weights[3][:,start_idx:end_idx])

            if model_name in ["EleutherAI/gpt-j-6B"]:
              for name, para in model.named_parameters():
                if name in ["transformer.h." + str(layer_id) + ".attn.k_proj.weight", "transformer.h." + str(layer_id) + ".attn.v_proj.weight", "transformer.h." + str(layer_id) + ".attn.q_proj.weight","transformer.h." + str(layer_id) + ".attn.out_proj.weight"]:
                  head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][start_idx:end_idx,:]),torch.flatten(head_weights[1][start_idx:end_idx,:]),torch.flatten(head_weights[2][start_idx:end_idx,:])))
              head_attn_bias = torch.randn(0).to(device) #any empty tensor because this model does not have a bias parameter for the attention heads
              head_proj_weight = torch.flatten(head_weights[3][:,start_idx:end_idx])

            if model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
              for name, para in model.named_parameters():
                if name in ["model.layers." + str(layer_id) + ".self_attn.q_proj.weight", "model.layers." + str(layer_id) + ".self_attn.k_proj.weight", "model.layers." + str(layer_id) + ".self_attn.v_proj.weight","model.layers." + str(layer_id) + ".self_attn.o_proj.weight"]:
                  head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][start_idx:end_idx,:]),torch.flatten(head_weights[1][start_idx:end_idx,:]),torch.flatten(head_weights[2][start_idx:end_idx,:])))
              head_attn_bias = torch.randn(0).to(device) #any empty tensor because this model does not have a bias parameter for the attention heads
              head_proj_weight = torch.flatten(head_weights[3][:,start_idx:end_idx])



            if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]:
              for name, para in model.named_parameters():
                if name in ["transformer.h." + str(layer_id) + ".attn.c_attn.weight", "transformer.h." + str(layer_id) + ".attn.c_attn.bias", "transformer.h." + str(layer_id) + ".attn.c_proj.weight"]:
                  head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][:, start_idx:end_idx]),torch.flatten(head_weights[0][:, start_idx + head_dim*num_heads: end_idx + head_dim*num_heads]), torch.flatten(head_weights[0][:, start_idx + 2*head_dim*num_heads : end_idx + 2*head_dim*num_heads])))
              head_attn_bias = torch.cat((torch.flatten(head_weights[1][start_idx:end_idx]),torch.flatten(head_weights[1][start_idx + head_dim*num_heads : end_idx + head_dim*num_heads]), torch.flatten(head_weights[1][start_idx + 2*head_dim*num_heads : end_idx + 2*head_dim*num_heads])))
              head_proj_weight = torch.flatten(head_weights[2][start_idx:end_idx,:])

            elif model_name in ["distilroberta-base", "distilbert-base-cased","bert-base-cased", "bert-large-cased", "roberta-base","roberta-large"]:
              for name, para in model.named_parameters():
                if (".encoder.layer." + str(layer_id) + ".attention.self." in name) or (".encoder.layer." + str(layer_id) + ".attention.output.dense.weight" in name) or (".transformer.layer." + str(layer_id) + ".attention.q" in name) or (".transformer.layer." + str(layer_id) + ".attention.k" in name)  or (".transformer.layer." + str(layer_id) + ".attention.v" in name)  or (".transformer.layer." + str(layer_id) + ".attention.out_lin.weight" in name):
                    head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][:, start_idx:end_idx]),torch.flatten(head_weights[2][:, start_idx:end_idx]), torch.flatten(head_weights[4][:, start_idx:end_idx])))
              head_attn_bias = torch.cat((torch.flatten(head_weights[1][start_idx:end_idx]),torch.flatten(head_weights[3][start_idx:end_idx]), torch.flatten(head_weights[5][start_idx:end_idx])))
              head_proj_weight = torch.flatten(head_weights[6][start_idx:end_idx,:])

            elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
              for name, para in model.named_parameters():
                if ("model.decoder.layers." + str(layer_id) + ".self_attn.k_proj" in name) or ("model.decoder.layers." + str(layer_id) + ".self_attn.v_proj" in name) or ("model.decoder.layers." + str(layer_id) + ".self_attn.q_proj" in name) or ("model.decoder.layers." + str(layer_id) + ".self_attn.out_proj.weight" in name):
                    head_weights.append(para)

              head_attn_weight = torch.cat((torch.flatten(head_weights[0][start_idx:end_idx,:]),torch.flatten(head_weights[2][start_idx:end_idx,:]), torch.flatten(head_weights[4][start_idx:end_idx,:])))
              head_attn_bias = torch.cat((torch.flatten(head_weights[1][start_idx:end_idx]),torch.flatten(head_weights[3][start_idx:end_idx]), torch.flatten(head_weights[5][start_idx:end_idx])))
              head_proj_weight = torch.flatten(head_weights[6][:,start_idx:end_idx])


            if weights_norm == "magnitude_l1_structured":
              head_weights_norm = torch.norm(torch.cat((head_attn_weight, head_attn_bias, head_proj_weight)), p=1)
            elif weights_norm == "magnitude_l2_structured":
              head_weights_norm = torch.norm(torch.cat((head_attn_weight, head_attn_bias, head_proj_weight)), p=2)
            elif weights_norm == "magnitude_linf_structured":
              head_weights_norm = torch.norm(torch.cat((head_attn_weight, head_attn_bias, head_proj_weight)), float('inf'))

            model_dict[weights_norm].append(head_weights_norm.tolist())


      # we normalize the scores for each head
      for score in  weights_norms:  
        absolute_max = (max(abs(max(model_dict[score])), abs(min(model_dict[score]))))
        for head_id in range(num_layers * num_heads):
          model_dict[score][head_id] /= absolute_max



      # if 'mask_gradient_l2_structured' not in model_dict.keys():
      #   # if we have the gradient scores already for a model, we don't need to compute them again
      #   stride = 512
      #   torch.manual_seed(1)
      #   names = []
      #   with open("./model/wikitext-2-raw-v1_valid.txt", 'r') as fp:
      #       for line in fp:
      #           x = line
      #           names.append(x)

      #   encodings = tokenizer("".join(names) , return_tensors="pt")
      #   seq_len = encodings.input_ids.size(1)
      #   head_importance = torch.zeros(num_layers, num_heads).to(device)
      #   head_mask = torch.ones(num_layers, num_heads).to(device)
      #   head_mask.requires_grad_(requires_grad=True)

      #   prev_end_loc = 0
      #   heads_gradients = torch.zeros([num_layers * num_heads]).to(device)
      #   for begin_loc in tqdm(range(0, seq_len, stride)):
      #       end_loc = min(begin_loc + max_length, seq_len)
      #       trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
      #       input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
      #       target_ids = input_ids.clone()
      #       target_ids[:, :-trg_len] = -100


      #       outputs = model(input_ids, labels=target_ids, head_mask=head_mask)

      #       outputs.loss.backward()
      #       head_importance += head_mask.grad.abs().detach()

      #       prev_end_loc = end_loc
      #       if end_loc == seq_len:
      #           break

      #   # Layerwise importance normalization
      #   exponent = 2
      #   norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
      #   head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20
      #   head_importance = head_importance.reshape(num_layers * num_heads).tolist()

      #   absolute_max = (max(abs(max(head_importance)), abs(min(head_importance))))
      #   for head_id in range(num_layers * num_heads):
      #     head_importance[head_id] /= absolute_max
      #   model_dict['mask_gradient_l2_structured'] = head_importance

      head_contributions[model_name] = model_dict

    json.dump(head_contributions, open("./model/head_contributions.json", "w"))