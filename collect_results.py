import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BloomForCausalLM, AutoModelForCausalLM, AutoTokenizer, AutoModelWithLMHead, BloomTokenizerFast, OPTForCausalLM, AutoModelForCausalLM, GPT2Tokenizer, GPTJForCausalLM
from transformers_pruning import BloomForCausalLM, GPTNeoForCausalLM, GPTNeoXForCausalLM
import torch
from main import parse_args
import json
import numpy as np
from argparse import ArgumentParser


def parse_args():
    """Parses the command line arguments."""
    parser = ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "head_scores",
            "compare_to_baselines",
        ],
        default="compare_to_baselines",
        help="The experiment that we want to run",
    )

    parser.add_argument(
        "--account",
        choices=[
            "abdel", "olewicki"
        ],
        default="olewicki",
        help="The Compute Canada account that we work on",
    )


    parser.add_argument("-model", "--model_list", nargs="+", default=[])

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df_all_seeds = pd.DataFrame()
    args = parse_args()

    # this files has the information about the groups that are targeted in the validation data prompts (for example, different religions, genders, etc.)
    groups_valid = {}
    groups_valid["axis"] =  json.load(open("./prompts/holistic/social_biases_valid_groups.json", "r"))["axis"]
    groups_valid["bucket"] = json.load(open("./prompts/holistic/social_biases_valid_groups.json", "r"))["bucket"]
    groups_valid = pd.DataFrame.from_dict(groups_valid)

    groups_test = {}
    groups_test["axis"] =  json.load(open("./prompts/holistic/social_biases_test_groups.json", "r"))["axis"]
    groups_test["bucket"] = json.load(open("./prompts/holistic/social_biases_test_groups.json", "r"))["bucket"]
    groups_test = pd.DataFrame.from_dict(groups_test)

    for model_name in args.model_list:
        if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
            model = AutoModelWithLMHead.from_pretrained("./saved_models/cached_models/" + model_name).to(device)
            # number of heads per layer, and number of layers
            num_heads, num_layers = model.config.n_head, model.config.n_layer

        elif model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m"]:
            model = GPTNeoXForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)
            num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers

        elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
            model = GPTNeoForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)
            num_heads, num_layers = model.config.num_heads, model.config.num_layers

        elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
            model = OPTForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name).to(device)
            num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers

        if args.experiment == "compare_to_baselines":
            methods = ["random_structured", "mask_gradient_l2_structured", "magnitude_l2_structured","FASP", "ppl_only", "bias_only"]
            gammas = ["0.2","0.3","0.4","0.5","0.6", "0.7"]
            pruned_heads_ratios = np.linspace(0,0.2,11,endpoint=True)
            head_knockouts = ["None"]
        elif args.experiment == "compute_scores":
            methods = ["None"]
            gammas = ["0.5"] # the value of gamma shouldn't matter here, but 0.5 is an arbitrary value that is not None
            pruned_heads_ratios = ["0"]
            head_knockouts = range(0,int(num_heads * num_layers))
            
        groups = ["gender_and_sex","nationality", "race_ethnicity", "religion", "sexual_orientation"]
        seeds = ["1", "2", "3"]
        
        for split in ["valid", "test"]:
            for method in methods:
                for gamma in gammas:
                    for seed in seeds:
                        for prompting in ["holistic"]:
                            for pruned_heads_ratio in pruned_heads_ratios:
                                for group in groups:
                                    for head_knockout in head_knockouts:
                                        csv_directory = (
                                            "/scratch/" + args.account + "/BOLD_2/ours/seed_"
                                            + str(seed)
                                            + "/output/"
                                            + "prompt_"
                                            + str(prompting)
                                            + "_h" + str(head_knockout)   
                                            + "_" + str(split)  
                                            + "/" + str(method) + "_" + str(pruned_heads_ratio) + "_gamma" + str(gamma) + "/"                    
                                        ) 
                                        
                                        file_name = (
                                            csv_directory
                                            + model_name.replace("/", "_")
                                            + "_"
                                            + str(group)
                                            + ".csv"
                                        )     
                                        print(file_name)  
                                        
                                        if os.path.exists(file_name):
                                            print(csv_directory)
                                            df = pd.read_csv(file_name,lineterminator='\n')  
                                            if split == "valid":
                                                groups = groups_valid
                                            elif split == "test":
                                                groups = groups_test

                                            if len(df) != len(list(groups["axis"][groups["axis"] == group])):
                                                print("Something is wrong!!")
                                                continue
                                            df["axis"] = list(groups["axis"][groups["axis"] == group])
                                            df["bucket"] = list(groups["bucket"][groups["axis"] == group])

                                            for group in df["axis"].unique():
                                                bias = 0
                                                toxicity_group = df["toxicity_score"][df["axis"] == group].mean()
                                                for subgroup in df["bucket"].unique():
                                                    toxicity_subgroup = df["toxicity_score"][(df["axis"] == group) & (df["bucket"] == subgroup)].mean()
                                                    bias += abs(toxicity_group-toxicity_subgroup)

                                                ppl=df["perplexity"].mean()       
                                                percent_toxic =len(df[(df["group"] == group) & (df["toxicity_score"] > 0.5)])/(len(df[(df["group"] == group)]))*100
                                                df_all_seeds = df_all_seeds.append({'Group': group, 'Bias': bias, 'Average toxicity': toxicity_group, 'Model': model_name, 
                                                                                    'Prompting': prompting,'% of toxic output': percent_toxic, 'gamma': gamma if gamma != "None" else None,'Group': group,
                                                                                    'Head id': (head_knockout + 1) if head_knockout != "None" else None, 'Method': method, 'Pruning ratio': pruned_heads_ratio,
                                                                                    'Seed': seed, 'Split': split, 'PPL': ppl}, ignore_index = True)     



    df_all_seeds.to_csv(
        str(args.experiment)
        + "_"
        + str(args.model_list[0])
        + ".csv",
        index=False,
    ) 





