import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path


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
            "EleutherAI/gpt-j-6B",
            "meta-llama/Llama-2-7b",   
            "meta-llama/Llama-2-7b-chat-hf",    
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
        default=0.0,
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
        default=1024,
        help="Batch size for the language model.",
    )
    #128
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
        help="Whether or not to use the head scores for gender bias when reducing other biases",
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
    torch.manual_seed(args.seed)

    tox_model = torch.load("./saved_models/unbiased/unbiased.pt")
    tox_model.device = device 
    model_name = args.model.replace("/", "_")

    for split in ["valid","test"]: 
        output_dir = args.output_dir + "/prompt_" + str(args.prompting) + "_h" + str(args.head_knockout) 
        output_dir += "_" + split + "/"
        output_dir +=  str(args.method) + "_" + str(args.pruned_heads_ratio) + "_gamma" + str(args.gamma)  + "/"

        df = pd.read_csv(output_dir + model_name + "_" + args.targeted_holistic_bias + ".csv", engine="python",)
        df['generation'] = df["generation"].apply(lambda x: x.replace("\n", " ").lower())
        toxicity = []
        for i in range(int(np.ceil(len(df) / args.batch_size))):
            print(i)
            toxicity += tox_model.predict(list(df.iloc[i * args.batch_size : (i + 1) * args.batch_size]["generation"]))["toxicity"]

        df["toxicity_score_2"] = toxicity
        df.to_csv(output_dir + model_name + "_" + args.targeted_holistic_bias + "_fixed.csv",index=False,)        


            
        






        




            
