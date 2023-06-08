import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from main import parse_args
from utils import get_head_dim_idx



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_all_seeds = pd.DataFrame()
args = parse_args()
# for model in ["EleutherAI_gpt-neo-125M"]:
#   for prompting in ["M_CM", "CF_F","MCF_FCM", "M_F"]:
#     for famous_names in [False,True]:
#         for seed in ["1", "2", "3", "4", "5"]:
#                     csv_directory = (
#                         "/scratch/abdel1/BOLD_2/ours/seed_"
#                         + str(seed)
#                         + "/output/"
#                         + "prompting_"
#                         + str(prompting)
#                         + "_" + "famous_names_"
#                         + str(famous_names)                        
#                         + "/"
#                     ) 
#                     file_name = (
#                         csv_directory
#                         + model
#                         + "_outputs"
#                         + ".csv"
#                     )       
                    
#                     if os.path.exists(file_name):
#                         print(csv_directory)
#                         df = pd.read_csv(file_name,lineterminator='\n')       
#                         for gender in ["Male", "Female"]:
#                             # print(df)
#                             if gender == "Male":
#                                 toxicity_percentage=len(df[(df["domain"] == "gender") & (df["group"] == "American_actors") & (df["toxicity_score"] > 0.5)])/(len(df[(df["domain"] == "gender") & (df["group"] == "American_actors")]))*100
#                             elif gender == "Female":
#                                 toxicity_percentage=len(df[(df["domain"] == "gender") & (df["group"] == "American_actresses") & (df["toxicity_score"] > 0.5)])/(len(df[(df["domain"] == "gender") & (df["group"] == "American_actresses")]))*100
#                             df_all_seeds = df_all_seeds.append({'Gender': gender,'Famous names': str(famous_names),'Gender': gender,'Prompting': str(prompting),'% of toxic output': toxicity_percentage}, ignore_index = True)           


# # print(df)
# # df_all_seeds = df_all_seeds.rename(columns={"Counterfactual_prompts": "Counterfactual prompts"})
# sns.set(style='white')
# plt.figure()
# sns.catplot(x="Prompting", y='% of toxic output', kind="bar", col="Famous names", hue = 'Gender',  data = df_all_seeds)
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds" + '.pdf')
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds" + '.png')


# df_all_seeds.to_csv(
#     "/scratch/abdel1/BOLD_2/ours/seed_1/output/"
#     + "output_all_seeds.csv",
#     index=False,
# )

# 
# 

def compute_ppl(model_name, beta, intraprocessing_method):
    model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name)

    if intraprocessing_method == "temperature_scaling":
        for layer_id in range(model.config.num_hidden_layers):
            for name, para in model.named_parameters():
                if "transformer.h." + str(layer_id) + ".attn.attention.q_proj.weight" in name:
                    para.data *= float(beta)    

    elif intraprocessing_method == "random_perturbation":
        for name, para in model.named_parameters():
            para.data *= (torch.randn_like(para) * args.random_perturbation_std + args.random_perturbation_mean)

    tokenizer = AutoTokenizer.from_pretrained("./saved_models/cached_tokenizers/" + model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model=model.to(device)

    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    names = []    
    with open("/scratch/abdel1/BOLD_2/ours/seed_1/model/wikitext-2-raw-v1.txt", 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line

            # add current item to the list
            names.append(x)

    encodings = tokenizer("".join(names) , return_tensors="pt")

    max_length = 2048
    #model.config.n_positions
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            print("shoof de", input_ids.shape)
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = round(float(torch.exp(torch.stack(nlls).mean())), 3)
    print(ppl)

    return ppl
# 
#"woman", "black"
#"temperature_scaling","random_perturbation"
#"None","0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.8", "1.9","2.0", "3.0", "4.0", "5.0"
#,"EleutherAI/gpt-neo-2.7B"
#for prompting in ["BOLD"]:
head_knockout = "None"            
for model_name in ["EleutherAI/gpt-neo-125M"]:
    for split in ["valid"]:
        for method in ["magnitude", "random", "ours_ppl", "ours_magnitude"]:
        # for intraprocessing_method in ["random_perturbation","temperature_scaling"]:
            for beta in ["None"]:
                # ppl = compute_ppl(model_name, beta, intraprocessing_method)
                # for group in ["gender", "political_ideology", "profession", "race", "religious_ideology"]:
                for group in ["woman","non-binary", "asian","black","man","middle-aged","white","adult","hispanic","young","child","pacific-islander","native-american","senior"]:
                    for seed in ["1", "2", "3"]:
                        for prompting in ["PANDA"]:
                            for pruned_heads_ratio in ["0.0"]:
                            # for pruned_heads_ratio in ["0.007", "0.014", "0.025", "0.03", "0.035", "0.042", "0.049", "0.056", "0.063"]:
                                for alpha in ["0.0", "0.1", "0.2"]:
                                # for alpha in ["0.0", "0.25", "0.5", "0.75", "1.0"]:
                                
                                # for head_knockout in ["None"]:
                                # for head_knockout in range(143,144):
                                    print("group:", str(group), ", seed:", str(seed), ", head:", str(head_knockout), ", method:", str(method),", pruning_ratio:", str(pruned_heads_ratio),", alpha:", str(alpha))
                                    csv_directory = (
                                        "/scratch/abdel1/BOLD_2/ours/seed_"
                                        + str(seed)
                                        + "/output/"
                                        + "prompt_"
                                        + str(prompting)
                                        # + "_" + str(group)
                                        + "_" + str(beta)
                                        + "_False" 
                                        # + "_" + str(intraprocessing_method=="random_perturbation")
                                        + "_h" + str(head_knockout)   
                                        + "_" + str(split)  
                                        + "/" + str(method) + "_" + str(pruned_heads_ratio) + "_alpha" + str(alpha) + "/"                    
                                    ) 
                                    file_name = (
                                        csv_directory
                                        + model_name.replace("/", "_")
                                        + "_outputs"
                                        + ".csv"
                                    )       

                                    if model_name == "EleutherAI/gpt-neo-1.3B":
                                        model_type = "GPT-Neo"
                                        number_of_parameters = "1.3B"
                                    elif model_name == "EleutherAI/gpt-neo-2.7B":
                                        model_type = "GPT-Neo"
                                        number_of_parameters = "2.7B"
                                    elif model_name == "EleutherAI/gpt-neo-125M":
                                        model_type = "GPT-Neo"
                                        number_of_parameters = "125M"

                                    print(file_name)
                                    # model = AutoModelForCausalLM.from_pretrained("./saved_models/cached_models/" + model_name)
                                    # _, _, layer_id = get_head_dim_idx(head_knockout, model) 
                                    # head_id = head_knockout - (layer_id + 1) * model.config.num_attention_heads
                                    
                                    if os.path.exists(file_name):
                                        print(csv_directory)
                                        df = pd.read_csv(file_name,lineterminator='\n')       

                                        if prompting == "BOLD":
                                            for group_id in range(len(df["group"].unique())):
                                                subgroup=df["group"].unique()[group_id]   
                                                ppl=df["perplexity"].mean()       
                                                percent_toxic =len(df[(df["group"] == subgroup) & (df["toxicity_score"] > 0.5)])/(len(df[(df["group"] == subgroup)]))*100
                                                percent_pos_sentiment = len(df[(df["group"] == subgroup) & (df["sentiment_score"] > 0.5)])/(len(df[(df["group"] == subgroup)]))*100
                                                percent_neg_sentiment = len(df[(df["group"] == subgroup) & (df["sentiment_score"] < -0.5)])/(len(df[(df["group"] == subgroup)]))*100 
                                                percent_neutral_sentiment = len(df[(df["group"] == subgroup) & (df["sentiment_score"] > -0.5) & (df["sentiment_score"] < 0.5)])/(len(df[(df["group"] == subgroup)]))*100                        
                                                df_all_seeds = df_all_seeds.append({'Group': group, 'Model': model_type, 'Number of parameters': number_of_parameters, 'Subgroup': str(subgroup),
                                                                                    '% of neutral sentiment': percent_neutral_sentiment, 'Prompting': prompting,'% of toxic output': percent_toxic,
                                                                                    '% of negative sentiment': percent_neg_sentiment, '% of positive sentiment': percent_pos_sentiment, 
                                                                                    'Seed': seed, 'Beta': beta, 'PPL': ppl}, ignore_index = True)           

                                        elif prompting == "PANDA": 
                                                perturbed_prompts=df[(df["group"] == group) & (df["prompt_type"] == "perturbed")]
                                                original_prompts=df[(df["group"] == group) & (df["prompt_type"] == "original")]
                                                ppl=df["perplexity"].mean()
                                                

                                                perc_change_toxicity = (perturbed_prompts["toxicity_score"].mean()-original_prompts["toxicity_score"].mean())/(original_prompts["toxicity_score"].mean())*100
                                                perc_change_sentiment = (perturbed_prompts["sentiment_score"].mean()-original_prompts["sentiment_score"].mean())/(original_prompts["sentiment_score"].mean())*100
                                                df_all_seeds = df_all_seeds.append({'Group': group, 'Model': model_type, 'Number of parameters': number_of_parameters, 'Split': split, 
                                                                                    '% of change in sentiment': perc_change_sentiment, 'Prompting': prompting,'% of change in toxicity': perc_change_toxicity,
                                                                                    "Method": method, "alpha": alpha, "pruned_heads_ratio": pruned_heads_ratio, "Split": split,
                                                                                    'Head id': (head_knockout + 1) if head_knockout != "None" else None, 'Seed': seed, 'Beta': beta, 'PPL': ppl}, ignore_index = True)           




# df_all_seeds.to_csv(
#     "/scratch/abdel1/BOLD_2/ours/seed_1/output/"
#     + "output_all_seeds_BOLD" + "_EAT_rp.csv",
#     index=False,
# )
df_all_seeds.to_csv(
    "/scratch/abdel1/BOLD_2/ours/seed_1/output/"
    + "output_all_seeds_PANDA_pruning_methods_22" + ".csv",
    index=False,
)

# print(df)
#, col='Number of parameters',
# df_all_seeds = df_all_seeds.rename(columns={"Counterfactual_prompts": "Counterfactual prompts"})

# df_all_seeds = pd.read_csv("/scratch/abdel1/BOLD_2/ours/seed_1/output/"
#     + "output_all_seeds_BOLD.csv",engine="python",
#     error_bad_lines=False,
# )
# df_all_seeds = df_all_seeds[(df_all_seeds["Number of parameters"] == "1.3B") & (df_all_seeds["Beta"] != 0.0) & (df_all_seeds["Beta"] != 0.1) & (df_all_seeds["Beta"] != 0.2) & (df_all_seeds["Beta"] != 0.3) & (df_all_seeds["Beta"] != 0.4)]
# sns.set(style='white')
# plt.figure()
# sns.catplot(x="Group", y='% of negative sentiment', kind="bar", hue = 'Beta',  data = df_all_seeds)
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_neg_sent" + '.pdf')
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_neg_sent" + '.png')

# sns.catplot(x="Group", y='% of positive sentiment', kind="bar", hue = 'Beta',  data = df_all_seeds)
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_pos_sent" + '.pdf')
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_pos_sent" + '.png')

# sns.catplot(x="Group", y='% of toxic output', kind="bar", hue = 'Beta',  data = df_all_seeds)
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_tox" + '.pdf')
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_tox" + '.png')

# sns.catplot(x="Group", y='% of neutral sentiment', kind="bar", hue = 'Beta',  data = df_all_seeds)
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_neutral_sent" + '.pdf')
# plt.savefig("/scratch/abdel1/BOLD_2/ours/seed_1/output/output_all_seeds_neutral_sent" + '.png')




