import torch
import json
import pickle
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
# model.save_pretrained("./saved_models/cached_models/EleutherAI/gpt-neo-1.3B")

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
# tokenizer.save_pretrained("./saved_models/cached_tokenizers/EleutherAI/gpt-neo-1.3B")

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B")
# model.save_pretrained("./saved_models/cached_models/EleutherAI/gpt-neo-2.7B")

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
# tokenizer.save_pretrained("./saved_models/cached_tokenizers/EleutherAI/gpt-neo-2.7B")

# for model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]:
#     model = GPT2LMHeadModel.from_pretrained(model_name)
#     model.save_pretrained("./saved_models/cached_models/" + model_name)

#     tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
#     tokenizer.save_pretrained("./saved_models/cached_tokenizers/" + model_name)

# train_dataset=torch.load(
#         "./prompts/panda/train_dataset.pt",
#     )

# valid_dataset=torch.load(
#         "./prompts/panda/valid_dataset.pt",
#     )

# with open('./prompts/panda/train_dataset_dict.json', 'r') as f:
#   train_dataset = json.load(f)

# with open('./prompts/panda/valid_dataset_dict.json', 'r') as f:
#   valid_dataset = json.load(f)

# identity_groups = list(dict.fromkeys(train_dataset["target_attribute"]))

# dictionary={}
# for identity_group in identity_groups:
#   dictionary[identity_group]={}
  
# for i in range(len(train_dataset)):
#   print(i)
#   dictionary[train_dataset["target_attribute"][i]][train_dataset["selected_word"][i]]=[[train_dataset["original"][i]][0],[train_dataset["perturbed"][i]][0]]

# for i in range(len(valid_dataset)):
#   print(i)
#   dictionary[valid_dataset["target_attribute"][i]][valid_dataset["selected_word"][i]]=[[valid_dataset["original"][i]][0],[valid_dataset["perturbed"][i]][0]]

# json_object = json.dumps(dictionary, indent=4)
 
# with open("./prompts/panda/social_biases.json", "w") as outfile:
#     outfile.write(json_object)

# try:
#     import cPickle as pickle
# except ImportError:  # Python 3.x
#     import pickle

with open('./prompts/PANDA/train_dataset.p', 'rb') as fp:
    train_dataset = pickle.load(fp)

with open('./prompts/PANDA/valid_dataset.p', 'rb') as fp:
    valid_dataset = pickle.load(fp)
############################################
identity_groups = list(dict.fromkeys(train_dataset["target_attribute"]))

dictionary={}
for identity_group in identity_groups:
  dictionary[identity_group]={}
  dictionary[identity_group]["original_and_perturbed"] = []
  
for i in range(len(train_dataset["original"])):
  print(i)
  dictionary[train_dataset["target_attribute"][i]]["original_and_perturbed"] += [[train_dataset["original"][i]][0],[train_dataset["perturbed"][i]][0]]


json_object = json.dumps(dictionary, indent=4)
 
# Writing to sample.json
with open("./prompts/panda/social_biases_train.json", "w") as outfile:
    outfile.write(json_object)
############################################
identity_groups = list(dict.fromkeys(valid_dataset["target_attribute"]))

dictionary={}
for identity_group in identity_groups:
  dictionary[identity_group]={}
  dictionary[identity_group]["original_and_perturbed"] = []

for i in range(len(valid_dataset["original"])):
  print(i)
  dictionary[valid_dataset["target_attribute"][i]]["original_and_perturbed"] += [[valid_dataset["original"][i]][0],[valid_dataset["perturbed"][i]][0]]

json_object = json.dumps(dictionary, indent=4)
 
# Writing to sample.json
with open("./prompts/panda/social_biases_valid.json", "w") as outfile:
    outfile.write(json_object)