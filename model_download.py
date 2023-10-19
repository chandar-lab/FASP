from transformers import AutoModelWithLMHead,AutoTokenizer
import torch
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, GPTNeoForCausalLM, GPTNeoXTokenizerFast, BloomForCausalLM, AutoTokenizer, AutoModelWithLMHead, BloomTokenizerFast, OPTForCausalLM, GPT2Tokenizer, GPTJForCausalLM

# This file used to download the models from huggingface and save them in the cached_models folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "facebook/opt-350m", "facebook/opt-1.3b","bigscience/bloom-560m", "bigscience/bloom-1b1","gpt2", "gpt2-medium", "gpt2-large","distilgpt2", "bert-base-cased",  "bert-large-cased", "roberta-base","roberta-large"]:
    print(model_name)
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")   # Initialize tokenizer
        # number of heads per layer, and number of layers
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 
    elif model_name in ["distilroberta-base", "distilbert-base-cased", "bert-base-cased",  "bert-large-cased", "roberta-base","roberta-large"]:
        model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")   # Initialize tokenizer
        # number of heads per layer, and number of layers
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 
        
    elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        num_heads, num_layers = model.config.num_heads, model.config.num_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif model_name in ["EleutherAI/gpt-j-6B"]:
        model =  GPTJForCausalLM.from_pretrained(model_name).to(device)
        #,revision="float16", torch_dtype=torch.float16,
        tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 

    elif model_name in ["EleutherAI/gpt-neox-20b"]:
        model = GPTNeoXForCausalLM.from_pretrained(model_name)
        tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif model_name in ["facebook/opt-350m", "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b"]:
        model = OPTForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif model_name in ["bigscience/bloom-560m", "bigscience/bloom-1b1","bigscience/bloom-3b", "bigscience/bloom-7b1"]:
        model = BloomForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = BloomTokenizerFast.from_pretrained(model_name, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.n_layer
        head_dim, max_length = int(model.config.hidden_size/num_heads), 1024

    elif model_name in ["EleutherAI/pythia-70m","EleutherAI/pythia-160m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-1.4b","EleutherAI/pythia-2.8b","EleutherAI/pythia-6.9b","EleutherAI/pythia-12b"]:
        model = GPTNeoXForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    model.save_pretrained("./saved_models/cached_models/" + model_name)
    tokenizer.save_pretrained("./saved_models/cached_tokenizers/" + model_name)
