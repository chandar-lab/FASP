import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import GPTNeoForCausalLM, GPTJForCausalLM, AutoModelWithLMHead
               
# This file used to download the models from huggingface and save them in the cached_models folder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for model_name in ["gpt2","distilgpt2","EleutherAI/gpt-neo-125M","EleutherAI/gpt-neo-1.3B","EleutherAI/gpt-neo-2.7B","EleutherAI/gpt-j-6B","meta-llama/Llama-2-7b-chat-hf"]:
    print(model_name)
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2",  "gpt2-xl"]:
        model = AutoModelWithLMHead.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")   # Initialize tokenizer
        # number of heads per layer, and number of layers
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 
        
    elif model_name in ["EleutherAI/gpt-neo-125M", "EleutherAI/gpt-neo-1.3B", "EleutherAI/gpt-neo-2.7B"]:
        model = GPTNeoForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name, padding_side="left")
        num_heads, num_layers = model.config.num_heads, model.config.num_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    elif model_name in ["EleutherAI/gpt-j-6B"]:
        model =  GPTJForCausalLM.from_pretrained(model_name,revision="float16", torch_dtype=torch.float16,).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left")
        num_heads, num_layers = model.config.n_head, model.config.n_layer
        head_dim, max_length = int(model.config.n_embd/num_heads), model.config.n_positions 

    elif model_name in ["meta-llama/Llama-2-7b-chat-hf"]:
        print("./saved_models/cached_models/" + model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        model = LlamaForCausalLM.from_pretrained(model_name, token= 'ENTER_YOUR_TOKEN_HERE').to(device)
        num_heads, num_layers = model.config.num_attention_heads, model.config.num_hidden_layers
        head_dim, max_length = int(model.config.hidden_size/num_heads), model.config.max_position_embeddings 

    model.save_pretrained("./saved_models/cached_models/" + model_name)
    tokenizer.save_pretrained("./saved_models/cached_tokenizers/" + model_name)
    print("./saved_models/cached_models/" + model_name)
