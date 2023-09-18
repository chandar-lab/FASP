
def parameters_to_prune(model_name, model):
    """
    Get the parameters to be pruned. Some parts are taken from https://github.com/sai-prasanna/bert-experiments/blob/master/src/run_glue.py
    Args:
        model_name: str
        model: transformers.PreTrainedModel
    Returns:
        parameters_to_prune: list
    """

    parameters_to_prune = []

    
    if model_name in ["gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"]:
        layers = model.base_model.h
        
    for layer in layers:
        parameters = [
            (layer.attn.c_attn, 'weight'),
            (layer.attn.c_attn, 'bias'),
            (layer.attn.c_proj, 'weight'),
            (layer.attn.c_proj, 'bias'),   
            (layer.mlp.c_fc, 'weight'),
            (layer.mlp.c_fc, 'bias'),
            (layer.mlp.c_proj, 'weight'),
            (layer.mlp.c_proj, 'bias'),                        
        ]
        parameters_to_prune.extend(parameters)

    return parameters_to_prune

