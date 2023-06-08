import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_head_dim_idx(head, model):
    """
    Get the start and end indices of the head to be pruned using the model's hidden states.
    Args:
        head: int
        model: transformers.PreTrainedModel
    Returns:
        start: int
        end: int
        layer: int
    """
    
    layer = int(head/model.config.num_attention_heads)
    head_dim = int(model.config.hidden_size/model.config.num_attention_heads)
    start = head_dim*(head - layer*model.config.num_attention_heads)
    end = head_dim*(head - (layer*model.config.num_attention_heads) + 1)
    return start, end, layer