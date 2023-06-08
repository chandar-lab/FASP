import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
softmax = torch.nn.Softmax(dim=1).to(device)


def get_head_dim_idx(head, model):
 layer = int(head/model.config.num_attention_heads)
 head_dim = int(model.config.hidden_size/model.config.num_attention_heads)
 start = head_dim*(head - layer*model.config.num_attention_heads)
 end = head_dim*(head - (layer*model.config.num_attention_heads) + 1)
 return start, end, layer