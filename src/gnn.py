import torch
from torch.utils.data import TensorDataset,random_split
import numpy as np
from numpy import ndindex
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv

def layer_disturb(model):#对参数进行正态随机扰动
    if model.training and model.noise_ratio>0.0:
        bias= model.bias+torch.randn_like(model.bias)*model.noise_ratio if model.bias is not None else model.bias
        weight=model.weight+torch.randn_like(model.weight)*model.noise_ratio
        #weight = model.weight * torch.exp(torch.randn_like(model.weight) * model.noise_ratio) #另一种可选的扰动
    else:
        bias=model.bias
        weight=model.weight
    return weight,bias

class Linear(nn.Linear):#重写linear,layernorm,embedding，加入参数的随机扰动
    def __init__(self, *args, **kwargs):
        self.noise_ratio=kwargs.pop("noise_ratio")
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight,bias=layer_disturb(self)
        return F.linear(input,weight,bias)


class Embedding(nn.Embedding):
    def __init__(self,*args,**kwargs):
        self.noise_ratio=kwargs.pop("noise_ratio")
        self.bias=None
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight,_=layer_disturb(self)
        return F.embedding(input,weight,self.padding_idx,self.max_norm,self.norm_type,
                           self.scale_grad_by_freq,self.sparse)

class ModularAdditionGNN(nn.Module):
    def __init__(self, embedding_dim:int=128, hidden_dim:int=256, vocab_len:int=7, noise_ratio:float=0.0):
        super(ModularAdditionGNN, self).__init__()
        self.embedding = Embedding(vocab_len, embedding_dim, noise_ratio=noise_ratio)  # +2 for op and eq
        self.gcn1 = GCNConv(embedding_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = Linear(hidden_dim, vocab_len, noise_ratio=noise_ratio)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        edge_index = torch.tensor([[0, 1, 2], [3, 3, 3]], dtype=torch.long, device=x.device)
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        x = F.relu(self.gcn1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.gcn2(x, edge_index))
        x = self.dropout(x)
        output = self.fc(x)
        return output
