import torch
from torch.utils.data import TensorDataset,random_split
import numpy as np
from numpy import ndindex
from torch import nn
from torch.nn import functional as F

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

class ModularAdditionLSTM(nn.Module):
    def __init__(self, embedding_dim:int=128, hidden_dim:int=128, num_layers:int=2, vocab_len:int=7, noise_ratio:float=0.0):
        super(ModularAdditionLSTM, self).__init__()
        self.embedding = Embedding(vocab_len, embedding_dim, noise_ratio=noise_ratio)  # +2 for op and eq
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = Linear(hidden_dim, vocab_len, noise_ratio=noise_ratio)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)  # Use the last output for prediction
        return output