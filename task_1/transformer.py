import torch
from torch import nn 
from torch.nn import functional as F 
from numpy import sin,cos,sqrt

def layer_disturb(model):#对参数进行正态随机扰动
    if model.training and model.noise_ratio>0.0:
        bias= model.bias+torch.randn_like(model.bias)*model.noise_ratio if model.bias is not None else model.bias
        weight=model.weight+torch.randn_like(model.weight)*model.noise_ratio
        #weight = model.weight * torch.exp(torch.randn_like(model.weight) * model.noise_ratio) #另一种可选的扰动
    else:
        bias=model.bias
        weight=model.weight
    return weight,bias

def pos_encoding(context_len: int, d_model: int):#位置编码
    pe=torch.zeros(context_len,d_model)
    for pos in range(context_len):
        pe[pos]=torch.tensor([sin(pos/(1e4**(float(i)/d_model))) if i%2==0 else cos(pos/(1e4**(float(i-1)/d_model))) 
                      for i in range(d_model)]).reshape(1,d_model)
    return pe


class Linear(nn.Linear):#重写linear,layernorm,embedding，加入参数的随机扰动
    def __init__(self, *args, **kwargs):
        self.noise_ratio=kwargs.pop("noise_ratio")
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight,bias=layer_disturb(self)
        return F.linear(input,weight,bias)

class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        self.noise_ratio=kwargs.pop("noise_ratio")
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight,bias=layer_disturb(self)
        return F.layer_norm(input,self.normalized_shape,weight,bias,self.eps)

class Embedding(nn.Embedding):
    def __init__(self,*args,**kwargs):
        self.noise_ratio=kwargs.pop("noise_ratio")
        self.bias=None
        super().__init__(*args, **kwargs)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight,_=layer_disturb(self)
        return F.embedding(input,weight,self.padding_idx,self.max_norm,self.norm_type,
                           self.scale_grad_by_freq,self.sparse)

class Self_AttnHead(nn.Module):#自注意力
    def __init__(self, d_model: int, d_key: int, noise_ratio: float) -> None:
        super().__init__()
        self.d_key=d_key
        self.Wq=Linear(d_model,d_key,bias=False,noise_ratio=noise_ratio)
        self.Wk=Linear(d_model,d_key,bias=False,noise_ratio=noise_ratio)
        self.Wv=Linear(d_model,d_key,bias=False,noise_ratio=noise_ratio)
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,input: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        queries=self.Wq(input)
        keys=self.Wk(input)
        values=self.Wv(input)

        attn=torch.matmul(queries,keys.transpose(-2,-1))
        if mask is not None:
            attn.masked_fill_(mask==0,float("-inf"))
        result=torch.matmul(self.softmax(attn/sqrt(self.d_key)),values)

        return result

class Self_MultiHead(nn.Module):
    def __init__(self, d_model: int, heads: int, noise_ratio: float=0.0) -> None :
        super().__init__()
        d_key=int(d_model/heads)
        attn_heads=[Self_AttnHead(d_model,d_key,noise_ratio) for _ in range(heads)]
        self.attn_heads=nn.ModuleList(attn_heads)
        self.Wo=Linear(d_model,d_model,bias=False,noise_ratio=noise_ratio)

    def forward(self,input:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
        head_results=[head(input,mask) for head in self.attn_heads]
        result=self.Wo(torch.cat(head_results,dim=-1))

        return result

class FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, noise_ratio: float=0.0) -> None:
        super().__init__()
        self.ffn=nn.Sequential(Linear(d_model,d_hidden,bias=False,noise_ratio=noise_ratio),
                               nn.GELU(),Linear(d_hidden,d_model,bias=False,noise_ratio=noise_ratio))

    def forward(self,x:torch.Tensor):
        return self.ffn(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model:int, heads: int, dropout:float=0.1, noise_ratio=0.0) -> None:
        super().__init__()
        self.self_attn=Self_MultiHead(d_model,heads,noise_ratio)
        self.self_attn_drop=nn.Dropout(p=dropout)
        self.self_attn_norm=LayerNorm(d_model,noise_ratio=noise_ratio)
        self.ffn=FFN(d_model,4*d_model,noise_ratio=noise_ratio)
        self.ffn_drop=nn.Dropout(p=dropout)
        self.ffn_norm=LayerNorm(d_model,noise_ratio=noise_ratio)

    def forward(self,x:torch.Tensor,self_attn_mask)->torch.Tensor:
        y_1=self.self_attn(x,self_attn_mask)
        #y_1=self.self_attn_drop(y_1) #测试发现这里不要加dropout
        y_1=self.self_attn_norm(x+y_1)

        y_2=self.ffn(y_1)
        y_2=self.ffn_drop(y_2)
        y_2=self.ffn_norm(y_1+y_2)

        return y_2
    
class Decoder(nn.Module):
    def __init__(self, d_model:int, heads:int, num_blocks:int, 
                 dropout:float, noise_ratio:float) -> None:
        super().__init__()

        self.blocks=nn.ModuleList([DecoderBlock(d_model,heads,dropout=dropout,noise_ratio=noise_ratio) 
                                   for _ in range(num_blocks)])
    def forward(self,x:torch.Tensor,self_attn_mask):
        a=x
        for block in self.blocks:
            a=block(a,self_attn_mask)
        return a
    
class Transformer(nn.Module):
    def __init__(self, n_layers:int=2,n_heads:int=4,d_model:int=128,dropout:float=0.1,
                 context_len:int=4,vocab_len:int=7,noise_ratio:float=0.0) -> None:
        super().__init__()
        self.embedding=Embedding(vocab_len,d_model,noise_ratio=noise_ratio)
        self.register_buffer("position_encoding",pos_encoding(context_len,d_model))
        self.register_buffer("self_attn_mask",torch.ones([context_len,context_len]).tril())
        self.decoder = Decoder(d_model,n_heads,n_layers,dropout,noise_ratio=noise_ratio)
        self.layernorm=LayerNorm(d_model,noise_ratio=noise_ratio)
        self.linear = Linear(d_model, vocab_len, bias=False, noise_ratio=noise_ratio)

    
    def forward(self,input:torch.Tensor)->torch.Tensor:
        x=self.embedding(input)+self.position_encoding
        decoded=self.decoder(x,self.self_attn_mask)
        decoded=self.layernorm(decoded)
        y=self.linear(decoded)
        return y