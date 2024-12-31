import torch
from torch import nn, optim
import torch.nn.functional as F 
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from math import ceil
from .dataprocess import data_generation, loss_record, multi_data_generation, cal_accuracy
from .transformer import Transformer

def linear_warmup_schedule(warmup_steps):#学习率线性warm-up
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # 在 warmup 阶段，学习率线性增加
            return float(current_step) / warmup_steps
        else:
            # 在 warmup 之后，学习率保持恒定
            return 1.0
    return lr_lambda

def transformer_fraction(fraction:float=0.5,modulus:int=31,num_sum:int=2,max_opt_times:int=50000,
                         dropout=0.1,noise_ratio:float=0.0,optim_class=torch.optim.AdamW,optim_params={}):
    device = torch.device("mps" if torch.backends.mps.is_available() 
                        else "cuda" if torch.cuda.is_available() else "cpu")
    #fraction=0.7#训练集占比
    data_num=modulus**num_sum
    train_num=int(data_num*fraction)
    valid_num=data_num-train_num
    batch_size=min(ceil(train_num/2.0),512)
    max_epoches=int(max_opt_times/ceil(train_num/batch_size))
    #train_dataset,test_dataset=data_generation(device,fraction=fraction,MODULUS:int=31,num_sum:int=2)#生成训练集，测试集
    train_dataset,test_dataset=multi_data_generation(device,fraction=fraction,MODULUS=modulus,num_sum=num_sum)#生成训练集，测试集
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    ce_loss=nn.CrossEntropyLoss()
    T_model=Transformer(context_len=2*num_sum,vocab_len=modulus+2,dropout=dropout,noise_ratio=noise_ratio).to(device)
    optimizer=optim_class(T_model.parameters(),**optim_params)
    scheduler=LambdaLR(optimizer,lr_lambda=linear_warmup_schedule(warmup_steps=10))#学习率管理
    iter=0
    logs = {}
    for tag in ["train_loss", "train_acc", "val_loss", "val_acc"]:
        logs[tag] = []
    for i in tqdm(range(max_epoches), total=max_epoches):
        for data,label in train_loader:
            T_model.train()
            T_model.zero_grad()
            output=T_model(data)#[:,-1][:,-1]
            loss=ce_loss(output[:,-1].view(-1,modulus+2),label.view(-1))#[:,-1]
            loss.backward()
            optimizer.step()
            scheduler.step()
            logs["train_loss"].append(loss.item())
            logs["train_acc"].append(cal_accuracy(output[:,-1],label)/label.size(-1))
            #train_loss[iter],train_acc[iter]=loss_record(T_model,train_loader)#误差计算
            val_loss, val_acc = loss_record(T_model,test_loader)
            logs["val_loss"].append(val_loss.item() / valid_num)
            logs["val_acc"].append(val_acc / valid_num)
            iter+=1
        #print(f'epoch {i} finished!')
        #print(train_acc[iter-1],valid_acc[iter-1])
    return logs
