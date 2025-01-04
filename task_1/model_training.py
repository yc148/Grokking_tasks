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

def perturb_dataset(dataset, args):
    all_indices = dataset.indices
    perturbed_indices = all_indices[:int(len(all_indices) * args.perturb_ratio)]
    dataset.dataset.tensors[1][perturbed_indices] = (dataset.dataset.tensors[1][perturbed_indices] + torch.randint(high=args.prime, size=(len(perturbed_indices),), device=args.device)) % args.prime
    return dataset

def get_scale_logs(value_list):
    l1_norm = 0.0
    l2_norm = 0.0
    max_norm = 0.0
    for value in value_list:
        l1_norm += torch.abs(value).sum().item()
        l2_norm += torch.sqrt((value ** 2).sum()).item()
        max_norm += torch.abs(value).max().item()
    return {
        "l1": l1_norm,
        "l2": l2_norm,
        "linf": max_norm,
    }

def transformer_fraction(args,fraction:float=0.5,modulus:int=31,num_sum:int=2,max_opt_times:int=50000,
                         dropout=0.1,noise_ratio:float=0.0,optim_class=torch.optim.AdamW,optim_params={}):
    device = args.device
    #fraction=0.7#训练集占比
    data_num=modulus**num_sum
    train_num=int(data_num*fraction)
    valid_num=data_num-train_num
    batch_size=min(ceil(train_num/2.0),512)
    max_epoches=int(max_opt_times/ceil(train_num/batch_size))
    #train_dataset,test_dataset=data_generation(device,fraction=fraction,MODULUS:int=31,num_sum:int=2)#生成训练集，测试集
    train_dataset,test_dataset=multi_data_generation(device,fraction=fraction,MODULUS=modulus,num_sum=num_sum)#生成训练集，测试集
    if args.perturb_ratio > 0:
        train_dataset = perturb_dataset(train_dataset, args)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
    ce_loss=nn.CrossEntropyLoss()
    T_model=Transformer(context_len=2*num_sum,vocab_len=modulus+2,dropout=dropout,noise_ratio=noise_ratio).to(device)
    optimizer=optim_class(T_model.parameters(),**optim_params)
    scheduler=LambdaLR(optimizer,lr_lambda=linear_warmup_schedule(warmup_steps=10))#学习率管理
    iter=0
    logs = {}
    for tag in ["train_loss", "train_acc", "val_loss", "val_acc", "step", "wl1", "wl2", "wlinf", "gl1", "gl2", "glinf"]:
        logs[tag] = []
    for i in tqdm(range(max_epoches), total=max_epoches):
        for data,label in train_loader:
            T_model.train()
            T_model.zero_grad()
            output=T_model(data)#[:,-1][:,-1]
            loss=ce_loss(output[:,-1].view(-1,modulus+2),label.view(-1))#[:,-1]
            loss.backward()
            if args.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(T_model.parameters(), args.max_grad_norm)
            if iter % args.log_every == 0:
                grad_logs = get_scale_logs([param.grad for param in T_model.parameters()])
                for key in ["l1", "l2", "linf"]:
                    logs[f"g{key}"].append(grad_logs[key])
                weight_logs = get_scale_logs([param for param in T_model.parameters()])
                for key in ["l1", "l2", "linf"]:
                    logs[f"w{key}"].append(weight_logs[key])
            optimizer.step()
            scheduler.step()
            if iter % args.log_every == 0:
                val_loss, val_acc = loss_record(T_model,test_loader)
                logs["train_loss"].append(loss.item())
                logs["train_acc"].append(cal_accuracy(output[:,-1],label)/label.size(-1))
                logs["val_loss"].append(val_loss.item() / valid_num)
                logs["val_acc"].append(val_acc / valid_num)
                logs["step"].append(iter)
            iter+=1
        #print(f'epoch {i} finished!')
        #print(train_acc[iter-1],valid_acc[iter-1])
    return logs
