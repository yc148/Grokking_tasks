import math
import random
import torch
from torch import nn
from torch.utils.data import TensorDataset,random_split
import numpy as np
from numpy import ndindex
#MODULUS=11
#DATA_NUM=MODULUS**3
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
def data_generation(device,fraction:float=0.5,MODULUS:int=31,num_sum:int=2):#生成a+b=c(mod p)的数据集
    DATA_NUM=MODULUS**num_sum
    data=torch.zeros(DATA_NUM,4,dtype=torch.long)
    label=torch.zeros(DATA_NUM,4)
    index=0
    for a in range(MODULUS):
        for b in range(MODULUS):
            c=(a+b)%MODULUS
            data[index]=torch.tensor([a,MODULUS,b,MODULUS+1],dtype=torch.long)
            label[index]=torch.tensor([MODULUS,b,MODULUS+1,c],dtype=torch.long)
            index+=1
    data=data.to(device)
    label=label.to(device)
    dataset=TensorDataset(data,label)#生成可以被dataloader导入的训练集
    train_size=int(fraction*DATA_NUM)
    test_size=DATA_NUM-train_size
    train_data,test_data=random_split(dataset,[train_size,test_size])#随机分划
    return train_data,test_data

def cal_accuracy(output,label):
    predict=torch.argmax(output,dim=-1)
    return (predict==label).sum().item()

def loss_record(model,dataset):#calculate the training/validation loss and accuracy after each epoch
    model.eval()
    loss=0.0
    accuracy=0.0
    lossfun_ce=nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data,label in dataset:
            #data,label=data.to(device),label.to(device)
            #data=data.reshape(-1,input_size)[:,-1][:,-1]
            output=model(data)
            #print(output.size(),label.size())
            loss+=lossfun_ce(output[:,-1],label)#loss*label.size(0)[:,-1]
            accuracy+=cal_accuracy(output[:,-1],label)#accuracy[:,-1]
    return loss,accuracy

#import itertools
def multi_data_generation(device,fraction:float=0.5,MODULUS:int=31,num_sum:int=2):
    DATA_NUM=MODULUS**num_sum
    equation=torch.zeros([MODULUS**num_sum,2*num_sum+1],dtype=torch.long)
    eq=0
    index=np.zeros(np.ones(num_sum,dtype=int)*MODULUS)
    #index=itertools.product(range(MODULUS), repeat=num_added)
    #ori_data=torch.zeros(index)
    for idx in ndindex((MODULUS,)*num_sum):
        sum=0
        '''
        if sum==0:
            print(index[2])
            break
        '''
        for iter in range(num_sum):
            i=idx[iter]
            equation[eq,2*iter]=i
            equation[eq,2*iter+1]=MODULUS
            sum+=i
        equation[eq,2*num_sum-1]=MODULUS+1
        equation[eq,2*num_sum]=sum%MODULUS
        eq+=1
    data=equation[:,:-1]
    label=equation[:,-1]
    data=data.to(device)
    label=label.to(device)
    dataset=TensorDataset(data,label)#生成可以被dataloader导入的训练集
    #print(data.size(),label.size(),DATA_NUM)
    train_size=int(fraction*DATA_NUM)
    test_size=DATA_NUM-train_size
    #print(train_size,test_size)
    train_data,test_data=random_split(dataset,[train_size,test_size])#随机分划
    return train_data,test_data