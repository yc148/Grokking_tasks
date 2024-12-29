from .model_training import transformer_fraction
from .dataprocess import set_seed
from matplotlib import pyplot as plt
from numpy import log10
import torch
#import os

def trial_1(save_path,max_opt=[15000,3000,2000],modulus=97,num_sum=2,noise_ratio=0.001):
    seed=42
    set_seed(seed)
    max_opt_1=max_opt[0]
    max_opt_2=max_opt[1]
    max_opt_3=max_opt[2]

    optim_params={'lr': 1e-3, 'betas': (0.9, 0.98), 'weight_decay': 1.0}

    train_err_1,train_acc_1,valid_err_1,valid_acc_1=transformer_fraction(fraction=0.3,modulus=modulus,num_sum=num_sum,
                                                max_opt_times=max_opt_1,noise_ratio=noise_ratio,optim_params=optim_params)
    train_err_2,train_acc_2,valid_err_2,valid_acc_2=transformer_fraction(fraction=0.5,modulus=modulus,num_sum=num_sum,
                                                max_opt_times=max_opt_2,noise_ratio=noise_ratio,optim_params=optim_params)
    train_err_3,train_acc_3,valid_err_3,valid_acc_3=transformer_fraction(fraction=0.7,modulus=modulus,num_sum=num_sum,
                                                max_opt_times=max_opt_3,noise_ratio=noise_ratio,optim_params=optim_params)
    torch.save({'train_err_0_3': train_err_1, 'train_acc_0_3': train_acc_1,
            'valid_err_0_3': valid_err_1, 'valid_acc_0_3': valid_acc_1, 
            'train_err_0_5': train_err_2, 'train_acc_0_5': train_acc_2,
            'valid_err_0_5': valid_err_2, 'valid_acc_0_5': valid_acc_2,
            'train_err_0_7': train_err_3, 'train_acc_0_7': train_acc_3,
            'valid_err_0_7': valid_err_3, 'valid_acc_0_7': valid_acc_3}, save_path)

trial_1('./task_1/q1.pt')
trial_1('./task_1/q4_3.pt',modulus=23,num_sum=3)
trial_1('./task_1/q4_4.pt',modulus=11,num_sum=4)
