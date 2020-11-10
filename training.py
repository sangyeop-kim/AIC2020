import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm, tqdm_notebook
from matplotlib import pyplot as plt
torch.cuda.set_device(device=0)
import warnings
warnings.filterwarnings('ignore')
from easydict import EasyDict
from torch.optim.lr_scheduler import StepLR
import os
import sys
from Model_template import Model_template
from utils import MyDataset
import argparse
from models import *

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', type=str, default='attn', choices=['conv', 'attn'])
    parser.add_argument('--data', type=int, default=120102, choices=[201912, 202001, 202002, 
                                                                     202003, 202004, 202005,
                                                                     202006, 202007, 120102])
    parser.add_argument('--ckpt', type=str, default=None)
    
    args = parser.parse_args()
    model = args.model
    data = args.data
    ckpt = args.ckpt
    
    front = pd.read_feather('data/%s_front.ftr' % data)
    hz = pd.read_feather('data/log_%s.ftr' % data)
    train_dataset = MyDataset(front, hz, False)
    
    if model == 'attn':
        hyperparameters = EasyDict({'lr' : 0.003,
                                'max_epochs' :50,
                                'step_size' : 10, # scheduler
                                'gamma' : 0.9, # schduler
                                'batch_size' : 128, # train_batch_size
                                'test_batch_size' : 128, # test_batch_size
                                'gpus' : [0],
                                'num_workers' : 128,
                                'auto_lr_find' : False,
                                'save_top_k' : 3,
                                'folder' : 'conv_attn_ae_%s' % data,
                                'early_stopping' : True,
                                'patience' : 5
                                })
    else:
        hyperparameters = EasyDict({'lr' : 0.001,
                            'max_epochs' :50,
                            'step_size' : 10, # scheduler
                            'gamma' : 0.9, # schduler
                            'batch_size' : 64, # train_batch_size
                            'test_batch_size' : 512, # test_batch_size
                            'gpus' : [0],
                            'num_workers' : 128,
                            'auto_lr_find' : False,
                            'save_top_k' : 3,
                            'folder' : 'convae',
                             'early_stopping' : True,
                             'patience' : 5
                            })

    if not os.path.isdir(hyperparameters['folder']) :
        os.mkdir(hyperparameters['folder'])

    
    if model == 'attn':
        model = Conv_Attn_Autoencoder(hyperparameters)
    else:
        model = Conv_Autoencoder(hyperparameters)

    if ckpt is not None:
        model = model.load_model(ckpt)
        
    model.fit(train_dataset, train_dataset)



if __name__=="__main__":
	main()