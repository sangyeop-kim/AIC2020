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
from Model_template import Model_template
from torch.utils.data import DataLoader
from tsnecuda import TSNE
import pickle
from sklearn.cluster import KMeans
from collections import Counter
from models import *
from utils import *
import os
import argparse


parser = argparse.ArgumentParser()
    
parser.add_argument('--date', type=int, default=120102, choices=[120102, 202003, 202004, 202005,
                                                                 202006, 202007])
parser.add_argument('--save', type=bool, default=True)

args = parser.parse_args()
date = args.date
save = args.save

load_file ={120102 : 'conv_attn_ae_120102/epoch=37_val_loss=0.0652',
            202003 : 'conv_attn_ae_202003/epoch=39_val_loss=0.0584',
            202004 : 'conv_attn_ae_202004/epoch=16_val_loss=0.0598',
            202005 : 'conv_attn_ae_202005/epoch=32_val_loss=0.0542',
            202006 : 'conv_attn_ae_202006/epoch=25_val_loss=0.0546',
            202007 : 'conv_attn_ae_202007/epoch=35_val_loss=0.0517'
           }

ckpt = load_file[date]


model = Conv_Attn_Autoencoder(hyperparameters)
model = model.load_model(ckpt)

front = pd.read_feather('data/%s_front.ftr' % date)
hz = pd.read_feather('data/log_%s.ftr' % date)

hidden, label, time, attn, stage = load_model_result_pkl(model, front, hz)

random = 6
tsne = TSNE(random_seed=random)
embedded = tsne.fit_transform(hidden)

model_name = model.hparams.now.split('_')[-1]

visualize_tsne(embedded, label, label, False, save, 'ori_%s_%s' %(model_name, date))


n_clusters = 10
km = KMeans(n_clusters=n_clusters, n_jobs=64, random_state=1)
km.fit(hidden)
pred = km.predict(hidden)

visualize_tsne(embedded, pred, label, True, save, '%s_%s' %(model_name, date))

if date != 202007:
    date_list = np.array(list(load_file.keys()))
    date_test = date_list[np.where(date_list == date)[0][0] + 1]

    front_test = pd.read_feather('data/%s_front.ftr' % date_test)
    hz_test = pd.read_feather('data/log_%s.ftr' % date_test)
    
    hidden_test, label_test, time_test, attn_test, stage_test = load_model_result_pkl(model, 
                                                                                  front_test, 
                                                                                  hz_test)
    
    embedded_test = tsne.fit_transform(hidden_test)
    pred_test = km.predict(hidden_test)
    
    embedded_total = np.concatenate((embedded, embedded_test), axis=0)
    pred_total = np.concatenate((pred, pred_test), axis=0)
    label_total = np.concatenate((label, label_test), axis=0)
    
    visualize_tsne(embedded_total, pred_total, label_total, True, save, '%s_%s' %(model_name,
                                                                                  date_test))