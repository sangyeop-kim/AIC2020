import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import pickle



# sample hyperparameter
hyperparameters = EasyDict({'lr' : 0.001,
                            'max_epochs' :30,
                            'step_size' : 10, # scheduler
                            'gamma' : 0.9, # schduler
                            'batch_size' : 64, # train_batch_size
                            'test_batch_size' : 512, # test_batch_size
                            'gpus' : [0],
                            'num_workers' : 128,
                            'auto_lr_find' : False,
                            'save_top_k' : 3,
                            'folder' : 'best_model',#'canv_arrn_ae',
                             'early_stopping' : True,
                             'patience' : 5
                            }) 


class MyDataset(Dataset):
    def __init__(self, front, hz, supervised):
        self.supervised = supervised
        self.degc = self.__normalize(front['degc'])
        self.label = front['label']
        self.date = front['meastime']
        self.stage = front['stage']
        self.hz = hz
    
    def __normalize(self, x):
        mean = x.mean()
        std = x.std() 
        return ((x - mean) / std).astype(np.float32)
    
    def __getitem__(self, index):
        degc = self.degc[index].reshape(-1)
        # 15!
        Hz = self.hz.iloc[index,15:].values
        Hz = self.__normalize(Hz)
        
        if self.supervised:
            label = self.label[index]
            date = self.date[index]
            stage = self.stage[index]
            return np.concatenate((degc, Hz)), label, date, stage
        return np.concatenate((degc, Hz)), np.concatenate((degc, Hz))
    
    def __len__(self):
        return len(self.label)

    
def return_reconstruction(model, front, hz, day=-1):
    if day != -1:
        day_f = front[front['meastime'].apply(lambda x : int(str(x)[-8:-6])) == day].reset_index(drop=True)
        day_h = hz[front['meastime'].apply(lambda x : int(str(x)[-8:-6])) == day].reset_index(drop=True)
    else:
        day_f = front
        day_h = hz
    
    if len(day_h) == 0:
        raise Exception(front['meastime'].apply(lambda x : int(str(x)[-8:-6])).unique())
    
    train_dataset = MyDataset(day_f, day_h, True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, 
                                  num_workers=64)
    
    loss = []
    label_list = []
    time_list = []
    stage_list = []
    
    for data, label, time, stage in tqdm(train_dataloader):
        pred = model(data)
        loss.append((torch.sum((data -  pred) ** 2, dim=1) / data.shape[-1]).detach().numpy())
        label_list.append(label.detach().numpy())
        time_list.append(time.detach().numpy())
        stage_list.append(stage)
    time_list = np.concatenate(time_list)
    label_list = np.concatenate(label_list)
    loss = np.concatenate(loss)
    stage_list = np.concatenate(stage_list)
    
    return loss, label_list, time_list, stage_list
    
    
    
def return_hidden(model, input_, tf):

    x = torch.unsqueeze(input_, dim=1)
    if tf:
        return model.encoder(x).detach().numpy(), None
    else:
        x = torch.squeeze(x, 1)
        return model(x).detach().numpy(), torch.squeeze(model.attn, -1).detach().numpy()
    
    
    
def hidden_data(model, front, hz):
    train_dataset = MyDataset(front, hz, True)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=False, num_workers=64)

    hidden_list = []
    label_list = []
    time_list = []
    attn_list = []
    stage_list = []
    try:
        model.encoder
        tf = True
    except:
        try:
            del model.decoder
        except:
            pass
        tf = False
        
    for data, label, time, stage in tqdm(train_dataloader):
        hidden, attn = return_hidden(model, data, tf)

        hidden_list.append(hidden)
        label_list.append(label)
        time_list.append(time)
        attn_list.append(attn)
        stage_list.append(stage)
        
    hidden = np.concatenate(hidden_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    time = np.concatenate(time_list, axis=0)
    try:
        attn = np.concatenate(attn_list, axis=0)
    except:
        attn = None
    stage = np.concatenate(stage_list, axis=0)
    return hidden, label, time, attn, stage



def visualize_tsne(embedded, cluster, label=None, visualize_cluster=False, save=False, name=''):
    folder = 'tsne_visualization/'
    if not os.path.isdir(folder):
        os.mkdir(folder)
        
    plt.figure(figsize=(20,20))
    plt.scatter(embedded[:, 0], embedded[:, 1], s = 1, c = cluster, alpha=0.5)
    
    if label is not None:
        plt.scatter(embedded[label==1, 0], embedded[label==1, 1], s = 200, c = 'r', alpha=1)
    
    if save:
        plt.savefig(folder+name)
        plt.close()
        print('successfully saved : %s' % (folder+name))
    else:
        plt.show()
    
    if visualize_cluster:
        for i in np.unique(cluster):
            plt.figure(figsize=(20,20))
            plt.title(str(i) + ' cluster', fontsize=30)
            plt.scatter(embedded[cluster == i, 0], embedded[cluster == i, 1], s = 3, alpha=1)
            if label is not None:
                idx = (cluster == i) & (label==1)
                plt.scatter(embedded[idx, 0], embedded[idx==1, 1], s = 200, c = 'r')
            plt.ylim((-38,38))
            plt.xlim((-38,38))
            
            if save:
                plt.savefig(folder+name+'_%s'%i)
                plt.close()
                print('successfully saved : %s'%(folder+name+'_%s'%i))
            else:
                plt.show()
    
    
def load_model_result_pkl(model, front, hz):
    date = str(front['meastime'].iloc[0])[:6]
    if date == 201912:
        date = 120102
    model_name = model.hparams.now.split('_')[-1]
    
    if not os.path.isfile('saved_pkl/model_%s_data_%s' % (model_name, date)):

        hidden, label, time, attn, stage = hidden_data(model, front, hz)

        dict_model_result = {}
        dict_model_result['hidden'] = hidden
        dict_model_result['label'] = label
        dict_model_result['time'] = time
        dict_model_result['attn'] = attn
        dict_model_result['stage'] = stage


        if not os.path.isdir('saved_pkl'):
            os.mkdir('saved_pkl')

        with open('saved_pkl/model_%s_data_%s' % (model_name, date), 'wb') as f:
            pickle.dump(dict_model_result, f)

    else:
        print('saved_pkl/hidden_pkl_%s already exists' % date)
        with open('saved_pkl/model_%s_data_%s' % (model_name, date), 'rb') as f:
            dict_model_result = pickle.load(f)
            hidden = dict_model_result['hidden']
            label = dict_model_result['label']
            time = dict_model_result['time']
            attn = dict_model_result['attn']
            stage = dict_model_result['stage']
            
    return hidden, label, time, attn, stage