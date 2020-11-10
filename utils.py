import numpy as np
from torch.utils.data import Dataset

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