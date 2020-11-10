import os
import sys
from Model_template import Model_template
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR




class Conv_Autoencoder(Model_template):
    def __init__(self, hyperperameters):
        super().__init__(hyperperameters)
        self.loss = nn.MSELoss()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, 10, 2),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.Conv1d(16, 4, 5, 3),
            nn.MaxPool1d(2),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(1660, 128),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 9996-15+1),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        hidden = self.encoder(x)
        x = self.decoder(hidden)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    
class Conv_Attn_Autoencoder(Model_template):
    def __init__(self, hyperperameters):
        super().__init__(hyperperameters)
        self.loss = nn.MSELoss()
        
        self.embedding = nn.Sequential(
            nn.Conv1d(1, 32, 5, 3, 2),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 5, 3, 2),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 5, 3, 2),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 5, 3),
            nn.ReLU(True),
        )
        
        self.expansion = nn.parameter.Parameter(torch.randn(32, 500))
        self.weight = nn.parameter.Parameter(torch.randn(500, 1))
        
        self.encoder_nn = nn.Sequential(
            nn.Flatten(),
            nn.Linear(122*4, 128),
            nn.Linear(128, 64)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 9996-15+1),
        )

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        x = self.embedding(x)
        x = torch.transpose(x, -1, -2)
        attn = F.relu(torch.matmul(x, self.expansion))
        self.attn = F.softmax(torch.matmul(attn, self.weight), 1)
        x = F.avg_pool2d(x, (1, x.shape[2]//4))
        x = torch.squeeze(x, -1)
        x = torch.mul(x, self.attn)
        self.hidden = self.encoder_nn(x)
        try:
            x = self.decoder(self.hidden)
            return x
        except:
            return self.hidden
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = StepLR(optimizer, step_size=self.hparams.step_size, 
                           gamma=self.hparams.gamma)
        
        return [optimizer], [scheduler]
    
    