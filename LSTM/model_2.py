import torch.nn as nn
import torch
import numpy as np
from data import Wavset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle

class LSTM(nn.Module):
    def __init__(self, input_size, num_class):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = 512, #128
            num_layers =  1,
            batch_first = True,
            bidirectional  = False,
        )
        # self.gru = nn.GRU(
        #     input_size  = input_size,
        #     hidden_size = 512, #128
        #     num_layers =  1,
        #     batch_first = True,
        #     bidirectional  = False
        # )
        self.dropout = nn.Dropout(p=0.1) 
        self.fc = nn.Linear(512, num_class)

    def forward(self, input):
        input = input.reshape(1,-1,self.input_size)  
        # lstm_out = batch * seq_len * hidden_size
        #lstm_out, (hn, cn) = self.lstm(input, None)
        lstm_out, hn = self.gru(input, None)
        # last_hidden = batch * hidden_size(num_layers*isbidirectional * hidden_size) = (1*2*128)
        last_hidden = hn.transpose(0,1).contiguous().view(1, -1)
        
        # predict = batch * num_class
        predict = self.fc(last_hidden)
        #predict = self.classifier(last_hidden)
        
        return predict