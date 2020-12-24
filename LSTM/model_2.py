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
            hidden_size = 128,
            num_layers =  1,
            batch_first = True,
            bidirectional  = True,
        )
        self.fc = nn.Linear(2*128, num_class)

    def forward(self, input):
        input = input.reshape(1,-1,self.input_size)  
        # lstm_out = batch * seq_len * hidden_size
        lstm_out, (hn, cn) = self.lstm(input, None)
        # last_hidden = batch * hidden_size(num_layers*isbidirectional * hidden_size) = (1*2*128)
       
        last_hidden = hn.transpose(0,1).contiguous().view(1, -1)
        
        # predict = batch * num_class
        predict = self.fc(last_hidden)
        
        
        return predict
        
        


