import numpy as np
from data import Wavset
from model_2 import LSTM
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import logging
import pickle
import argparse
import opts
import random
# from focalloss import FocalLoss
config = opts.parse_opt()
DEVICE = 'cuda:'+str(config.train.gpuid)
os.environ["CUDA_AVAILABLE_DEVICES"] = str(config.train.gpuid)
def save(model, path, epoch, f):
    torch.save(
        {
            'state_dict' : model.state_dict(),
            'epoch': epoch,
            'f': f
        },
        path
    )
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(config):
    config = opts.parse_opt()
    logging.info('Loading training data from')
    train = pickle.load(open('data/opensmile/'+config.dataset+'_train.pkl', 'rb'))
    train_loader = DataLoader(train, batch_size=config.train.batch_size, num_workers=0, shuffle=True, collate_fn=train.collate_fn)
    logging.info('Loading validate data')
    valid = pickle.load(open('data/opensmile/'+config.dataset+'_valid.pkl', 'rb'))
    valid_loader = DataLoader(valid, batch_size=config.train.batch_size, num_workers=0, shuffle=False, collate_fn=valid.collate_fn)
    
    train_on_gpu = torch.cuda.is_available()
    logging.info('Initial model')
    model = LSTM(config.bi_lstm.input_size, len(config.class_labels))

    if train_on_gpu:
        model.to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    loss_function = nn.CrossEntropyLoss()
    #loss_function = FocalLoss(gamma = 2)

    n_epochs = config.train.epoch_num
    valid_loss_min = np.Inf
    counter = 0

    TRAIN_LOSS = []
    VALID_LOSS = []
    model.train()
    logging.info('Training')
    
    for epoch in range(1, n_epochs+1):
        
        train_loss = []
        progress = tqdm(total=len(train_loader))
        for d in train_loader:
            progress.update(1)
            counter += 1
            feature, label = [t.to(DEVICE) for t in d]

            optimizer.zero_grad()
            train_predict = model(feature)
                 
            # calculate the batch loss
            loss = loss_function(train_predict,label)
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()
        
        with torch.no_grad():
            logging.info('Validating')
            val_losses = []
            model.eval()
            valid_progress = tqdm(total=len(valid_loader))
            for c in valid_loader:
                valid_progress.update(1)
                # move tensors to GPU if CUDA is available
                valid_feature, valid_label = [t.to(DEVICE) for t in c]
                val_predict = model(valid_feature)
                val_loss = loss_function(val_predict,valid_label)
                val_losses.append(val_loss.item())

            model.train()
            print("Epoch: {}/{}...".format(epoch, n_epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.6f}...".format(np.mean(train_loss)),
                    "Val Loss: {:.6f}".format(np.mean(val_losses)))
            TRAIN_LOSS.append(np.mean(train_loss))
            VALID_LOSS.append(np.mean(val_losses))
            if np.mean(val_losses) <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)
                checkpoint_path = f'model_state/{config.dataset}/hidden/opensmile/best.pt'
                torch.save(
                    {
                        'state_dict' : model.state_dict(),
                        'epoch': epoch,
                    },
                    checkpoint_path
                )
    with open(f'model_state/{config.dataset}/hidden/opensmile/train_loss.pkl', 'wb') as f:
        pickle.dump(TRAIN_LOSS, f)    
    with open(f'model_state/{config.dataset}/hidden/opensmile/valid_loss.pkl', 'wb') as f:
        pickle.dump(VALID_LOSS, f)  
    
if __name__ == '__main__':
    config = opts.parse_opt()
    setup_seed(666)
    train(config)
