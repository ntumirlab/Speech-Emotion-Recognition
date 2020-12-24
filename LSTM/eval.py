import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from model_2 import LSTM
import logging
import os
import argparse
import json
import opts
import sys
import random
BATCH_SIZE = 32
DEVICE = 'cpu'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval():
    config = opts.parse_opt()
    model = LSTM(76, len(config.class_labels))

    eval_data = pickle.load(open('data/opensmile/'+config.dataset+'_test.pkl', 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=1,num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)

    best = -100
    best_model = ''
    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir('model_state/'+config.dataset+'/hidden/opensmile/')
    for filename in os.listdir('.'):
        if not filename.endswith('pt'):
                continue
        
    
        ckpt = torch.load(filename,map_location=DEVICE)
        model.load_state_dict(ckpt['state_dict'])
        #model.cuda()
        model.eval()

        m = nn.Softmax(dim=1)
        
        prediction = []
        reference = []
        pred = []
        ref = []

        valid_progress = tqdm(total=len(eval_loader))
        for d in eval_loader:
            valid_progress.update(1)
            feature, label = [t.to(DEVICE) for t in d]
            predict = model(feature)
            output = torch.argmax(m(predict), dim=1).view(-1).tolist()
            target = label.view(-1).tolist()
            prediction.extend(output)
            reference.extend(target)
        
            

        assert len(prediction) == len(reference)
        acc = sum([x == y for x, y in zip(prediction, reference)]) / len(prediction)
        print(acc)

        if acc > best:
            best = acc
            best_model = filename
    print('Best model : {}, Best acc : {}'.format(best_model,best))        

    


if __name__ == '__main__':
    setup_seed(666)
    eval()
