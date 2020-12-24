import re
import numpy as np
import pickle
from data import load
import logging
import os
from opensmile import get_data
import opts
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
def cmvn(feat):
    # ddof=1 for same result between np.std and torch.std
    # feat = [fea_dim, timestep]
    print(feat.shape)
    return (feat - feat.mean(1, keepdims=True)) / (1e-10 + feat.std(1, keepdims=True, ddof=1))
if __name__ == "__main__":
    config = opts.parse_opt()
    X = []
    y = []
    data = get_data(config, config.data_path, train = True)
    for item in data:
        #X.append(cmvn(item[0].T).T)
        X.append(item[0])
        y.append(item[1])
    
    
    X = np.array(X)
    y = np.array(y)
    print(X.shape, y.shape)
    X_train, test, y_train, test_label = train_test_split(X, y, test_size=0.1, stratify=y)
    #print(X_train.shape, y_train.shape,  test.shape, test_label.shape)
    #print(np.unique(y_train, return_counts=True),np.unique(test_label, return_counts=True))

    train, valid, train_label, valid_label = train_test_split(X_train, y_train, test_size=0.1, stratify=y_train)
    print(train.shape, valid.shape, test.shape, train_label.shape, valid_label.shape, test_label.shape)
    print(np.unique(train_label, return_counts=True),np.unique(valid_label, return_counts=True),np.unique(test_label, return_counts=True))
    
    
    print("loading training data")
    with open('data/opensmile/'+config.dataset+'_train.pkl', 'wb') as f:
        pickle.dump(load('train', train, train_label), f)

    print("loading validation data")
    with open('data/opensmile/'+config.dataset+'_valid.pkl', 'wb') as f:
        pickle.dump(load('train', valid, valid_label), f)

    print("loading testing data")
    with open('data/opensmile/'+config.dataset+'_test.pkl', 'wb') as f:
        pickle.dump(load('train', test, test_label), f)
    
    
