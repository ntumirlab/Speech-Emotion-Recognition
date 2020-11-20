# Importing the required libraries
import os
import random
import sys
# package
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import pickle

from preprocessing import pre_processing
from hmmlearn import hmm

label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]

# Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def train_model_gmm_hmm(features):
    model = hmm.GMMHMM(n_components=10, n_mix=1, n_iter=100)
    model.fit(features)
    labels = model.predict(features)
    return (model,labels)

def train_model_hmm(features):
    model = hmm.GaussianHMM(n_components=5, n_iter=500)
    model.fit(features)
    labels = model.predict(features)
    return (model,labels)

def train_model(features):
    gmm = GaussianMixture(n_components=5).fit(features)
    labels = gmm.predict(features)
    return (gmm,labels)

def save_model(model,name):
    save_dir = os.path.join(os.getcwd(), 'Models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = name+'.sav'
    model_path = os.path.join(save_dir, filename)
    print(model_path)
    pickle.dump(model, open(model_path, 'wb'))

# Training
#Train Emotion
def training():
    print('Begining of Training & save!')
    #anger model

    # for index, item in enumerate(label_name):
    #     print('Training & save:{}'.format(format(label_name[index]))

    print('Training & save: angry')
    gmm_anger, train_anger_labels = train_model(anger)
    save_model(gmm_anger, "anger_model")

    #boredom model
    print('Training & save: boredom')
    gmm_boredom, train_boredom_labels = train_model(boredom)
    save_model(gmm_boredom, "boredom_model")
    #disgust model
    print('Training & save: disgust')
    gmm_disgust, train_disgust_labels = train_model(disgust)
    save_model(gmm_disgust, "disgust_model")
    #neutral model
    print('Training & save: neutral')
    gmm_neutral, train_neutral_labels = train_model(neutral)
    save_model(gmm_neutral, "neutral_model")
    #fear model
    print('Training & save: fear')
    gmm_fear, train_fear_labels = train_model(fear)
    save_model(gmm_fear, "fear_model")
    #happiness model
    print('Training & save: happiness')
    gmm_happiness, train_happiness_labels = train_model(happiness)
    save_model(gmm_happiness, "happiness_model")
    #sadness model
    print('Training & save: sadness')
    gmm_sadness, train_sadness_labels = train_model(sadness)
    save_model(gmm_sadness, "sadness_model")
    print('finished Training & Save!')

# Training
#Train Emotion
def training_hmm():
    print('Begining of Training & save!')
    #anger model

    # for index, item in enumerate(label_name):
    #     print('Training & save:{}'.format(format(label_name[index]))

    print('Training & save: angry')
    gmm_anger, train_anger_labels = train_model_hmm(anger)
    save_model(gmm_anger, "anger_model")


    #boredom model
    print('Training & save: boredom')
    gmm_boredom, train_boredom_labels = train_model_hmm(boredom)
    save_model(gmm_boredom, "boredom_model")
    #disgust model
    print('Training & save: disgust')
    gmm_disgust, train_disgust_labels = train_model_hmm(disgust)
    save_model(gmm_disgust, "disgust_model")
    #neutral model
    print('Training & save: neutral')
    gmm_neutral, train_neutral_labels = train_model_hmm(neutral)
    save_model(gmm_neutral, "neutral_model")
    #fear model
    print('Training & save: fear')
    gmm_fear, train_fear_labels = train_model_hmm(fear)
    save_model(gmm_fear, "fear_model")
    #happiness model
    print('Training & save: happiness')
    gmm_happiness, train_happiness_labels = train_model_hmm(happiness)
    save_model(gmm_happiness, "happiness_model")
    #sadness model
    print('Training & save: sadness')
    gmm_sadness, train_sadness_labels = train_model_hmm(sadness)
    save_model(gmm_sadness, "sadness_model")
    print('finished Training & Save!')

# Training
#Train Emotion
def training_gmm_hmm():
    print('Begining of Training & save!')
    #anger model

    # for index, item in enumerate(label_name):
    #     print('Training & save:{}'.format(format(label_name[index]))

    print('Training & save: angry')
    gmm_anger, train_anger_labels = train_model_gmm_hmm(anger)


    save_model(gmm_anger, "anger_model")
    #boredom model
    print('Training & save: boredom')
    gmm_boredom, train_boredom_labels = train_model_gmm_hmm(boredom)
    save_model(gmm_boredom, "boredom_model")
    #disgust model
    print('Training & save: disgust')
    gmm_disgust, train_disgust_labels = train_model_gmm_hmm(disgust)
    save_model(gmm_disgust, "disgust_model")
    #neutral model
    print('Training & save: neutral')
    gmm_neutral, train_neutral_labels = train_model_gmm_hmm(neutral)
    save_model(gmm_neutral, "neutral_model")
    #fear model
    print('Training & save: fear')
    gmm_fear, train_fear_labels = train_model_gmm_hmm(fear)
    save_model(gmm_fear, "fear_model")
    #happiness model
    print('Training & save: happiness')
    gmm_happiness, train_happiness_labels = train_model_gmm_hmm(happiness)
    save_model(gmm_happiness, "happiness_model")
    #sadness model
    print('Training & save: sadness')
    gmm_sadness, train_sadness_labels = train_model_gmm_hmm(sadness)
    save_model(gmm_sadness, "sadness_model")
    print('finished Training & Save!')



if __name__ == "__main__":
    # pre_processing()
    with open('./Data/train/anger.pickle', 'rb') as handle:
        anger = pickle.load(handle)
    
    with open('./Data/train/boredom.pickle', 'rb') as handle:
        boredom = pickle.load(handle)

    with open('./Data/train/disgust.pickle', 'rb') as handle:
        disgust = pickle.load(handle)

    with open('./Data/train/neutral.pickle', 'rb') as handle:
        neutral = pickle.load(handle)
    
    with open('./Data/train/fear.pickle', 'rb') as handle:
        fear=pickle.load(handle)

    with open('./Data/train/happiness.pickle', 'rb') as handle:
        happiness=pickle.load(handle)
    
    with open('./Data/train/sadness.pickle', 'rb') as handle:
        sadness=pickle.load(handle)

    with open('./Data/train/test.pickle', 'rb') as handle:
        test=pickle.load(handle)
    
    with open('./Data/train/test_label.pickle', 'rb') as handle:
        test_label=pickle.load(handle)

    training()
    # training_hmm()
    # training_gmm_hmm()