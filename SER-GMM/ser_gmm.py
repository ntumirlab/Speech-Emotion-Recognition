# Importing the required libraries
import os
import random
import sys
import csv

# package
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import pickle

from preprocessing import pre_processing

# Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def train_model(features):
    gmm = GaussianMixture(n_components=512).fit(features)
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

def test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,data):
    predict = pd.DataFrame(columns=['predict'])
    for i in tqdm(range(len(data))):
        X, sample_rate = librosa.load('Data/wav/' + data[i], res_type='kaiser_fast',duration=input_duration,sr=16000,offset=0.5)

        sample_rate = np.array(sample_rate)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, hop_length=int(0.010*sample_rate), n_fft=int(0.020*sample_rate), n_mfcc=13)
        feature = mfccs.transpose()
        mfcc_delta=librosa.feature.delta(feature)
        mfcc_delta2=librosa.feature.delta(feature, order=2)
        ener = librosa.feature.rms(y=X, frame_length= int(0.020*sample_rate), hop_length = int(0.010* sample_rate))
        ener=ener.transpose()

        # ZCR
        zcrs = librosa.feature.zero_crossing_rate(y=X, frame_length= int(0.020*sample_rate), hop_length = int(0.010* sample_rate))
        zcrs=zcrs.transpose()

        f0, voiced_flag, voiced_probs = librosa.pyin(X, frame_length= int(0.020*sample_rate), hop_length = int(0.010* sample_rate), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0[np.isnan(f0)] = 0.0
        f0 = f0.reshape(len(f0),1)
        # # F0
        # d = zcrs.shape[0]
        # F0 = sa_signal.get_F_0(X, sample_rate)

        # list_f0 = np.zeros((d,0))
        # for i in range(d):
        #     list_f0[i] = F0[0]
        # # print("list_F0:", list_f0.shape)
        
        # # HNR
        # hnr = sa_signal.get_HNR(X, sample_rate)
        # # print("hnr:", hnr)

        # list_HNR = np.zeros((d,0))
        # for i in range(d):
        #     list_HNR[i] = hnr

        feature= np.hstack((feature, mfcc_delta, mfcc_delta2, ener, zcrs, f0))
        anger_score= gmm_anger.score(feature)
        boredom_score= gmm_boredom.score(feature)
        disgust_score= gmm_disgust.score(feature)
        neutral_score= gmm_neutral.score(feature)
        fear_score= gmm_fear.score(feature)
        happiness_score= gmm_happiness.score(feature)
        sadness_score= gmm_sadness.score(feature)

        dicta = {
            "anger": anger_score,
            "boredom": boredom_score,
            "disgust": disgust_score,
            "neutral": neutral_score,
            "fear": fear_score,
            "happiness": happiness_score,
            "sadness": sadness_score
        }
        x=max(dicta, key=dicta.get)
        predict.loc[i]=x
    return predict

# def load_model(name):
#     with open('Models/'+name+".sav", 'rb') as pickle_file:
#         return pickle.load(pickle_file)

anger, boredom, disgust, neutral, fear, happiness, sadness, test, test_label = pre_processing()

training()

# gmm_anger= load_model("anger_model")
# gmm_boredom= load_model("boredom_model")
# gmm_disgust=load_model("disgust_model")
# gmm_neutral=load_model("neutral_model")
# gmm_fear= load_model("fear_model")
# gmm_happiness=load_model("happiness_model")
# gmm_sadness=load_model("sadness_model")

# predict  = test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,test)

# # print(predict.predict.value_counts())

# # c = confusion_matrix(test_label, predict)
# # print('accuracy:', accuracy(c)) 


# from sklearn.preprocessing import label_binarize
# from sklearn.metrics import roc_curve, auc, f1_score

# testing_label= label_binarize(test_label, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])
# testing_predict= label_binarize(predict, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])
# # [110, 84]

# # print('testing_label:', testing_label)
# # print('testing_predict:', testing_predict)
# for i in range(7):
#     #print(label_name[i])
#     print('{} f1_scores:{}'.format(label_name[i], f1_score(testing_label[:, i], testing_predict[:, i])))



# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(7):
#     fpr[i], tpr[i], _ = roc_curve(testing_label[:, i], testing_predict[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(testing_label.ravel(), testing_predict.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(7):
#     fpr[i], tpr[i], _ = f1_score(testing_label[:, i], testing_predict[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = f1_score(testing_label.ravel(), testing_predict.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# from scipy import interp
# from itertools import cycle

# lw=2
# n_classes=7
# # First aggregate all false positive rates
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# # Then interpolate all ROC curves at this points
# mean_tpr = np.zeros_like(all_fpr)
# for i in range(n_classes):
#     mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# # Finally average it and compute AUC
# mean_tpr /= n_classes

# fpr["macro"] = all_fpr
# tpr["macro"] = mean_tpr
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


# label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]
# # Plot all ROC curves
# plt.figure()
# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.plot(fpr["macro"], tpr["macro"],
#          label='macro-average ROC curve (area = {0:0.2f})'
#                ''.format(roc_auc["macro"]),
#          color='navy', linestyle=':', linewidth=4)

# colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
# for i, color in zip(range(n_classes), colors):
#     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(label_name[i], roc_auc[i]))

# plt.plot([0, 1], [0, 1], 'k--', lw=lw)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic of multi-class')
# plt.legend(loc="lower right")
# plt.show()