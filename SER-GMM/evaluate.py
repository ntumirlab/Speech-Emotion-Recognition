
import librosa
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm

from preprocessing import pre_processing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, f1_score, accuracy_score
import Signal_Analysis.features.signal as sa_signal
from preprocessing import fearture_setting

# Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def load_model(name):
    with open('Models/'+name+".sav", 'rb') as pickle_file:
        return pickle.load(pickle_file)

def test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness, gmm_sadness, f_list):
    predict = pd.DataFrame(columns=['predict'])
    for i in tqdm(range(len(f_list))):
        feature = f_list[i]

        anger_score = gmm_anger.score(feature)
        #print("anger_score:", anger_score)
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
        x = max(dicta, key=dicta.get)
        predict.loc[i] = x
    
    return predict

def evaluate():
    gmm_anger= load_model("anger_model")
    gmm_boredom= load_model("boredom_model")
    gmm_disgust=load_model("disgust_model")
    gmm_neutral=load_model("neutral_model")
    gmm_fear= load_model("fear_model")
    gmm_happiness=load_model("happiness_model")
    gmm_sadness=load_model("sadness_model")

    with open('./Data/train/test_feature_list.pickle', 'rb') as handle:
        test_feature_list = pickle.load(handle)
    
    with open('./Data/train/test_label.pickle', 'rb') as handle:
        test_label = pickle.load(handle)

    predict  = test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,test_feature_list)

    label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]

    testing_label= label_binarize(test_label, classes = label_name )
    testing_predict= label_binarize(predict, classes = label_name )

    c = confusion_matrix(test_label, predict)
    aver_acc = accuracy(c)
    cm = c.astype('float') / c.sum(axis=1)[:, np.newaxis]*100
    print("aver_acc:{}%".format(round(aver_acc*100, 2)))
    for i, lname in enumerate(label_name):
        print('{} acc: {}%'.format(lname , round(cm[i][i], 2)))



    

if __name__ == "__main__":
    evaluate()
