
import librosa
import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import pickle
from tqdm import tqdm

from preprocessing import pre_processing
# input_duration = 3

# Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def load_model(name):
    with open('Models/'+name+".sav", 'rb') as pickle_file:
        return pickle.load(pickle_file)

def test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,data):
    predict = pd.DataFrame(columns=['predict'])
    for i in tqdm(range(len(data))):
        X, sample_rate = librosa.load('Data/wav/' + data[i], res_type='kaiser_fast',sr=16000)

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

def evaluate():
    gmm_anger= load_model("anger_model")
    gmm_boredom= load_model("boredom_model")
    gmm_disgust=load_model("disgust_model")
    gmm_neutral=load_model("neutral_model")
    gmm_fear= load_model("fear_model")
    gmm_happiness=load_model("happiness_model")
    gmm_sadness=load_model("sadness_model")

    anger, boredom, disgust, neutral, fear, happiness, sadness, test, test_label = pre_processing()

    predict  = test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,test)

    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc, f1_score

    label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]

    testing_label= label_binarize(test_label, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])
    testing_predict= label_binarize(predict, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])
    for i in range(7):
    #print(label_name[i])
        print('{} f1_scores:{}'.format(label_name[i], f1_score(testing_label[:, i], testing_predict[:, i])))


if __name__ == "__main__":
    evaluate()
