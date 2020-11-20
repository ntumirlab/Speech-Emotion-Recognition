
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

def test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness, gmm_sadness, data):
    predict = pd.DataFrame(columns=['predict'])
    for i in tqdm(range(len(data))):
        X, sample_rate = librosa.load('Data/wav/' + data[i], res_type='kaiser_fast', sr=16000)

        sample_rate = np.array(sample_rate)
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, hop_length=int(0.010*sample_rate), n_fft=512, n_mfcc=13)
        feature = mfccs.transpose()
        mfcc_delta=librosa.feature.delta(feature)
        mfcc_delta2=librosa.feature.delta(feature, order=2)
        ener = librosa.feature.rms(y=X, frame_length= int(0.025*sample_rate), hop_length = int(0.010* sample_rate))
        ener=ener.transpose()

        # ZCR
        zcrs = librosa.feature.zero_crossing_rate(y=X, frame_length= int(0.025*sample_rate), hop_length = int(0.010* sample_rate))
        zcrs=zcrs.transpose()

        f0, voiced_flag, voiced_probs = librosa.pyin(X, frame_length= int(0.025*sample_rate), hop_length = int(0.010* sample_rate), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0[np.isnan(f0)] = 0.0
        f0 = f0.reshape(len(f0),1)

        hnr = sa_signal.get_HNR(X, sample_rate)

        list_HNR = np.array([hnr for i in range(len(f0))]).reshape(len(f0), 1)

        # print("feature:", feature.shape)
        # print("mfcc_delta:",mfcc_delta.shape)
        # print("mfcc_delta2:", mfcc_delta2.shape)
        # print("ener:",ener.shape)
        # print("zcrs:",zcrs.shape)
        # print("zcrs:",zcrs)
        # print("f0:",f0.shape)
        
        # print("f0:",f0)
        # print("voiced_flag:",voiced_flag.shape)
        # print("X:", X.shape)
        
        feature= np.hstack((feature, mfcc_delta, mfcc_delta2, ener, zcrs, f0, list_HNR))

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


    with open('./Data/train/anger.pickle', 'rb') as handle:
        anger=pickle.Unpickler(handle).load()
    
    with open('./Data/train/boredom.pickle', 'rb') as handle:
        boredom=pickle.Unpickler(handle).load()

    with open('./Data/train/disgust.pickle', 'rb') as handle:
        disgust=pickle.load(handle)

    with open('./Data/train/neutral.pickle', 'rb') as handle:
        neutral=pickle.load(handle)
    
    with open('./Data/train/fear.pickle', 'rb') as handle:
        fear=pickle.load(handle)

    with open('./Data/train/happiness.pickle', 'rb') as handle:
        happiness=pickle.load(handle)
    
    with open('./Data/train/sadness.pickle', 'rb') as handle:
        sadness=pickle.load(handle)

    with open('./Data/train/test.pickle', 'rb') as handle:
        test = pickle.load(handle)
    
    with open('./Data/train/test_label.pickle', 'rb') as handle:
        test_label = pickle.load(handle)



    # anger, boredom, disgust, neutral, fear, happiness, sadness, test, test_label = pre_processing()

    predict  = test_model(gmm_anger, gmm_boredom, gmm_disgust, gmm_neutral, gmm_fear, gmm_happiness,gmm_sadness,test)

    label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]

    testing_label= label_binarize(test_label, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])
    testing_predict= label_binarize(predict, classes=["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"])

    # ss_sum = 0.0
    # for i in range(7):
    # #print(label_name[i])
    #     f1 = f1_score(testing_label[:, i], testing_predict[:, i])
    #     print('{} f1_scores:{}'.format(label_name[i], f1))
    #     ss_sum += float(f1)

    # print('averge f1 score:', ss_sum/7)

    ss_sum = 0.0
    for i in range(7):
    #print(label_name[i])
        accuracy = accuracy_score(testing_label[:, i], testing_predict[:, i])
        print('{} accuracy score:{}'.format(label_name[i], accuracy))
        ss_sum += float(accuracy)

    print('averge accuracy score:', ss_sum/7)

    print('----------------------------------------------------')

    # cm_sum = 0.0
    pds = []
    gts = []
    for i in range(7):
    #print(label_name[i])
        for j in range(len(testing_label)):
            print(i,j, testing_label[j][i], testing_predict[j][i])
        gts.extend(testing_label[:, i])
        pds.extend(testing_predict[:, i])
    
    con_matrix = confusion_matrix(gts, pds, label_name)
    # print('{} confusion_matrix score:{}'.format(label_name[i], accuracy))
    # cm_sum += float(con_matrix)
    # print(confusion_matrix)
    print(con_matrix)

    # print('averge accuracy score:', cm_sum / 7)


if __name__ == "__main__":
    evaluate()
