import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import Signal_Analysis.features.signal as sa_signal

def data_split(X,Y):
    xxx = StratifiedShuffleSplit(1, test_size=0.2, random_state=12)
    for train_index, test_index in xxx.split(X, Y):
        train, test = X.iloc[train_index], X.iloc[test_index]
        label_train, label_test = Y.iloc[train_index], Y.iloc[test_index]
    df=pd.DataFrame(columns=["song_name"])
    df=pd.DataFrame(train)
    df=df.reset_index(drop=True)
    test=test.reset_index(drop=True)
    label_test=label_test.reset_index(drop=True)
    return (df,test,label_test)

def fearture_setting(X, sample_rate, data, index):

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

    # f0, voiced_flag, voiced_probs = librosa.pyin(X, frame_length= int(0.025*sample_rate), hop_length = int(0.010* sample_rate), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    # f0[np.isnan(f0)] = 0.0
    # f0 = f0.reshape(len(f0),1)

    # hnr = sa_signal.get_HNR(X, sample_rate)

    # list_HNR = np.array([hnr for i in range(len(f0))]).reshape(len(f0), 1)
    
    feature= np.hstack((feature, mfcc_delta, mfcc_delta2, ener, zcrs))
    return feature

def feature_extraction_train(df):
    data = np.asarray(())
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load('Data/wav/' + df.song_name[i], res_type='kaiser_fast', sr=16000)
        feature = fearture_setting(X, sample_rate, df, i)

        if data.size==0:
            data = feature
        else:
            data = np.vstack((data,feature))
            
    return np.array(data)

def feature_extraction_test(df):
    feature_list = []
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load('Data/wav/' + df[i], res_type='kaiser_fast', sr=16000)
        feature = fearture_setting(X, sample_rate, df, i)

        feature_list.append(feature)
            
    return feature_list

def pre_processing():

    dir_list = os.listdir('Data/wav')
    dir_list.sort()

    Labels = { 'W':'anger', 'L':'boredom', 'E':'disgust', 'N':'neutral', 'A':'fear', 'F':'happiness', 'T':'sadness' }
    label_name = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]

    # Creating dataframe for Emo-DB
    data_df = pd.DataFrame(columns=['song_name', 'emo_labels'])
    count = 0
    for song_name in dir_list:
        label_key = song_name[5]
        emo_labels = Labels[label_key]
        data_df.loc[count] = [song_name,emo_labels]
        count += 1

    # Dropping the none
    data_df = data_df[data_df.emo_labels != 'none'].reset_index(drop=True)

    # Separating the dataset for different emotion
    df_anger = data_df[data_df.emo_labels == 'anger']
    df_anger = df_anger.reset_index(drop = True)

    df_boredom = data_df[data_df.emo_labels == 'boredom']
    df_boredom = df_boredom.reset_index(drop = True)

    df_disgust = data_df[data_df.emo_labels == 'disgust']
    df_disgust = df_disgust.reset_index(drop = True)

    df_neutral = data_df[data_df.emo_labels == 'neutral']
    df_neutral = df_neutral.reset_index(drop = True)

    df_fear = data_df[data_df.emo_labels == 'fear']
    df_fear = df_fear.reset_index(drop = True)

    df_happiness = data_df[data_df.emo_labels == 'happiness']
    df_happiness = df_happiness.reset_index(drop = True)

    df_sadness = data_df[data_df.emo_labels == 'sadness']
    df_sadness = df_sadness.reset_index(drop = True)

    #print("df_anger:", df_anger)

    # Data Spliting
    X = df_anger["song_name"]
    Y = df_anger["emo_labels"]
    d_anger, anger_test, anger_label_test = data_split(X, Y)

    #print("d_anger:", d_anger)

    X = df_boredom["song_name"]
    Y = df_boredom["emo_labels"]
    d_boredom, boredom_test, boredom_label_test = data_split(X, Y)

    X = df_disgust["song_name"]
    Y = df_disgust["emo_labels"]
    d_disgust, disgust_test, disgust_label_test = data_split(X, Y)

    X = df_neutral["song_name"]
    Y = df_neutral["emo_labels"]
    d_neutral, neutral_test, neutral_label_test = data_split(X, Y)

    X = df_fear["song_name"]
    Y = df_fear["emo_labels"]
    d_fear, fear_test, fear_label_test = data_split(X, Y)

    X = df_happiness["song_name"]
    Y = df_happiness["emo_labels"]
    d_happiness, happiness_test, happiness_label_test = data_split(X, Y)

    X = df_sadness["song_name"]
    Y = df_sadness["emo_labels"]
    d_sadness, sadness_test, sadness_label_test = data_split(X, Y)

    test = pd.concat([anger_test, boredom_test, disgust_test, neutral_test, fear_test, happiness_test, sadness_test], axis=0, sort=True).reset_index(drop=True)
    test_label = pd.concat([anger_label_test, boredom_label_test, disgust_label_test, neutral_label_test, fear_label_test, happiness_label_test, sadness_label_test], axis=0, sort=True).reset_index(drop=True)

    # print(d_anger)

    print('Feature Extraction: angry')
    anger = feature_extraction_train(d_anger)

    print('Feature Extraction: boredom')
    boredom = feature_extraction_train(d_boredom)
    print('Feature Extraction: disgust')
    disgust = feature_extraction_train(d_disgust)
    print('Feature Extraction: neutral')
    neutral = feature_extraction_train(d_neutral)
    print('Feature Extraction: fear')
    fear = feature_extraction_train(d_fear)
    print('Feature Extraction: happiness')
    happiness = feature_extraction_train(d_happiness)
    print('Feature Extraction: sadness')
    sadness = feature_extraction_train(d_sadness)

    print('Feature Extraction: test')
    test_feature_list = feature_extraction_test(test)
    # print("test_feature_list:", len(test_feature_list))
    # print("test_label:", test_label.shape)

    with open('./Data/train/anger.pickle', 'wb') as handle:
        pickle.dump(anger, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./Data/train/boredom.pickle', 'wb') as handle:
        pickle.dump(boredom, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./Data/train/disgust.pickle', 'wb') as handle:
        pickle.dump(disgust, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./Data/train/neutral.pickle', 'wb') as handle:
        pickle.dump(neutral, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./Data/train/fear.pickle', 'wb') as handle:
        pickle.dump(fear, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./Data/train/happiness.pickle', 'wb') as handle:
        pickle.dump(happiness, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./Data/train/sadness.pickle', 'wb') as handle:
        pickle.dump(sadness, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./Data/train/test.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./Data/train/test_feature_list.pickle', 'wb') as handle:
        pickle.dump(test_feature_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./Data/train/test_label.pickle', 'wb') as handle:
        pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # return anger, boredom, disgust, neutral, fear, happiness, sadness, test, test_label

if __name__ == "__main__":
    pre_processing()
    # print(anger.shape)
    # print(boredom.shape)
    # print(disgust.shape)
    # print(neutral.shape)
    # print(fear.shape)
    # print(happiness.shape)
    # print(sadness.shape)
    # print(test.shape)
    # print(test_label.shape)
