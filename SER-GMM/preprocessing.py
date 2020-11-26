import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
import pickle
from sklearn.model_selection import StratifiedShuffleSplit
import Signal_Analysis.features.signal as sa_signal

import utils.opts as opts

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

    f0, voiced_flag, voiced_probs = librosa.pyin(X, frame_length= int(0.025*sample_rate), hop_length = int(0.010* sample_rate), fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    f0[np.isnan(f0)] = 0.0
    f0 = f0.reshape(len(f0),1)

    hnr = sa_signal.get_HNR(X, sample_rate)

    list_HNR = np.array([hnr for i in range(len(f0))]).reshape(len(f0), 1)
    
    feature= np.hstack((feature, mfcc_delta, mfcc_delta2, ener, zcrs, f0, list_HNR))
    return feature

def feature_extraction_train(df, dataset_path):
    data = np.asarray(())
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load(dataset_path + df.song_name[i], res_type='kaiser_fast', sr=16000)
        feature = fearture_setting(X, sample_rate, df, i)

        if data.size==0:
            data = feature
        else:
            data = np.vstack((data,feature))
            
    return np.array(data)

def feature_extraction_test(df, dataset_path):
    feature_list = []
    #data = np.asarray(())
    for i in tqdm(range(len(df))):
        X, sample_rate = librosa.load(dataset_path + df[i], res_type='kaiser_fast', sr=16000)
        feature = fearture_setting(X, sample_rate, df, i)

        feature_list.append(feature)
        # if data.size==0:
        #     data = feature
        # else:
        #     data = np.vstack((data,feature))
            
    return feature_list

def pre_processing(config, label_name,dataset_path, pic_path):
    dir_list = os.listdir(dataset_path)
    dir_list.sort()

    if config.dataset_name == "emodb":
        Labels = { 'W':'anger', 'L':'boredom', 'E':'disgust', 'N':'neutral', 'A':'fear', 'F':'happiness', 'T':'sadness' }
        data_df = pd.DataFrame(columns=['song_name', 'emo_labels'])
        count = 0
        for song_name in dir_list:
            label_key = song_name[5]
            emo_labels = Labels[label_key]
            data_df.loc[count] = [song_name,emo_labels]
            count += 1
        print("total:{} speeches".format(count))

    elif config.dataset_name == "nnime":
        data_df = pd.DataFrame(columns=['song_name', 'emo_labels'])
        count = 0
        for song_name in dir_list:
            emo_labels = song_name[:-13]
            data_df.loc[count] = [song_name,emo_labels]
            count += 1
        print("total:{} speeches".format(count))
    
    elif config.dataset_name == "casia":
        data_df = pd.DataFrame(columns=['song_name', 'emo_labels'])
        count = 0
        for song_name in dir_list:
            emo_labels = song_name.split('-')[1]
            data_df.loc[count] = [song_name,emo_labels]
            count += 1
        print("total:{} speeches".format(count))

    # Dropping the none
    data_df = data_df[data_df.emo_labels != 'none'].reset_index(drop=True)

    emo_test_list, emo_test_labels_list = [], []
    for i, lname in enumerate(label_name):
        df_temp = data_df[data_df.emo_labels == lname]
        df_temp = df_temp.reset_index(drop = True)

        X = df_temp["song_name"]
        Y = df_temp["emo_labels"]
        d_train, emo_test, emo_test_labels = data_split(X, Y)

        print('Feature Extraction:{}'.format(lname))
        feature = feature_extraction_train(d_train, dataset_path)

        save_pick_name = pic_path + lname + '.pickle'
        with open(save_pick_name, 'wb') as p:
            pickle.dump(feature, p, protocol=pickle.HIGHEST_PROTOCOL)

        emo_test_list.append(emo_test)
        emo_test_labels_list.append(emo_test_labels)

    
    test = pd.concat( emo_test_list, axis=0, sort=True).reset_index(drop=True)
    test_label = pd.concat( emo_test_labels_list, axis=0, sort=True).reset_index(drop=True)

    with open(pic_path + 'test.pickle', 'wb') as handle:
        pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Feature Extraction: test')
    test_feature_list = feature_extraction_test(test, dataset_path)

    with open(pic_path + 'test_feature_list.pickle', 'wb') as handle:
        pickle.dump(test_feature_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(pic_path + 'test_label.pickle', 'wb') as handle:
        pickle.dump(test_label, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    config = opts.parse_opt()
    pre_processing(config, config.class_labels, config.dataset_path, config.pickle_path)
