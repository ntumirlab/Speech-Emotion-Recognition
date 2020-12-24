import os
import re
import sys
import librosa
import librosa.display
from random import shuffle
import numpy as np
from typing import Tuple
import pickle
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import utils.opts as opts

def features(X, sample_rate):

    stft = np.abs(librosa.stft(X))

    # fmin 和 fmax 对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(X, sr = sample_rate, S = stft, fmin = 70, fmax = 400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)
    pitchmean = np.mean(pitch)
    pitchstd = np.std(pitch)
    pitchmax = np.max(pitch)
    pitchmin = np.min(pitch)

    # 频谱质心
    cent = librosa.feature.spectral_centroid(y = X, sr = sample_rate)
    cent = cent / np.sum(cent)
    meancent = np.mean(cent)
    stdcent = np.std(cent)
    maxcent = np.max(cent)

    # 谱平面
    flatness = np.mean(librosa.feature.spectral_flatness(y = X))

    # 使用系数为50的MFCC特征
    mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
    mfccsstd = np.std(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)
    mfccmax = np.max(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 50).T, axis = 0)

    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)

    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis = 0)

    # ottava对比
    contrast = np.mean(librosa.feature.spectral_contrast(S = stft, sr = sample_rate).T, axis = 0)

    # 过零率
    zerocr = np.mean(librosa.feature.zero_crossing_rate(X))

    S, phase = librosa.magphase(stft)
    meanMagnitude = np.mean(S)
    stdMagnitude = np.std(S)
    maxMagnitude = np.max(S)

    # 均方根能量
    rmse = librosa.feature.rmse(S=S)[0]
    meanrms = np.mean(rmse)
    stdrms = np.std(rmse)
    maxrms = np.max(rmse)

    ext_features = np.array([
        flatness, zerocr, meanMagnitude, maxMagnitude, meancent, stdcent,
        maxcent, stdMagnitude, pitchmean, pitchmax, pitchstd,
        pitch_tuning_offset, meanrms, maxrms, stdrms
    ])

    ext_features = np.concatenate((ext_features, mfccs, mfccsstd, mfccmax, chroma, mel, contrast))

    return ext_features


def extract_features(file, pad = False):
    X, sample_rate = librosa.load(file, sr = None)
    max_ = X.shape[0] / sample_rate
    if pad:
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')
    return features(X, sample_rate)
    

def get_max_min(files):

    min_, max_ = 100, 0

    for file in files:
        sound_file, samplerate = librosa.load(file, sr = None)
        t = sound_file.shape[0] / samplerate
        if t < min_:
            min_ = t
        if t > max_:
            max_ = t

    return max_, min_


'''
get_data_path(): 获取所有音频的路径

输入:
    data_path: 数据集文件夹路径
    class_labels(list): 情感标签
输出:
    所有音频的路径
'''

def get_data_path(data_path: str, class_labels):
    config = opts.parse_opt()
    
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    if config.dataset == 'RAVDESS':
        folder = ['Actor_' + ('%.2d' % i) for i in range(1,25)]    
    elif config.dataset == 'emodb':
        folder = ['wav']
    elif config.dataset == 'NNIME':
        folder = ['Speech']
    for _, directory in enumerate(folder):
        os.chdir(directory)
        # 读取该文件夹下的音频
        for filename in os.listdir('.'):
            if not filename.endswith('wav'):
                continue
            filepath = os.path.join(os.getcwd(), filename)
            wav_file_path.append(filepath)

        os.chdir('..')
    os.chdir(cur_dir)

    shuffle(wav_file_path)
    return wav_file_path


'''
load_feature(): 从 csv 加载特征数据

输入:
    config(Class)
    feature_path: 特征文件路径
    train: 是否为训练数据

输出:
    训练数据、测试数据和对应的标签
'''

def load_feature(config, feature_path: str, train: bool):
    X, Y = [], []
    
        
    features = pd.DataFrame(data = joblib.load(feature_path), columns = ['file_name', 'features', 'emotion'])

    X.extend(list(features['features']))
    Y.extend(list(features['emotion']))

    # 标准化模型路径 要根據librosa或opensmile改
    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_OPENSMILE.m')

    if train == True:
        # 标准化数据 
        scaler = StandardScaler().fit(X)
        # 保存标准化模型
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.1, random_state = 42)
        return x_train, x_test, y_train, y_test
    
    else:
        # 标准化数据
        # 加载标准化模型
        scaler = joblib.load(scaler_path)
        X = scaler.transform(X)
        return X


'''
get_data(): 
    提取所有音频的特征: 遍历所有文件夹, 读取每个文件夹中的音频, 提取每个音频的特征，把所有特征保存在 feature_path 中

输入:
    config(Class)
    data_path: 数据集文件夹/测试文件路径
    feature_path: 保存特征的路径
    train: 是否为训练数据

输出:
    train = True: 训练数据、测试数据特征和对应的标签
    train = False: 预测数据特征
'''
def get_data(config, data_path: str, feature_path: str, train: bool):
    
    if(train == True):
        files = get_data_path(data_path, config.class_labels)
        
        max_, min_ = get_max_min(files)

        mfcc_data = []
        emodb_label = {'W':'angry', 'E':'disgust', 'A':'fear', 'F':'happy', 'T':'sad', 'N':'neutral'}
        ravdess_label = {'05':'angry', '07':'disgust', '06':'fear', '03':'happy', '04':'sad', '01':'neutral'}
        for file in files:
            if config.dataset == 'emodb':
                if file[-6] not in emodb_label:
                    continue
                _class = emodb_label[file[-6]]
            elif config.dataset == 'RAVDESS':
                file_label = file.split('/')[-1].split('.')[0].split('-')[2]
                if file_label not in ravdess_label:
                    continue
                _class = ravdess_label[file_label]
            elif config.dataset == 'NNIME':
                _class = file.split('/')[-1].split('_')[0]    

            #label = re.findall(".*-(.*)-.*", file)[0]

            # 三分类
            # if(label == "sad" or label == "neutral"):
            #     label = "neutral"
            # elif(label == "angry" or label == "fear"):
            #     label = "negative"
            # elif(label == "happy" or label == "surprise"):
            #     label = "positive"

            features = extract_features(file, max_)
            mfcc_data.append([file, features, config.class_labels.index(_class)])
            
        
    else:
        features = extract_features(data_path)
        mfcc_data = [[data_path, features, -1]]

    print(len(mfcc_data))
    cols = ['file_name', 'features', 'emotion']
    mfcc_pd = pd.DataFrame(data = mfcc_data, columns = cols)
    pickle.dump(mfcc_data, open(feature_path, 'wb'))
    
    return load_feature(config, feature_path, train = train)