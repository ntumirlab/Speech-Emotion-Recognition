import os
import csv
import sys
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import joblib
from sklearn.model_selection import train_test_split
import utils.opts as opts
from random import shuffle
# 每个特征集的特征数量
FEATURE_NUM = {
    'IS09_emotion': 384,
    'IS10_paraling': 1582,
    'IS11_speaker_state': 4368,
    'IS12_speaker_trait': 6125,
    'IS13_ComParE': 6373,
    'ComParE_2016': 6373
}


'''
get_feature_opensmile(): Opensmile 提取一个音频的特征

输入:
    config(Class)
    file_path: 音频路径

输出：
    该音频的特征向量
'''

def get_feature_opensmile(config, filepath: str):
    # 项目路径
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # single_feature.csv 路径
    single_feat_path = os.path.join(BASE_DIR, config.feature_path, 'single_feature.csv')
    # Opensmile 配置文件路径

    # Opensmile 命令
    cmd = 'cd ' + config.opensmile_path + ' && ./SMILExtract -C ' + config.opensmile_config + ' -I ' + filepath + ' -O ' + single_feat_path
    print("Opensmile cmd: ", cmd)
    os.system(cmd)
    
    reader = csv.reader(open(single_feat_path,'r'))
    rows = [row for row in reader]
    last_line = rows[-1]
    return last_line[1: FEATURE_NUM[config.opensmile_config.split('/')[-1].split('.')[0]] + 1]


'''
load_feature(): 从 .csv 文件中加载特征数据

输入:
    config(Class)
    feature_path: 特征文件路径
    train: 是否为训练数据

输出:
    训练数据、测试数据和对应的标签
'''

def load_feature(config, feature_path: str, train: bool):
    # 加载特征数据
    df = pd.read_csv(feature_path)
    features = [str(i) for i in range(1, FEATURE_NUM[config.opensmile_config.split('/')[-1].split('.')[0]] + 1)]

    X = df.loc[:,features].values
    Y = df.loc[:,'label'].values

    # 标准化模型路径
    scaler_path = os.path.join(config.checkpoint_path, 'SCALER_OPENSMILE.m')

    if train == True:
        # 标准化数据 
        scaler = StandardScaler().fit(X)
        # 保存标准化模型
        joblib.dump(scaler, scaler_path)
        X = scaler.transform(X)

        # 划分训练集和测试集
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
# Opensmile 提取特征
def get_data(config, data_path, feature_path: str, train: bool):

    writer = csv.writer(open(feature_path, 'w'))
    first_row = ['label']
    for i in range(1, FEATURE_NUM[config.opensmile_config.split('/')[-1].split('.')[0]] + 1):
        first_row.append(str(i))
    writer.writerow(first_row)

    writer = csv.writer(open(feature_path, 'a+'))
    print('Opensmile extracting...')
    emodb_label = {'W':'angry', 'E':'disgust', 'A':'fear', 'F':'happy', 'T':'sad', 'N':'neutral'}
    ravdess_label = {'08':'surprise','02':'calm','05':'angry', '07':'disgust', '06':'fear', '03':'happy', '04':'sad', '01':'neutral'}
    if train == True:
        files = get_data_path(data_path, config.class_labels)

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
                
            feature_vector = get_feature_opensmile(config, file)
            feature_vector.insert(0, config.class_labels.index(_class))
            # 把每个音频的特征整理到一个 csv 文件中
            writer.writerow(feature_vector)
    
    else:
        feature_vector = get_feature_opensmile(config, data_path)
        feature_vector.insert(0, '-1')
        writer.writerow(feature_vector)

    print('Opensmile extract done.')

    # 一个玄学 bug 的暂时性解决方案
    # 这里无法直接加载除了 IS10_paraling 以外的其他特征集的预测数据特征，非常玄学
    if(train == True):
        return load_feature(config, feature_path, train = train)
