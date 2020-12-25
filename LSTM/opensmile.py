import os
import sys
import librosa
import numpy as np
import opts
import re
from random import shuffle
import csv
def extract_features(filepath):
    # 项目路径
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    # single_feature.csv 路径
    single_feat_path = os.path.join('./data/single_feature.csv')
    # Opensmile 配置文件路径

    # Opensmile 命令
    cmd = '/home/victor/Project/util/opensmile/build/progsrc/smilextract/SMILExtract -C /home/victor/Project/util/opensmile/config/is09-13/IS10_paraling.conf -I ' + filepath + ' -D ' + single_feat_path
    print("Opensmile cmd: ", cmd)
    os.system(cmd)
    reader = csv.reader(open(single_feat_path,'r'))
    rows = [row for row in reader]
    feature = [i[0].split(";")[2:] for i in rows[1:]]
    feature = [[float(j) for j in i] for i in feature]
    return np.array(feature)
    




def get_data_path(data_path: str, class_labels):
    config = opts.parse_opt()
    
    wav_file_path = []

    cur_dir = os.getcwd()
    sys.stderr.write('Curdir: %s\n' % cur_dir)
    os.chdir(data_path)
    if config.dataset == 'RAVDESS':
        folder = ['Actor_' + ('%.2d' % i) for i in range(1,25)]    
    elif config.dataset == 'emodb':
        folder = ['emodb']
    elif config.dataset == 'NNIME':
        folder = ['Speech']
    elif config.dataset == 'CASIA':
        folder = class_labels
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

def get_data(config, data_path: str, train: bool):
    
    if train == True:
        files = get_data_path(data_path, config.class_labels)
        data = []
        emodb_label = {'L':'boredom', 'W':'angry', 'E':'disgust', 'A':'fear', 'F':'happy', 'T':'sad', 'N':'neutral'}
        ravdess_label = {'08':'surprise','02':'calm','05':'angry', '07':'disgust', '06':'fear', '03':'happy', '04':'sad', '01':'neutral'}
    
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
            elif config.dataset == 'CASIA':
                _class = re.findall(".*-(.*)-.*", file)[0]
        
            data.append([extract_features(file), config.class_labels.index(_class)])
    else:
        data = [[extract_features(data_path), -1]]

    return data

if __name__ == "__main__":
    data = []
    file = ['12a05Ab.wav']
    for i in file:
        extract_features('/mnt/hdd18.2t/dataset/emodb/wav/'+ i)
    