import os
import sys
import librosa
import numpy as np
import opts
import re
from random import shuffle

def extract_features(file, pad = False):
    X, sample_rate = librosa.load(file)
    print(sample_rate)
    max_ = X.shape[0] / sample_rate #second per frame
    if pad: 
        length = (max_ * sample_rate) - X.shape[0]
        X = np.pad(X, (0, int(length)), 'constant')

    frame_len =  int(0.015*sample_rate)
    hot_len = int(0.010* sample_rate) # shift
    sample_rate = np.array(sample_rate)
    # 39 dim mfcc per frame 
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, hop_length=hot_len, n_fft=512, n_mfcc=13).T  # t * mfcc (t*13)
    mfcc_delta = librosa.feature.delta(mfccs) # t * mfcc (t*13)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2) # t * mfcc (t*13)
    
    # 短時傅立葉轉換 abs取频率的振幅
    stft = np.abs(librosa.stft(X, hop_length=hot_len, n_fft=512)) #(1 + n_fft/2, t)
    # Compute root-mean-square (RMS) energy for each frame
    S, phase = librosa.magphase(stft)
    rmse = librosa.feature.rmse(S=S).T
    
    # pitch each frame
    # (d, t), d is the subset of FFT bins within fmin and fmax 
    #  default : fmin=150.0, fmax=4000.0
    pitches, magnitudes = librosa.piptrack(X, sr = sample_rate, hop_length=hot_len, n_fft=512)
    pitch = []
    for t in range(magnitudes.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])
    pitch = np.array(pitch).T.reshape(-1,1)
    
    return np.hstack((mfccs, mfcc_delta, mfcc_delta2, rmse, pitch)) # t * 41
    '''
    # ottava对比
    contrast = librosa.feature.spectral_contrast(y = X, sr = sample_rate, hop_length = hot_len).T # t * 7
    # 过零率
    zcrs = librosa.feature.zero_crossing_rate(y=X, frame_length= frame_len, hop_length = hot_len).T  # t * 1
    # 谱平面
    flatness = librosa.feature.spectral_flatness(y = X, hop_length = hot_len).T # t * 1
    # 频谱质心
    cent = librosa.feature.spectral_centroid(y = X, sr = sample_rate)
    cent = cent / np.sum(cent)
    # 色谱图
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, sr = sample_rate).T, axis = 0)
    # 梅尔频率
    mel = np.mean(librosa.feature.melspectrogram(X, sr = sample_rate).T, axis = 0)
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
    file = ['03-01-08-02-01-02-01.wav']
    for i in file:
        features = extract_features('/mnt/hdd18.2t/dataset/RAVDESS/Actor_01/'+ i)
        data.append([features, 1])
    for i in data:
        print(i[0][0].shape, i[1])