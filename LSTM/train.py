import os
import numpy as np
from keras.utils import np_utils
#import models
import extract_feats.opensmile as of
import extract_feats.librosa as lf
import utils.opts as opts
from sklearn.mixture import GaussianMixture
import pickle

'''
train(): 训练模型

输入:
	config(Class)

输出:
    model: 训练好的模型
'''
def train(config):
    x_train, x_test, y_train, y_test = [], [], [], []

    # 加载被 preprocess.py 预处理好的特征
    if(config.feature_method == 'o'):
        x_train, x_test, y_train, y_test = of.load_feature(config, config.train_feature_path_opensmile, train = True)

    elif(config.feature_method == 'l'):
        x_train, x_test, y_train, y_test = lf.load_feature(config, config.train_feature_path_librosa, train = True)
    print(np.array(x_train).shape, np.array(x_test).shape, np.array(y_train).shape, np.array(y_test).shape)


    gmm = GaussianMixture(n_components = 5, tol = 0.88,n_init=5 ).fit(x_train)

    labels = gmm.predict(x_train)
    print(labels)


    # x_train, x_test (n_samples, n_feats)
    # y_train, y_test (n_samples)

    # 搭建模型
    print(x_train.shape[1])
    model = models.setup(config = config, n_feats = x_train.shape[1])

    # 训练模型
    print('----- start training', config.model, '-----')
    if config.model in ['blstm', 'lstm', 'cnn1d', 'cnn2d']:
        y_train, y_val = np_utils.to_categorical(y_train), np_utils.to_categorical(y_test) # 独热编码
        print(np.array(x_train).shape, np.array(x_test).shape, np.array(y_train).shape, np.array(y_test).shape)
        #models/dnn/dnn.py裡面 因為LSTM繼承DNN_Model
        model.train(
            x_train, y_train,
            batch_size = config.batch_size,
            n_epochs = config.epochs
        )
    else:
        model.train(x_train, y_train)
    print('----- end training ', config.model, ' -----')

    # 验证模型 models/common.py裡面 因為dnn_model繼承common_model
    model.evaluate(x_test, y_test)
    # 保存训练好的模型
    model.save_model(config)


def train_model(features):
    gmm = GaussianMixture(n_components = 5, tol = 0.88,n_init=5 ).fit(features)
    labels = gmm.predict(features)
    return (gmm,labels)

def save_model(model,name):
    save_dir = os.path.join(os.getcwd(), '/home/victor/Project/Speech-Emotion-Recognition/LSTM/checkpoints/')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = name+'.sav'
    model_path = os.path.join(save_dir, filename)
    print(model_path)
    pickle.dump(model, open(model_path, 'wb'))

def training(p_list):
    print('Begining of Training & save!')
    class_labels = ["anger", "boredom", "disgust", "neutral", "fear", "happiness", "sadness"]
    for i, lname in enumerate(class_labels):
        m_name = str(lname)+"_model"
        gmm_model ,train_labels=train_model(p_list[i])
        save_model(gmm_model, m_name)

    print('finished Training & Save!')

if __name__ == '__main__':

    config = opts.parse_opt()
    train(config)