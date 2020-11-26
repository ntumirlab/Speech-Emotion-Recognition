from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import yaml
# own package
import utils.opts as opts

# Accuracy
def accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def load_model(config, name):
    with open( config.models_path + name+".sav", 'rb') as pickle_file:
        return pickle.load(pickle_file)

def test_model(model_list, f_list):
    predict = pd.DataFrame(columns=['predict'])
    for i in tqdm(range(len(f_list))):
        feature = f_list[i]
        score_list = []
        for m in model_list:
            score_list.append(m.score(feature))

        max_score = max(score_list)
        max_score_index = score_list.index(max_score)
        predict.loc[i] =  config.class_labels[max_score_index]# max_score_index
    
    return predict

def evaluate(config):
    model_list = []
    model_names = [ name + '_model' for name in config.class_labels]
    for model_name in model_names:
        model_list.append(load_model(config, model_name))

    with open( config.pickle_path + 'test_feature_list.pickle', 'rb') as handle:
        test_feature_list = pickle.load(handle)
    
    with open( config.pickle_path + 'test_label.pickle', 'rb') as handle:
        test_label = pickle.load(handle)

    predict  = test_model(model_list,test_feature_list)

    # acc
    c = confusion_matrix(test_label, predict)
    aver_acc = accuracy(c)
    cm = c.astype('float') / c.sum(axis=1)[:, np.newaxis]*100
    print("aver_acc:{}%".format(round(aver_acc*100, 2)))
    for i, lname in enumerate(config.class_labels):
        print('{} acc: {}%'.format(lname , round(cm[i][i], 2)))

if __name__ == "__main__":
    config = opts.parse_opt()
    evaluate(config)