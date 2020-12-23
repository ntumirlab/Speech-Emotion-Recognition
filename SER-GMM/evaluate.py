import time

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# own package
import utils.opts as opts

# Accuracy
def unwegiht_accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

<<<<<<< HEAD
def wegiht_accuracy(confusion_matrix):
    w_acc = 0
    for i, num in enumerate(confusion_matrix):
        acc = confusion_matrix[i][i]
        row_sum = confusion_matrix[i].sum()
        lab_acc = acc / row_sum
        w_acc += lab_acc
    return w_acc / len(confusion_matrix)

=======
>>>>>>> 86b42fe65c528ef862971c03dae3f53defe778d0
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

<<<<<<< HEAD
def evaluate(config,comp):
    model_list = []
    model_names = [ name + '_model'+str(comp) for name in config.class_labels]
    for model_name in model_names:
        model_list.append(load_model(config, model_name))

    with open( config.pickle_path + 'X_test_feature_list.pickle', 'rb') as handle:
        test_feature_list = pickle.load(handle)
    
    with open( config.pickle_path + 'y_test.pickle', 'rb') as handle:
=======
def evaluate(config):
    model_list = []
    model_names = [ name + '_model' for name in config.class_labels]
    for model_name in model_names:
        model_list.append(load_model(config, model_name))

    with open( config.pickle_path + 'test_feature_list.pickle', 'rb') as handle:
        test_feature_list = pickle.load(handle)
    
    with open( config.pickle_path + 'test_label.pickle', 'rb') as handle:
>>>>>>> 86b42fe65c528ef862971c03dae3f53defe778d0
        test_label = pickle.load(handle)

    predict  = test_model(model_list,test_feature_list)

    # acc
    c = confusion_matrix(test_label, predict)
    u_acc = unwegiht_accuracy(c)
    w_acc = wegiht_accuracy(c)

    cm = c.astype('float') / c.sum(axis=1)[:, np.newaxis]*100
<<<<<<< HEAD

    for first_index in range(len(cm)): 
        for second_index in range(len(cm[first_index])): 
            cm[first_index][second_index] = round(cm[first_index][second_index], 1)

    plt.figure(figsize=(12, 8))
    cm = pd.DataFrame(cm , index = [i for i in config.class_labels] , columns = [i for i in config.class_labels])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix'+' Component('+str(comp)+')', size=20)
    plt.xlabel('Predicted Labels \n Unweight Accuracy:{}% \n Weight Accuracy:{}%'.format(round(u_acc*100, 2), round(w_acc*100, 2)), size=14)
    plt.ylabel('Actual Labels', size=14)

    localtime = time.asctime( time.localtime(time.time()) )

    plt.savefig(config.res_path + 'res '+ str(localtime)+ 'com'+str(comp) +'.png')

if __name__ == "__main__":
    config = opts.parse_opt()
    for com  in config.component_num:
        print("evaluate with component:", com)
        evaluate(config,com)
=======

    print(cm)
    print("aver_acc:{}%".format(round(aver_acc*100, 2)))
    for i, lname in enumerate(config.class_labels):
        print('{} acc: {}%'.format(lname , round(cm[i][i], 2)))

if __name__ == "__main__":
    config = opts.parse_opt()
    evaluate(config)
>>>>>>> 86b42fe65c528ef862971c03dae3f53defe778d0
