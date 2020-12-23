# Importing the required libraries
import os
# package
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm
from tqdm import tqdm
import pickle
import yaml
# own package
import utils.opts as opts

def train_model(config, features, comp):
    gmm = GaussianMixture(n_components=comp, tol=0.88,n_init=5).fit(features)
    labels = gmm.predict(features)
    return (gmm,labels)

def save_model(config, model,name, comp):
    save_dir = os.path.join(os.getcwd(), config.models_path)
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = name + str(comp) +'.sav'
    model_path = os.path.join(save_dir, filename)
    print(model_path)
    pickle.dump(model, open(model_path, 'wb'))

# Training
def training(config, p_list):
    print('Begining of Training & save!')
    for com  in config.component_num:
        for i, lname in enumerate(config.class_labels):
            m_name = str(lname)+"_model"
            gmm_model ,train_labels=train_model(config, p_list[i], com)
            save_model(config, gmm_model, m_name, com)
    print('finished Training & Save!')

if __name__ == "__main__":
    config = opts.parse_opt()
    pic_list = []
    for i, lname in enumerate(config.class_labels):
        pic_addr = config.pickle_path + lname +'.pickle'
        with open(pic_addr, 'rb') as p:
            pic_list.append(pickle.load(p))

    with open(config.pickle_path + 'X_test.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    
    with open(config.pickle_path + 'y_test.pickle', 'rb') as handle:
        y_test = pickle.load(handle)

    training(config, pic_list)
