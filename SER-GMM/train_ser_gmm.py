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

def train_model(features):
    gmm = GaussianMixture(n_components=5).fit(features)
    labels = gmm.predict(features)
    return (gmm,labels)

def save_model(config, model,name):
    save_dir = os.path.join(os.getcwd(), config.models_path)
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = name+'.sav'
    model_path = os.path.join(save_dir, filename)
    print(model_path)
    pickle.dump(model, open(model_path, 'wb'))

# Training
def training(p_list, config):
    print('Begining of Training & save!')
    for i, lname in enumerate(config.class_labels):
        m_name = str(lname)+"_model"
        gmm_model ,train_labels=train_model(p_list[i])
        save_model(config, gmm_model, m_name)
    print('finished Training & Save!')

if __name__ == "__main__":
    config = opts.parse_opt()
    pic_list = []
    for i, lname in enumerate(config.class_labels):
        pic_addr = config.pickle_path + lname +'.pickle'
        with open(pic_addr, 'rb') as p:
            pic_list.append(pickle.load(p))

    with open(config.pickle_path + 'test.pickle', 'rb') as handle:
        test=pickle.load(handle)
    
    with open(config.pickle_path + 'test_label.pickle', 'rb') as handle:
        test_label=pickle.load(handle)

    training(pic_list, config)