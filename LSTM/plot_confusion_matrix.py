import numpy as np
from tqdm import tqdm
import torch
import pickle
from torch.utils.data import DataLoader
import torch.nn as nn
from model_2 import LSTM
import opts
import random
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./model_state/emodb/cmt.png')
    plt.show()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def eval():
    DEVICE = 'cpu'
    config = opts.parse_opt()
    model = LSTM(76, len(config.class_labels))

    eval_data = pickle.load(open('data/opensmile/'+config.dataset+'_test.pkl', 'rb'))
    eval_loader = DataLoader(eval_data, batch_size=1,num_workers=0, shuffle=False, collate_fn=eval_data.collate_fn)

    
    ckpt = torch.load('model_state/'+config.dataset+'/hidden/opensmile/ckpt.17.pt',map_location=DEVICE)
    model.load_state_dict(ckpt['state_dict'])
    #model.cuda()
    model.eval()

    m = nn.Softmax(dim=1)
    
    prediction = []
    reference = []
    pred = []
    ref = []

    valid_progress = tqdm(total=len(eval_loader))
    for d in eval_loader:
        valid_progress.update(1)
        feature, label = [t.to(DEVICE) for t in d]
        predict = model(feature)
        output = torch.argmax(m(predict), dim=1).view(-1).tolist()
        target = label.view(-1).tolist()
        prediction.extend(output)
        reference.extend(target)
    
        

    assert len(prediction) == len(reference)

    cm = confusion_matrix(torch.Tensor(reference),torch.Tensor(prediction))


    plot_confusion_matrix(cm,config.class_labels,normalize = True)


if __name__ == '__main__':
    setup_seed(666)
    eval()
