import json
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import re
import pickle


class Wavset(Dataset):
    def __init__(self, mode, feature, label):
        self.mode = mode
        self.raw_data = {'feature':feature,'label':label}

    def __getitem__(self, idx):
        if self.mode == "test":
            feature = self.raw_data['feature'][idx]
            return (torch.LongTensor(feature))
        else:
            feature, label= self.raw_data['feature'][idx], self.raw_data['label'][idx]
            return (torch.from_numpy(feature), torch.LongTensor([label]))


    def __len__(self):
        return len(self.raw_data['feature'])


    def collate_fn(self, samples):
        return samples[0][0].float(), samples[0][1]


def load(mode, feature, label):
    return Wavset(mode, feature, label)

if __name__ == "__main__":
    TRAIN = 'data/emodb_valid.pkl'
    DEVICE = 'cuda:0'
    train = pickle.load(open(TRAIN, 'rb'))
    train_loader = DataLoader(train, batch_size=1, num_workers=0, shuffle=True, collate_fn=train.collate_fn)
    i = 0
    for d in train_loader:
        i += 1
        input_ids, label = [t.to(DEVICE) for t in d]
        print(i, input_ids.float(), label)
        break
