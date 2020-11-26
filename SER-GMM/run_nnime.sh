#! /bin/sh
python3 preprocessing.py --config configs/gmm_nnime.yaml
python3 train_ser_gmm.py --config configs/gmm_nnime.yaml
python3 evaluate.py --config configs/gmm_nnime.yaml