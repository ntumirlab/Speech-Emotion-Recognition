#! /bin/sh
python3 preprocessing.py
python3 train_ser_gmm.py
python3 evaluate.py