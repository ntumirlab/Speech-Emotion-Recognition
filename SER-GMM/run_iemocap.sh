#! /bin/sh
if [[ $1 == "p" ]]
then
python3 preprocessing.py --config configs/gmm_iemocap.yaml
python3 train_ser_gmm.py --config configs/gmm_iemocap.yaml
python3 evaluate.py --config configs/gmm_iemocap.yaml
else
python3 train_ser_gmm.py --config configs/gmm_iemocap.yaml
python3 evaluate.py --config configs/gmm_iemocap.yaml
fi