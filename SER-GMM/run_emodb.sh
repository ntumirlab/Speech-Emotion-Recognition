#! /bin/sh
if [[ $1 == "p" ]]
then
python3 preprocessing.py --config configs/gmm_emodb.yaml
python3 train_ser_gmm.py --config configs/gmm_emodb.yaml
python3 evaluate.py --config configs/gmm_emodb.yaml
else
python3 train_ser_gmm.py --config configs/gmm_emodb.yaml
python3 evaluate.py --config configs/gmm_emodb.yaml
fi
