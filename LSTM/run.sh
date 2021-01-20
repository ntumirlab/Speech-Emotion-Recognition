#! /bin/sh

python3 train.py --config $1
python3 plot_confusion_matrix.py --config $1
python3 plot_loss.py --config $1