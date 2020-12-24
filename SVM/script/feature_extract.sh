#!/bin/bash
#vi .bash_profile
PATH=$PATH:$HOME/bin

openSMILEPath=/home/victor/Project/util/opensmile # opensmile install path
feature_op_Path=/home/victor/Project/Speech-Emotion-Recognition/SVM/feature/opensmile/
feature_svm_Path=/home/victor/Project/Speech-Emotion-Recognition/SVM/feature/svm/

if [[ $1 == "emodb" ]]
then
dir=/mnt/E/dataset/emodb # Dataset path
Open_OPATH=$feature_op_Path/EmoDB  # Feature output path
SVM_OPATH=$feature_svm_Path/EmoDB # SVM Feature Output Path
elif [[ $1 == "casia" ]]
then
dir=/mnt/E/dataset/CASIA/full # Dataset path
Open_OPATH=$feature_op_Path/CASIA  # Feature output path
SVM_OPATH=$feature_svm_Path/CASIA # SVM Feature Output Path
elif [[ $1 == "nnime" ]]
then
dir=/mnt/E/dataset/NNIME/RecordingsofAudio/SentenceLevel/Speech # Dataset path
Open_OPATH=$feature_op_Path/nnime  # Feature output path
SVM_OPATH=$feature_svm_Path/nnime # SVM Feature Output Path
elif [[ $1 == "iemocap" ]]
then
dir=/mnt/E/dataset/IEMOCAP_full_release/selected_wav # Dataset path
Open_OPATH=$feature_op_Path/iemocap  # Feature output path
SVM_OPATH=$feature_svm_Path/iemocap # SVM Feature Output Path
fi

rm -rf $Open_OPATH/*
rm -rf $SVM_OPATH/*

# 1. Feature Extract by openSmile
for wav in $(ls $dir); do
    $openSMILEPath/build/progsrc/smilextract/SMILExtract -C $openSMILEPath/config/is09-13/IS09_emotion.conf -I $dir/$wav -O $Open_OPATH/$wav.txt
    echo "$wav is extracted"
done

echo "opensmile: $1 finished!"

# 2.Convert Feature into SVM style
for wav in $(ls $Open_OPATH); do

    if [[ $1 == "emodb" ]]
    then
    python script/arff2svm_emodb.py $Open_OPATH/$wav $SVM_OPATH/$wav
    elif [[ $1 == "casia" ]]
    then
    python script/arff2svm_casia.py $Open_OPATH/$wav $SVM_OPATH/$wav
    elif [[ $1 == "iemocap" ]]
    then
    python script/arff2svm_iemocap.py $Open_OPATH/$wav $SVM_OPATH/$wav
    fi
    echo "$wav is converted"
done

echo "convert: $1 finished!"
