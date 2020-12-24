data_set=$1
model_path=models/$data_set/train.scale.model
traScale_path=scales/$data_set/train.scale
tesScale_path=scales/$data_set/test.scale
traOut_path=outs/$data_set/train.scale.out
tesOut_path=outs/$data_set/test.scale.out
# 1.Train
svm-train -b 1 -c $2 -g $3 $traScale_path $model_path
# 2.Output
svm-predict -b 1 $traScale_path $model_path result/res/$data_set/$4/Result_train$5.txt > result/accuracy/$data_set/$4/Accuracy_train$5.txt
svm-predict -b 1 $tesScale_path $model_path result/res/$data_set/$4/Result_test$5.txt > result/accuracy/$data_set/$4/Accuracy_test$5.txt