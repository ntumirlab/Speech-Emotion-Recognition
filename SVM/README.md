# SVM emotion-recognition
使用 [libsvm](https://www.csie.ntu.edu.tw/~cjlin/libsvm/) 做語音情緒識別

## 使用方法
1. 下載 [openSMILE](http://audeering.com/technology/opensmile/)
2. 解壓縮後並依照 document 進行編譯，將此資料夾放至 openSMILE-X.X.0 之中
3. 照順序執行 

`emodb/casia/iemocap`
下面以emodb舉例

```
$ bash script/feature_extract.sh emodb 
$ python script/preprocessing_emodb.py --config configs/emodb.yaml 
$ python train.py --config configs/emodb.yaml
$ python script/confusionMatrix_emodb.py --config configs/emodb.yaml
```

## 檔案說明
- `configs` 資料集路徑配置等等
- `feature/opensmile` 使用 openSMILE 的 IS09\_emotion.conf 提取的 feature
- `feature/svm` arff 處理後 libsvm 可以接受的類型
- `foldData/` 處理後的 test matrix 及 train matrix (5fold)
- `result/res` 儲存 svm-predict 的結果供計算 confusion martix 使用
- `result/accuracy/` libsvm 跑出的辨識率結果
- `script/feature_extract.sh` 將資料集中的音檔提取 feature，呼叫 `arff2svm.py` 將 opensmile 檔案轉成 libsvm 接受格式
- `script/preprocessing.py` 將 `feature/svm/` 中的檔案分為 5-fold 需要的內容
- `models` 存儲訓練之後的models
- `scales` 存儲libsvm scale之後的結果（libsvm讀進feature之後需要scale之後才能進行訓練）
- `outs` 存儲訓練日誌，以及可視化展示的結果
- `util/` libsvm 提供的 python 工具
- `train.py` 找到 svm scale回傳的最佳參數用來train，並保存起來。

## Confusion matrix
做完實驗後得到所有數據執行 `script/confusionMatrix.py` 結果存在 `result/confusion matrix/` 中
