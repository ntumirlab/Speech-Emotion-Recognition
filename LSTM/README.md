### Environment
```
pip install -r requirements.txt
```
### Modify config file - lstm.yaml
<ol>
<li>更改相對應的dataset名稱、路徑和類別</li>
<li>更改opensmile的路徑</li>
<li>不同的特徵提取方式（feature_method）
    <ol>
    <li>o -> opensmile</li>
    <li>l -> librosa</li>
    </ol>
</li>
<li>不同模型（model）
    <ol>
    <li>lstm</li>
    <li>blstm</li>
    </ol>
</li>
<li>checkpoint_name為存model的名稱</li>
</ol>
    
### Data Preprocess
```
python preprocess.py --config configs/lstm.yaml
```
### Train
```
python train.py --config configs/lstm.yaml
```


