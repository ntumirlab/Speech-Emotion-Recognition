'''
提取数据集中音频的特征并保存
'''

import extract_feats.opensmile as of
import extract_feats.librosa as lf
import utils.opts as opts


if __name__ == '__main__':

    config = opts.parse_opt()
    print(config.feature_method)
    if(config.feature_method == 'o'):
        a,b,c,d = of.get_data(config, config.data_path, config.train_feature_path_opensmile, train = True)
        print(a.shape,b.shape)
    elif(config.feature_method == 'l'):
        a,b,c,d = lf.get_data(config, config.data_path, config.train_feature_path_librosa, train = True)
        print(a.shape,b.shape)