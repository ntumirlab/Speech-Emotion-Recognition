import os
import util.opts as opts

def svmScale(config, foldType, num):

    # 1. scale the feature
    foldPath = config.fold_path
    scalePath = config.scale_path
    oTraScalePath = config.o_tra_scale_path
    oTesScalePath = config.o_tes_scale_path
    outPath = config.out_path
    pngPath = config.png_path
    svm_scale_s = 'svm-scale -s '+ scalePath + ' ' + foldPath + '/' + foldType + '/train' + num + '.txt > ' + oTraScalePath
    svm_scale_r = 'svm-scale -r '+ scalePath + ' ' + foldPath + '/' + foldType + '/test' + num + '.txt > ' + oTesScalePath
    print(svm_scale_s)
    print(svm_scale_r)
    os.system(svm_scale_s)
    os.system(svm_scale_r)
    print('------------')
    grid_cmd = 'python util/grid.py ' + foldType + ' ' + num + ' -out '+ outPath + ' -png '+ pngPath + ' '+ oTraScalePath
    os.system(grid_cmd)
    print("grid_cmd:", grid_cmd)

if __name__ == "__main__":
    conf = opts.parse_opt()
    l_foldType = ['5fold', '10fold']
    fold_num = ['0', '1', '2', '3', '4']
    emodb_speakers = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]

    casia_speakers = ["liuchanhg", "wangzhe", "zhaoquanyin", "ZhaoZuoxiang"]

    # for i,n in enumerate(fold_num):
    #     svmScale(conf, l_foldType[0], n)

    # for i,n in enumerate(emodb_speakers):
    #     svmScale(conf, l_foldType[1], n)
    for i,n in enumerate(casia_speakers):
        svmScale(conf, l_foldType[1], n)

    print( conf.dataset_name + ' ----- Ending --------- ')