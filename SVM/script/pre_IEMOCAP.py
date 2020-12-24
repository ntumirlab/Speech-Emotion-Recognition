import shutil
import os

IEMOCAP_PATH = '/mnt/E/dataset/IEMOCAP_full_release'

LABEL_PATH = '/dialog/EmoEvaluation'
WAV_PATH = '/sentences/wav'
SENSSIONS = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

FULL_PATH = IEMOCAP_PATH + '/' + 'full_wav'
if not os.path.exists(FULL_PATH):
    os.makedirs(FULL_PATH)
    print('Created full_wav!')

SELECTED_PATH = IEMOCAP_PATH + '/' + 'selected_wav'
if not os.path.exists(SELECTED_PATH):
    os.makedirs(SELECTED_PATH)
    print('Created selected_wav!')

SMALL_PATH = IEMOCAP_PATH + '/' + 'small_wav'
if not os.path.exists(SELECTED_PATH):
    os.makedirs(SELECTED_PATH)
    print('Created selected_wav!')

for i, s in enumerate(SENSSIONS):
    label_txts = IEMOCAP_PATH + '/' + s + LABEL_PATH
    txt_files = [f for f in os.listdir(label_txts) if f[0]!='.' ]
    for f in txt_files:
        if f[-3:] != "txt":
            continue
        wav_files_name = f[:-4]
        wav_path = IEMOCAP_PATH + '/' + s + WAV_PATH + '/' + wav_files_name

        f_path = label_txts + '/' + f
        with open( f_path , "r") as ses:  # 打开文件
            for line in ses.readlines():
                if line[0] == "[":
                    wname = line.split('\t')[1]
                    wlab = line.split('\t')[2]

                    wpath = wav_path + '/' + wname + '.wav'
                    new_wpath = wav_path + '/' + wlab + '-' + wname + '.wav'
                    copy_path = FULL_PATH + '/' + wlab + '-' + wname + '.wav'
                    os.rename(wpath, new_wpath)
                    shutil.copyfile( new_wpath, copy_path) 
                    print("copy_wpath:",copy_path)

selected_emo = ['ang', 'hap', 'neu', 'sad']

for sw in os.listdir(FULL_PATH):
    for emo in selected_emo:
        if sw[:3] == emo:
            wpath = FULL_PATH + '/' + sw
            cpath = SELECTED_PATH + '/' + sw
            shutil.copyfile(wpath, cpath)
            print("copying:", cpath)

for sw in os.listdir(SELECTED_PATH):
    count = 0
    for emo in selected_emo:
        if sw[:3] == emo:
            if count<=100:
                wpath = SELECTED_PATH + '/' + sw
                print(wpath)
                cpath = SMALL_PATH + '/' + sw
                print(cpath)
                count += 1
            shutil.copyfile(wpath, cpath)
            print("copying:", cpath)

for emo in selected_emo:
    count = 0
    for sw in os.listdir(SELECTED_PATH):
        if (sw[:3] == emo) & (count<=100) :
            wpath = SELECTED_PATH + '/' + sw
            # print(wpath)
            cpath = SMALL_PATH + '/' + sw
            # print(cpath)
            shutil.copyfile(wpath, cpath)
            count += 1

# for sw in os.listdir(SELECTED_PATH):
#     for emo in selected_emo:
#         if sw[:3] == 'exc':
#             wpath = SELECTED_PATH + '/' + sw
#             npath = SELECTED_PATH + '/' + 'hap'+sw[3:]
#             os.rename(wpath, npath)
#             print(wpath)
#             print(npath)
            # shutil.copyfile(wpath, cpath)
            # print("copying:", cpath)