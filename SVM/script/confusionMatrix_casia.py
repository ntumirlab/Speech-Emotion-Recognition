import os,sys
from os import listdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

emotionDict = {ord('g') - ord('a'): 0, ord('a') - ord('a'): 1, ord('p') - ord('a'): 2,  ord('u') - ord('a'): 3, ord('d') - ord('a'): 4, ord('r') - ord('a'): 5}
emotions = ["anger", "fear", "happy", "neutral", "sad", "surprise"]

oriPath = ["foldData/casia/5fold"]

def unwegiht_accuracy(confusion_matrix):
    diagonal_sum = confusion_matrix.trace()
    sum_of_all_elements = confusion_matrix.sum()
    return diagonal_sum / sum_of_all_elements

def wegiht_accuracy(confusion_matrix):
    w_acc = 0
    for i, num in enumerate(confusion_matrix):
        acc = confusion_matrix[i][i]
        row_sum = confusion_matrix[i].sum()
        lab_acc = acc / row_sum
        w_acc += lab_acc
    return w_acc / len(confusion_matrix)

def draw(confusion_m, f, labels, path):
    # acc
    u_acc = unwegiht_accuracy(confusion_m)
    w_acc = wegiht_accuracy(confusion_m)

    cm = confusion_m.astype('float') / confusion_m.sum(axis=1)[:, np.newaxis]*100
   
    for first_index in range(len(cm)): 
        for second_index in range(len(cm[first_index])): 
            cm[first_index][second_index] = round(cm[first_index][second_index], 1)
    plt.figure(figsize=(12, 8))
    cm = pd.DataFrame(cm , index = [i for i in labels] , columns = [i for i in labels])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix'+' ('+str(f[:-4])+')', size=20)
    plt.xlabel('Predicted Labels \n Unweight Accuracy:{}% \n Weight Accuracy:{}%'.format(round(u_acc*100, 2), round(w_acc*100, 2)), size=14)
    plt.ylabel('Actual Labels', size=14)

    plt.savefig(path + 'res ' + str(f[:-4]) +'.png')

for path in oriPath:
    for f in listdir(path):
        testLabel = []
        resultLabel = []
        confusionMatrix = [[0] * 6 for _ in range(6)]
        oriName = path + "/" + f
        resultName = "result/res/" + path[9:] + "/" + "Result_" + f

        # read file
        with open(oriName, "r") as fin:
            lines = fin.readlines()
        for line in lines:
            testLabel.append(line.split(" ")[0])
        with open(resultName, "r") as fin:
            fin.readline() # skip line 1
            lines = fin.readlines()
        for line in lines:
            resultLabel.append(line.split(" ")[0])
        # caculate the confusion martix
        for i in range(len(testLabel)):
            x = testLabel[i]
            y = resultLabel[i]
            x_ = emotionDict[int(x)]
            y_ = emotionDict[int(y)]
            confusionMatrix[x_][y_] += 1
        # ouptut format
        outputTxt = "                        |"
        for emotion in emotions:
            outputTxt += "%15s|" % (emotion)
        outputTxt += "\n"
        for i in range(len(confusionMatrix)):
            sum_ = sum(confusionMatrix[i])
            outputTxt += "%15s(%3d/%3d)|" % (emotions[i], confusionMatrix[i][i], sum_)
            for j in confusionMatrix[i]:
                accuracy = float(j) / sum_ if sum_ != 0 else 0
                text = "     %3d (%.2f)" % (j, accuracy)
                outputTxt += text + "|"
            outputTxt += ("\n")
            
        wacc = wegiht_accuracy(np.array(confusionMatrix))
        uacc = unwegiht_accuracy(np.array(confusionMatrix))
        outputTxt += "\n wegiht_accuracy: %3f" % (wacc)
        outputTxt += "\n unwegiht_accuracy: %3f" % (uacc)

        save_path = "result/confusion matrix/" + path[9:] + "/"

        draw(np.array(confusionMatrix), f, emotions, save_path)

        with open(save_path + f, "w") as fout:
            fout.write(outputTxt)

# confusionMatrix = np.array(confusionMatrix)
# print(confusionMatrix)
