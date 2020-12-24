import sys

if(len(sys.argv) < 3):
    print("usage: ./python convertToSVM.py inputfilename outputfilename")

inputfilename = sys.argv[1]
fin = open(inputfilename,'r')
lines = fin.readlines()
fin.close()
outputfilename = sys.argv[2]
fout = open(outputfilename,'w')

beginToRead = False
for line in lines:
    if beginToRead == True:
        if len(line) > 5:# not an empty line
            dataList = line.split(',')
            wavLab_name = inputfilename.split('/')[-1][6]
            if wavLab_name != "neutral":
                labelNum = ord(wavLab_name[2]) - ord('a')
            elif wavLab_name == "neutral":
                labelNum = ord(wavLab_name[0]) - ord('a')
            resultLine = ''
            resultLine += str(labelNum)
            resultLine += ' '
            for i in range(1,len(dataList)-1):
                resultLine += str(i)
                resultLine += (":"+dataList[i]+" ")
            #print(resultLine)
            fout.write(resultLine+"\n")

    if line[0:5] == '@data':
        beginToRead = True

fout.close()

