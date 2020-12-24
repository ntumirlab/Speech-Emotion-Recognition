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
            labelNum = ord(inputfilename[-10]) - ord('A')
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

