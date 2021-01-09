from os import listdir
import numpy as np
from kNN import classify0


# 将图像文本转化为向量
def img2vector(filename):
    file = open(filename)
    imgVector = np.zeros((1, 1024))
    for i in range(32):
        lineStr = file.readline()
        for j in range(32):
            imgVector[0, i*32+j] = lineStr[j]
    return imgVector


# 读取文件夹下所有文件并创建数据集和标签
def file2array(dirname):
    fileList = listdir(dirname)
    numOfLabels = len(fileList)
    labels = []
    dataSet = np.zeros((numOfLabels, 1024))
    for i in range(numOfLabels):
        fileNameStr = fileList[i]
        fileName = fileNameStr.split('.')[0]
        label = int(fileName.split('_')[0])
        labels.append(label)
        dataSet[i,:] = img2vector(dirname + fileNameStr)
    return dataSet, labels


# 手写数字识别算法测试
def handwritingAlgTest():
    trainingSet, trainingLabels = file2array('./digits/trainingDigits/')
    testSet, testLabels = file2array('./digits/testDigits/')
    numTestSet = testSet.shape[0]
    errorCount = 0
    for i in range(numTestSet):
        classifyResult = classify0(testSet[i,:], trainingSet, trainingLabels, 5)
        # print('the classifier result is: %d, the real answer is: %d'\
        #       % (classifyResult, testLabels[i]))
        if classifyResult != testLabels[i]:
            errorCount += 1
    print('错误率：%f' % (errorCount / float(numTestSet)))

if __name__ == '__main__':
    handwritingAlgTest()
    print('ok')