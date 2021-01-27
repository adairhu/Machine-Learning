import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import operator

# 处理中文和负号
matplotlib.rcParams['font.family'] = 'SimHei' # 'SimHei'是黑体
matplotlib.rcParams['font.style'] = 'italic' # 斜体
matplotlib.rcParams['font.size'] = '12'
matplotlib.rcParams['axes.unicode_minus'] = False # 用来正常显示负号


# kNN算法
def classify0(inX, dataSet, labels, k):
    # dataSetSize = dataSet.shape[0]
    # diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    diffMat = inX - dataSet
    sqrDiffMat = diffMat ** 2
    sqrDistances = sqrDiffMat.sum(axis=1)
    distances = sqrDistances ** 0.5
    sortedDistIndicies = d3istances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] =  classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 从文本中解析训练样本矩阵和标签
def file2array(filename):
    file = open(filename, 'r')
    fileLines = file.readlines()
    numberOfLines = len(fileLines)
    dataArray = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in fileLines:
        line = line.strip() # 参数为空时，默认删除开头、结尾处空白符（包括'\n', '\r', '\t', ' ')
        listFromLine = line.split('\t')
        dataArray[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return dataArray, classLabelVector


# 归一化数据
def autoNorm(dataSet):
    minVal = dataSet.min(0) # 0代表每列
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    normDataSet = np.zeros(np.shape(dataSet))
    normDataSet = (dataSet - minVal) / ranges
    return normDataSet, ranges, minVal


# 添加legend,思路是把三种不同标签的图分开，分成三个子图，画在一张图里面
def scatterLengend(dataSet, labels, x, y):
    type1 = []
    type2 = []
    type3 = []
    for i in range(len(labels)):
        if labels[i] == 1:
            type1.append(np.array(dataSet[i]))
        elif labels[i] == 2:
            type2.append(np.array(dataSet[i]))
        else:
            type3.append(np.array(dataSet[i]))
    type1 = np.array(type1)
    type2 = np.array(type2)
    type3 = np.array(type3)

    g1 = plt.scatter(type1[:, x], type1[:, y], c='red')
    g2 = plt.scatter(type2[:, x], type2[:, y], c='yellow')
    g3 = plt.scatter(type3[:, x], type3[:, y], c='blue')
    plt.legend(handles=[g1, g2, g3], labels=['不喜欢', '魅力一般', '极具魅力'])
    plt.show()

# 测试算法
def algTest():
    testRatio = 0.1
    dataSet, labels = file2array('./datingTestSet2.txt')
    normDataSet, ranges, minVal = autoNorm(dataSet)
    numDataSet = normDataSet.shape[0]
    numTestSet = int(numDataSet * testRatio)
    errorCount = 0
    for i in range(numTestSet):
        classifyResult = classify0(normDataSet[i,:], normDataSet[numTestSet:numDataSet,:],a'q'\
                                   labels[numTestSet:numDataSet],7)
        #print('the classifier result: %d, the real answer is: %d'\
        #     %(classifyResult, labels[i]))
        if classifyResult != labels[i]:
            errorCount += 1
    print('the total error rate is: %f' % (errorCount / float(numTestSet)))

# 使用算法
def classifyPerson():
    resultList = ['不喜欢', '魅力一般', '极具魅力']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent of filer miles earned per years?'))
    iceCream = float(input('liters of ice cream consumed per years?'))
    dataSet, labels = file2array('./datingTestSet2.txt')
    normDataSet, ranges, minVal = autoNorm(dataSet)
    inArr = [ffMiles, percentTats, iceCream]
    classifyResult = classify0((inArr - minVal) / ranges, normDataSet, labels, 7)
    print('你可能对这个人的评价是：%s' % resultList[classifyResult - 1])


if __name__ == '__main__':

    #
    # data, labels = file2array('./datingTestSet2.txt')
    #
    # normDataSet = autoNorm(data)
    #
    # scatterLengend(normDataSet, labels, 0, 1)

    #algTest()
    classifyPerson()
