# -*- coding: utf-8 -*-
#__author__ = 'lenovo'


import operator
from os import listdir
import numpy as np
"""
inX:用于分类的输入向量
dataSet:输入的训练样本集
lables：标签向量
k:用于选择最近邻居的数目
"""
def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0] # 数据集的行数
    # 将inX沿x轴复制一倍，沿y轴复制dataSetSize倍，变成dataSetSize*1024的数组
    # 再与数据集dataSet相减，得到inX与训练样本的差值
    diffMat = np.tile(inX,(dataSetSize,1)) -dataSet
    sqDiffMat = diffMat**2
    # axis=1，将数组的每一行相加，得到inX与测试样本的欧式距离的平方和
    sqDistances = sqDiffMat.sum(axis=1)
    distance = sqDistances**0.5
    # 将distance中的元素从小到大排列，提取其对应的index(索引)
    sortedDistIndicies = distance.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    """
    classCount.iteritems()将classCount字典分解为元组列表，operator.itemgetter(1)按照第二个元素的次序对元组进行排序，
    reverse=True是逆序，即按照从大到小的顺序排列
    """
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

# 将图像转换成测试向量
# 把一个32*32的二进制图像矩阵转换成1*1024的向量
def img2vector(filename):
    # 返回一个用0填充的1*1024的数组
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

    return returnVect

#print(img2vector('testDigits/0_13.txt')[0,32:64])

def handwritingClassTest():
    hwLables = []
    # 获取目录trainingDigits目录下的所有文件
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        # 下面三行获取训练文件的分类标签
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errcount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        # 同上
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'% fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLables,3)
        print("the classifier came back with: %d,the real answer is :%d"%(classifierResult,classNumStr))
        if (classifierResult!=classNumStr):
            errcount += 1.0
        print("\nthe total number of errors is :%d" % errcount)
        print("\nthe total error rate is :%s"%(errcount/float(mTest)))

def main():
    handwritingClassTest()

if __name__ == '__main__':
    main()