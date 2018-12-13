import numpy as np 
import pandas as pd 
import random
from functools import reduce

'''
函数功能： 创建实验数据集
参数说明： 无
返回：
    dataSet：切分好的样本词条
    classVec：类标签向量
'''
def loadDataSet():
    dataSet =[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
              ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
              ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
              ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
              ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
              ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
             ] #切分好的词条
    classVec = [0,1,0,1,0,1] # 类别标签向量，1代表侮辱性词汇，0代表非侮辱性词汇
    return dataSet , classVec

'''
函数功能：将切分的样本词条整理成词汇表
参数说明：
    dataSet：切分好的样本词条
返回：
    vocabList：不重复的词汇表
'''
def createVocabList(dataSet):
    vocabSet = set() # 创建一个空的set()-->去重
    for doc in dataSet: # 遍历dataSet中的每一条言论
        vocabSet = vocabSet | set(doc) #两个set集合取并集
        vocaList = list(vocabSet)
    return vocaList # 不重复的词汇表

'''
函数功能：根据vocabList词汇表，将inputSet（待测样本）向量化，向量的每一个元素为1 or 0
参数说明：
    vocabList: 词汇表
    inputSet：切分好的词条列表中的一条（待测样本）
返回：
    returnVec：文档向量，词集模型
'''
def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0] * len(vocabList) # 创建一个其中所含元素都为0的向量
    for word in inputSet: # 遍历每个词条
        if word in inputSet: # 如果词条存在于词汇表中，则变为1
            returnVec[vocabList.index(word)] = 1
        else:
            print(f"{word} is not in my Vocabulary!")
    return returnVec #　返回文档向量


'''
函数功能：生成训练集向量列表
参数说明：
    dataSet: 切分好的样本词条
返回：
    trainMat: 所有词条向量组成的列表
'''
def get_trainMat(dataSet):
    trainMat = [] # 初始化向量列表
    vocabList = createVocabList(dataSet) #生成词汇表
    for inputSet in dataSet: # 遍历样本词条中的每一条样本
        returnVec = setOfWords2Vec(vocabList,inputSet) # 将当前词条向量化
        trainMat.append(returnVec) # 追加到向量列表中
    return trainMat

'''
函数功能: 朴素贝叶斯分类器训练函数
参数说明：
    trainMat: 训练文档矩阵
    classVec：训练类别标签向量
返回:
    p0v: 非侮辱类的条件概率数组
    p1v: 侮辱类的条件概率数组
    pAb: 文档属于侮辱类的概率
    
初始化问题：

拉普拉斯平滑（见pdf文档）

利用贝叶斯分类器对文档进行分类时，要计算多个概率的乘积以获得文档属于某个类别的概率，
即计算 p(w0|1)p(w1|1)p(w2|1)。如果其中有一个概率值为0，那么最后的成绩也为0。
显然，这样是不合理的，为了降低 这种影响，可以将所有词的出现数初始化为1，并将分母初始化为2。
这种做法就叫做拉普拉斯平滑(Laplace Smoothing)又被称为加1平滑，是比较常用的平滑方法，它就是为了解决0概率问题。
 
'''
def trainNB(trainMat,classVec):
    n = len(trainMat) # 计算训练文档的数目
    #print(n) ---> 6 
    m = len(trainMat[0]) # 计算每篇文档的词条数
    # print(m)---->32-->所有的不重复的单词共有32个
    pAb = sum(classVec)/n # 文档属于侮辱类的概率
    # print(pAb)--->0.5 ---> 3/6
    p0Num = np.ones(m) # 词条出现初始化为1-->列表长度为32
    p1Num = np.ones(m) # 词条出现初始化为1
    p0Denom = 2 # 分母初始化为2
    p1Denom = 2 # 分母初始化为2
    for i in range(n):  # 遍历每一个文档
        if classVec[i] == 1: # 统计属于侮辱类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else: # 统计属于非侮辱类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1v = np.log(p1Num / p1Denom)
    p0v = np.log(p0Num / p0Denom)
    return p0v , p1v , pAb  #返回属于非侮辱类，侮辱类和文档属于侮辱类的概率

'''
函数功能：朴素贝叶斯分类器分类函数
参数说明：
    vec2classify: 待分类的词条数组
    p0V: 非侮辱类的条件概率数组
    p1V：侮辱类的条件概率数组
    pAb：文档属于侮辱类的概率
返回：
    0：属于非侮辱类
    1：属于侮辱类
'''
def classifyNB(vec2classify,p0V,p1V,pAb):
    p1 = sum(vec2classify * p1V) +  np.log(pAb) # 对应元素相乘-->根据log函数的特点（loga + logb = log a*b）
    p0 = sum(vec2classify * p0V) +  np.log(1-pAb)
    print('p0:',p0)
    print('p1:',p1)
    if p1 > p0:
        return 1
    else:
        return 0

'''
函数功能：朴素贝叶斯测试函数
参数说明：
    testVec：待测样本
返回：
    测试样本类别
'''
def testingNB(testVec):
    dataSet,classVec = loadDataSet() # 创建实验样本
    vocabList = createVocabList(dataSet) # 创建词汇表
    trainMat = get_trainMat(dataSet) # 将实验样本向量化
    p0V , p1V ,pAb = trainNB(trainMat,classVec) # 训练朴素贝叶斯分类器
    thisone = setOfWords2Vec(vocabList,testVec) # 测试样本向量化
    if classifyNB(thisone,p0V,p1V,pAb): # 执行分类
        print(testVec,"属于侮辱类")
    else:
        print(testVec,"属于非侮辱类")

if __name__ == '__main__':
    #测试样本1 
    testVec1 = ['love', 'my', 'dalmation'] 
    testingNB(testVec1) 
    #测试样本2 
    testVec2 = ['stupid', 'garbage'] 
    testingNB(testVec2)