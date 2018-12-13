import numpy as np 
import pandas as pd 
import random
import re
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
函数功能：词袋模型，每遇到一次词将向量值+1
参数说明：
    vocabList: 词汇表
    inputSet：切分好的词条列表中的一条（待测样本）
返回：
    returnVec：文档向量，词集模型
'''
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

'''
函数功能: 朴素贝叶斯分类器训练函数
参数说明：
    trainMat: 训练文档矩阵
    classVec：训练类别标签向量
返回:
    p0v: 非垃圾类的条件概率数组
    p1v: 垃圾类的条件概率数组
    pAb: 文档属于垃圾类的概率
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
        if classVec[i] == 1: # 统计属于垃圾类的条件概率所需的数据
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else: # 统计属于非垃圾类的条件概率所需的数据
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1v = np.log(p1Num / p1Denom)
    p0v = np.log(p0Num / p0Denom)
    return p0v , p1v , pAb  #返回属于非垃圾类，垃圾类和文档属于垃圾类的概率

def classifyNB(vec2classify,p0V,p1V,pAb):
    p1 = sum(vec2classify * p1V) +  np.log(pAb) # 对应元素相乘-->根据log函数的特点（loga + logb = log a*b）
    p0 = sum(vec2classify * p0V) +  np.log(1-pAb)
    if p1 > p0:
        return 1
    else:
        return 0

'''
函数功能：将输入来的长字符串切分成单词
参数：
    bigString : 待切分的长字符串
返回：
    切分好的单词
'''
def textParse(bigString):
    listOfTokens = re.split(r'\W+',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) >2] # 去除长度小于2的单词，并将单词转变为小写
def spamText():
    docList = [] # 邮件中所有的单词
    classList = [] #  所有的邮件分类
    fullText = [] #　所有的邮件文本
    for i in range(1,26): # 遍历所有的垃圾和非垃圾邮件
        wordList = textParse(open('email/spam/%d.txt'%i,encoding = "ISO-8859-1").read()) #  读取垃圾邮件的内容
        docList.append(wordList) # 添加到单词表中
        fullText.extend(wordList) # 添加的邮件文本中
        classList.append(1) # 添加分类：1：表示垃圾邮件
        wordList = textParse(open('email/ham/%d.txt'%i,encoding = "ISO-8859-1").read()) # 读取非垃圾邮件的内容
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)         #添加分类：0：表示非垃圾邮件
    
    vocabList = createVocabList(docList) # 创建不重复的单词表
    trainingSet = list(range(50)) # 训练集大小
    testSet = [] # 测试集列表
    for i in range(10): 
        randIndex = int(np.random.uniform(0,len(trainingSet))) ## 随机选择10个作为测试样本，索引
        testSet.append(trainingSet[randIndex]) # 添加到测试集 ， 索引
        del(trainingSet[randIndex])# 从训练集中删除
    trainMat = [] # 训练集样本矩阵
    trainClasses = [] # 训练集标签
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex])) # 将每一个邮件文本向量化
        trainClasses.append(classList[docIndex]) # 邮件标签
    p0V , p1V , pSpam = trainNB(trainMat , trainClasses) # 开始训练
    
    errorCount = 0 # 错误数
    for docIndex in testSet: # 遍历测试集
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        predict = classifyNB(wordVector,p0V,p1V,pSpam)
        if  predict != classList[docIndex] :
            errorCount += 1
            #print("classification error:",docList[docIndex]) # 分类错误的邮件
            print("Original:%d-----Predict:%d"%(classList[docIndex],predict))
    print('the error rate is: ', float(errorCount)/len(testSet)) #输出错误率           
    print("*"*50)
    
if __name__ == '__main__':
    for i in range(10):
        print("第%d次训练"%(i+1))
        spamText()


'''
代码中，采用随机选择的方法从数据集中选择训练集，剩余的作为测试集。
这种方法的好处是，可以进行多次随机选择，得到不同的训练集和测试集，从而得到多次不同的错误率，
我们可以通过多次的迭代，求取平均错误率，这样就能得到更准确的错误率。
这种方法称为留存交叉验证
'''