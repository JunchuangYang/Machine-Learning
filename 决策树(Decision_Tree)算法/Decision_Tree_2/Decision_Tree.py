# -*- coding=utf-8 -*-

from math import log 
import operator

'''
函数功能：创建测试数据集
参数说明：无
返回：
    dataSet：数据集
    labels：分类属性
'''
def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
            [0, 0, 0, 1, 'no'],
            [0, 1, 0, 1, 'yes'],
            [0, 1, 1, 0, 'yes'],
            [0, 0, 0, 0, 'no'],
            [1, 0, 0, 0, 'no'],
            [1, 0, 0, 1, 'no'],
            [1, 1, 1, 1, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [1, 0, 1, 2, 'yes'],
            [2, 0, 1, 2, 'yes'],
            [2, 0, 1, 1, 'yes'],
            [2, 1, 0, 1, 'yes'],
            [2, 1, 0, 2, 'yes'],
            [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  
    return dataSet , labels
             
'''
函数功能：计算给定数据集的香农熵
参数说明：
    dataSet：数据集
返回：
    shannonEnt：香农熵
'''
def calcShannonEnt(dataSet):
    numEntires = len(dataSet) # 返回数据集的行数
    labelCounts = {} # 保存每个标签（Label）出现次数的字典
    for featVec in dataSet: # 对每组的特征向量进行统计
        currentLabel = featVec[-1] # 提取标签（Label）信息
        if currentLabel not in labelCounts.keys(): # 如果标签（Label）没有放入统计次数的字典，添加进去
            labelCounts[currentLabel] = 0 
        labelCounts[currentLabel] += 1 # Lable计数
    shannonEnt = 0.0 # 初始化香农熵
    for key in labelCounts: # 计算香农熵
        prob = float(labelCounts[key]) / numEntires # 选择该标签（Label）的概率
        shannonEnt += -(prob * log(prob,2)) # 利用公式计算
    return shannonEnt # 返回香农熵

'''
函数功能：按照给定特征划分数据集
参数说明：
    dataSet：待划分的数据集
    axis：划分数据集的特征
    value：需要返回的特征的值
返回：
    retDataSet：划分后的数据集
'''
def splitDataSet(dataSet,axis,value):
    retDataSet = [] # 创建返回的数据集列表
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:]) # 去掉axis的特征
            retDataSet.append(reducedFeatVec) # 将符合条件的添加到返回的数据集
    return retDataSet # 返回划分后的数据集

'''
函数功能：选择最优特征
参数说明：
    dataSet：数据集
返回：
    bestFeature : 信息增益最大的（最优）特征的索引值
'''
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1 # 特征数量
    baseEntropy = calcShannonEnt(dataSet) # 计算数据集的香农熵
    bestInfoGain = 0.0 # 初始化最优信息增益 
    bestFeature = -1 # 初始化最优特征的索引值
    for i in range(numFeatures): # 遍历所有特征值
        featList = [example[i] for example in dataSet] # 获取dataSet第i列的所有特征值
        uniqueVals = set(featList) # 使用set进行去重
        newEntropy = 0.0 # 初始化第i个标签下的信息信息增益
        for value in uniqueVals: # 计算信息增益
            subDataSet = splitDataSet(dataSet,i,value) # 划分后的数据集
            prob = len(subDataSet) / float(len(dataSet)) # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet) # 根据公式计算香农熵
        infoGain = baseEntropy - newEntropy # 信息增益
        if bestInfoGain < infoGain:
            bestInfoGain = infoGain  # 更新信息增益，找出最大的信息增益
            bestFeature = i # 记录信息增益最大的特征的索引值
    return bestFeature  # 返回信息增益最大的特征的索引值

'''
函数功能：统计classList中出现最多的元素（类标签）
参数说明：
    classList: 类标签列表
返回：
    sortedClasscount[0][0]：出现此处最多的元素（类标签）
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:  #统计classList中每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClasscount = sorted(classCount.items(),key  = operator.itemgetter(1) , reverse = True) #根据字典的值降序排序
    return sortedClasscount[0][0] #返回classList中出现次数最多的元素

'''
函数功能：创建决策树
参数说明：
    dataset：训练数据集
    labels：分类属性标签
    featLabels： 最优特征标签
返回：
    myTree：决策树
'''
def createTree(dataSet,labels,featLabels):
    classList = [example[-1] for example in dataSet]  #取分类标签(是否放贷:yes or no)
    if classList.count(classList[0]) == len(classList): #如果类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1 or len(labels) == 0:  #遍历完所有特征时返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet) #选择最优特征
    bestFeatLabel = labels[bestFeat] # 选择最优特征的标签
    featLabels.append(bestFeatLabel) 
    myTree = {bestFeatLabel:{}} #根据最优特征的标签生成树
    del(labels[bestFeat])  #删除已经使用特征标签
    featValues = [example[bestFeat] for example in dataSet] #得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues) #去掉重复的属性值
    for value in uniqueVals:  #遍历特征，创建决策树。
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),labels,featLabels)
    return myTree
'''
函数功能：使用决策树分类
参数：
    inputTree：已经生成的决策树
    featLabels：存储选择的最优特征标签
    textVec：测试数据列表，顺序对应最优特征标签
返回 ：
    classLabel： 分类结果
'''

def classify(inputTree,featLabels,textVec):
    firstStr = next(iter(inputTree))
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if textVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    featLabels = []
    myTree = createTree(dataSet, labels, featLabels)
    print("创建的决策树----->",myTree)
    testVec = [0,0]                                        #测试数据
    result = classify(myTree, featLabels, testVec)
    if result == 'yes':
        print('放贷')
    if result == 'no':
        print('不放贷')