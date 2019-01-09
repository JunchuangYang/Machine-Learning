#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import types


'''
函数功能：根据特征切分数据集合
参数说明：
    dataSet：数据集合
    feature：待切分的特征
    value：该特征的值
返回：
    mat0：切分的数据集合0
    mat1：切分的数据集合1
'''
def binSplitDataSet(dataSet,feature,value):
    # np.nonzero() 输出满足值为非0数的索引
    mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],: ]
    #print(np.nonzero(dataSet[:,feature] > value)[0]) --->[1]
    mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],: ]
    return mat0,mat1



'''
函数功能：加载数据
参数说明：
    fileMame: 文件名
返回：
    数据矩阵
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))                    #转化为float类型
        dataMat.append(fltLine)
    return dataMat



'''
函数功能：绘制数据集
参数说明：
    filename：文件名
返回：
    无
'''
def plotDataSet(filename):
    dataMat = loadDataSet(filename)                                        #加载数据集
    n = len(dataMat)                                                    #数据个数
    xcord = []; ycord = []                                                #样本点
    for i in range(n):
        if filename == 'ex0.txt':
            xcord.append(dataMat[i][1]) 
            ycord.append(dataMat[i][2])                              #样本点
        else:
            xcord.append(dataMat[i][0]) 
            ycord.append(dataMat[i][1]) 
    fig = plt.figure()
    ax = fig.add_subplot(111)                                            #添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)                #绘制样本点
    plt.title('DataSet')                                                #绘制title
    plt.xlabel('X')
    plt.show()

'''
函数功能：生成叶节点
参数说明：
    dataSet：数据集合
返回：
    目标变量的均值
'''
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])


'''
函数功能：误差估计函数
参数说明：
    dataSet：数据集合
返回：
    目标变量的总方差
'''
def regErr(dataSet):
    return np.var(dataSet[:,-1]) * np.shape(dataSet)[0]

'''
函数功能：找到数据的最佳二分切分方式
参数说明：
    dataSet: 数据集
    leafType：生成叶节点
    errType： 误差估计函数
    ops：用户定义的参数构成的元祖
返回：
    bestIndex：最佳切分特征
    bestValue：最佳特征值
'''
def chooseBestSplit(dataSet,leafType = regLeaf,errType = regErr,ops = (1,4)):
    tolS = ops[0] # tolS允许的误差下降值
    tolN = ops[1] # tolN切分的最少样本数
    #  如果当前所有值相等，则退出。（根据set集合的特性）
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    
    m,n = np.shape(dataSet) # 数据集合的行数和列数   
    S = errType(dataSet) # 默认最后一个特征为最佳切分特征，计算其误差估计
    
    bestS = float('inf') # 最佳误差，初始为最大值
    bestIndex = 0 # 最佳切分特征 索引值
    bestValue = 0 #  最佳切分特征
    
    # 遍历所有特征
    for featIndex  in range(n-1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:,featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0 , mat1 = binSplitDataSet(dataSet,featIndex,splitVal)
            # 如果划分的数据数少于用户设定的tolN，则退出
            if(np.shape(mat0)[0]<tolN) or (np.shape(mat1)[0]<tolN):
                continue
            # 计算划分后的数据集的误差估计
            newS = errType(mat0) + errType(mat1)
            
            # 如果误差估计更小，则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减小不大则退出
    if (S - bestS) < tolS:
        return None,leafType(dataSet)
    
    # 根据最佳的切分特征和特征值切分数据集合
    mat0 , mat1 = binSplitDataSet(dataSet,bestIndex,bestValue)
    
    #  如果划分的数据数少于用户设定的tolN，则退出
    if(np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0]<tolN):
        return None,leafType(dataSet)
    
    # 返回最佳切分特征索引和特征值
    return bestIndex , bestValue

'''
函数功能：回归树构建函数
参数说明：
    dataSet: 数据集
    leafType：生成叶节点
    errType： 误差估计函数
    ops：用户定义的参数构成的元祖
返回：
    retTree：构建的回归树
'''
def createTree(dataSet,leafType = regLeaf,errType = regErr,ops=(1,4)):
    # 选择最佳切分特征
    feat , val = chooseBestSplit(dataSet,leafType,errType,ops)
    #如果没有特征，则返回特征值
    if feat == None:
        return val
    # 回归树
    retTree = {}
    retTree['spInd'] = feat # 特征索引
    retTree['spVal'] = val  # 特征值
    
    # 根据特征索引和特征值将该特征一分为二
    lSet , rSet = binSplitDataSet(dataSet,feat,val)
    # 创建左子树和右子树
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    retTree['right'] = createTree(rSet,leafType,errType,ops)
    return retTree

'''
函数功能：判断测试输入的变量是否是一棵树
参数说明：
    obj：测试对象
返回：
    True or False
'''
def isTree(obj):
    return (type(obj).__name__ == 'dict')

'''
函数功能：对数进行塌陷处理（即返回树平均值）
参数说明：
    tree：树
返回：
    树的平均值
'''
def getMean(tree):
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
        
    return (tree['left'] + tree['right']) / 2.0

'''
函数功能：后剪枝
参数说明：
    tree：树
    testData：测试集
返回：
    树平均值 or 不符合剪枝条件返回树
'''
def prune(tree,testData):
    
    # 如果测试集为空，则对树进行塌陷处理
    if np.shape(testData)[0] == 0 :
        return getMean(tree)
    # 如果该树有左子树和右子树，则切分数据集
    if (isTree(tree['right']) or isTree(tree['left'])):
        lSet,rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
    # 处理左子树（剪枝）
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lSet)
    # 处理右子树（剪枝）
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rSet)
    # 如果当前的左右节点为叶节点    
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet , rSet = binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:,-1] - tree['left'],2)) + np.sum(np.power(rSet[:,-1] - tree['right'],2))
        # 计算合并的均值
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:,-1] - treeMean,2))
        # 如果合并的误差小于没有合并的误差，则合并
        if errorNoMerge > errorMerge:
            return treeMean
        else:
            return tree
    else:
        return tree
    
if __name__ == '__main__':
    train_filename = 'ex2.txt'
    train_data = loadDataSet(train_filename)
    train_Mat = np.mat(train_data)
    tree = createTree(train_Mat)
    print("剪枝前：\n")
    print(tree)
    print('*'*50)
    test_filename = 'ex2Test.txt'
    test_data = loadDataSet(test_filename)
    test_Mat = np.mat(test_data)
    print("剪枝后：\n")
    print(prune(tree,test_Mat))

