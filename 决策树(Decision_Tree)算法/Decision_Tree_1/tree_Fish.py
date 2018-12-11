# -*- coding=utf-8 -*-

# 创建数据集
import numpy as np
import pandas as pd

def createDataSet():
    row_data = {'no surfacing':[1,1,1,0,0],
                'flippers':[1,1,0,1,1],
                'fish':['yes','yes','no','no','no']}
    dataSet = pd.DataFrame(row_data)
    return dataSet

'''
函数功能：计算香农熵
参数说明：
    dataSet ：原始数据集
返回：
    end：香农熵的值
'''
def calEnt(dataSet):
    n = dataSet.shape[0] #数据总行数
    #print(dataSet)
    iset = dataSet.iloc[:,-1].value_counts() #标签的所有类别
    #print(list(iset))
    p = iset / n  # 每一类标签所占比
    #print(list(p))
    ent = (-p * np.log2(p)).sum() # 计算信息熵
    return ent 

'''
函数功能：根据信息增益选择出最佳数据集切分的列
参数说明：
    dataSet：原始数据集
返回：
    axis：数据集最佳切分列的索引
'''
# 选择最优的列进行切分
def bestSplit(dataSet):
    baseEnt = calEnt(dataSet) # 计算原始熵
    bestGain = 0 # 初始化信息增益
    axis = -1 # 初始化最佳分列，标签列
    for i in range( dataSet.shape[1] - 1): # 对特征的每一列进行循环
        levels = dataSet.iloc[:,i].value_counts().index # 提取出当前列的所有值
        #print(levels)
        ents = 0 #初始化子节点的信息熵
        for j in levels:  # 对当前列的每一个取值进行循环
            childSet = dataSet[dataSet.iloc[:,i]==j] # 某一个子节点的dataFrame
            #print(childSet)
            ent = calEnt(childSet) # 计算子节点的信息熵
            ents += (childSet.shape[0]/dataSet.shape[0])*ent # 计算当前列的信息熵
        infoGain = baseEnt - ents  #　计算当前列的信息增益
        if(infoGain > bestGain):
            bestGain = infoGain# 选取最大的信息增益
            axis = i # 最大信息增益列所在的索引
    return axis

'''
函数功能：按照给定的列划分数据集
参数说明：
    dataSet: 原始数据集
    axis: 指定的列索引
    value：指定的属性值
返回：
    redataSet：按照指定列数索引和属性值切分后的数据集
'''
def mySplit(dataSet,axis,value):
    col = dataSet.columns[axis] # col = no surfacing
    #print(col)
    redataSet = dataSet.loc[dataSet[col] == value,:].drop(col,axis=1)
    return redataSet

'''
函数功能：基于最大信息增益切分数据集，递归构建决策树
参数说明：
    dataSet: 原始数据集（最后一列是标签）
返回：
    myTree：字典形式的树
'''
def createTree(dataSet):
    featlist = list(dataSet.columns) # 提取出数据集所有的列
    # print(featlist)--->['no surfacing', 'flippers', 'fish']
    classlist = dataSet.iloc[:,-1].value_counts() # 获取最后一列类标签
    # print(list(classlist))-->[3, 2]
    # 判断最多标签数目是否等于数据集行数，或者数据集是否只有一列
    if classlist[0] == dataSet.shape[0] or dataSet.shape[1] == 1:
        return classlist.index[0] # 如果是，返回类标签
    axis = bestSplit(dataSet) # 确定出当前最佳分裂的索引
    bestfeat = featlist[axis] # 获取该索引列对应的特征
    myTree = {bestfeat:{}} # 采用字典嵌套的方式存储树信息
    del featlist[axis] # 删除当前特征
    valuelist = set(dataSet.iloc[:,axis]) # 提取最佳分列所有属性值
    for value in valuelist: # 对每一个属性值递归建树
        myTree[bestfeat][value] = createTree(mySplit(dataSet,axis,value))
        #print(myTree)
    return myTree

'''
函数功能：对一个测试实例进行分类
参数说明：
    inputTree：已经生成的决策树
    labels：存储选择的最优特征标签
    testVec：测试数据列表，顺序对应原数据
返回：
    classlabel：分类结果
'''
def classify(inputTree,labels,testVec):
    firstStr = next(iter(inputTree)) # 获取决策树的第一个节点
    secondDict = inputTree[firstStr] # 下一个字典
    featIndex = labels.index(firstStr) # 第一个节点所在列的索引
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]) == dict:
                classLabel = classify(secondDict[key],labels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


'''
函数功能：对测试集进行预测，并返回预测后的结果
函数说明：
    train: 训练集
    test：测试集
返回：
    test：预测好分类的测试集
'''
def acc_classify(train,test):
    inputTree = createTree(train) # 根据训练集生成一棵树
    labels = list(train.columns)  # 数据集所有的列名称
    result = [] 
    for i in range(test.shape[0]): # 对测试集中每一天数据进行循环
        testVec = test.iloc[i,: -1] # 测试集中的一个实例
        classLabel = classify(inputTree,labels,testVec) # 预测该实例的分类
        result.append(classLabel)#将预测结果追加到result列表中
    test['predict'] = result #aa将预测结果追加到测试集的最后一列
    acc = (test.iloc[:,-1] == test.iloc[:,-2]).mean() # 计算准确率
    print('模型预测准确率为{%.2f}'%acc)
    return test


# 测试函数
def main():
	dataSet = createDataSet()
	train = dataSet
	#.iloc：根据标签的所在位置，从0开始计数，选取列
	#.loc：根据DataFrame的具体标签选取列
	#data.iloc[0:2,8]  # ',' 前的部分标明选取的行，‘,’后的部分标明选取的列
	test = dataSet.iloc[:3,:] # 0,1,2行的数据作为测试数据
	print(acc_classify(train , test))

if __name__ == '__main__':
	main()
