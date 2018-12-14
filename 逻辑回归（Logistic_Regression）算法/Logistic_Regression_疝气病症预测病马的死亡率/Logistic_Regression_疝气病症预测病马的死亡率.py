import numpy as np
import math
import pandas as pd
'''
定义辅助函数：
函数功能：计算sigmoid函数值
参数说明：
    inX：数值型数据
返回：
    s：经过sigmoid函数计算后的函数值
'''
def sigmoid(inX):
    s = 1/(1+ np.exp(-inX))
    return s

'''
为什么要标准化，见（pdf）
标准化函数
函数功能：标准化（期望为0，方差为1）
参数说明：
    xMat：特征矩阵
返回：
    inMat: 标准化之后的特征函数
'''
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,axis = 0) # axix = 0 ,压缩行，对各列求均值，返回1*n的矩阵 
    #print(inMeans)---->[[0.0300122 6.57611  ]]
    inVar = np.std(inMat,axis = 0) #求每一列的标准差
    #print(inVar)---->[[1.16447043 4.6216594 ]]
    inMat = (inMat - inMeans)/inVar # 标准化
    return inMat
'''
函数功能： 使用BGD求解逻辑回归

使用的梯度下降法的矩阵方式（见pdf）

参数说明：
    dataSet ： DF数据集
    alpha ： 步长
    maxCycle：最大迭代次数
返回：
    weights： 各特征权重值
'''
def BGD_LR(dataSet,alpha = 0.001,maxCycles = 500):
    xMat = np.mat(dataSet.iloc[:,:-1].values) # 提取出特征值
    yMat = np.mat(dataSet.iloc[:,-1].values).T # 提取出标签的值并转置
    xMat = regularize(xMat) # 标准化
    m , n = xMat.shape # 行，列:100,2
    weights = np.zeros((n,1)) # 生成一个n*1的列表
    # print(weights)---->[[0.]
    #                     [0.]
    #                     ]
    for i in range(maxCycles): # 迭代maxCycles次
        grad = xMat.T * (xMat * weights - yMat) / m # 各个特征的梯度
        weights = weights - alpha * grad # 更新权重值
    return weights


'''
函数功能： 使用SGD求解逻辑回归

使用的梯度下降法的矩阵方式（见pdf）

参数说明：
    dataSet ： DF数据集
    alpha ： 步长
    maxCycle：最大迭代次数
返回：
    weights： 各特征权重值
'''
def SGD_LR(dataSet,alpha = 0.001,maxCycles = 500):
    dataSet = dataSet.sample(maxCycles,replace = True)
    # maxCycles的含义是抽样的个数，是整数;smaple()函数中的replace参数的意思是：是否允许抽样值重复
    dataSet.index = range(dataSet.shape[0]) # 重新编排索引值
    xMat = np.mat(dataSet.iloc[:,:-1].values) # 提取出特征值
    yMat = np.mat(dataSet.iloc[:,-1].values).T # 提取出标签的值并转置
    xMat = regularize(xMat) # 标准化
    m , n = xMat.shape # 行，列:100,2
    weights = np.zeros((n,1)) # 生成一个n*1的列表
    # print(weights)---->[[0.]
    #                     [0.]
    #                     ]
    for i in range(m): # 迭代m次
        grad = xMat.T * (xMat * weights - yMat) / m # 各个特征的梯度
        weights = weights - alpha * grad # 更新权重值
    return weights

'''
函数功能: 给定测试数据和权重，返回类别标签
参数说明:
    inX : 测试数据
    weights: 特征权重
返回：
    标签
'''
def classify(inX,weights):
    p = sigmoid(inX * weights)
    if p < 0.5:
        return 0
    else:
        return 1

'''
函数功能：logistic分类模型 
参数说明：    
    train：测试集    
    test：训练集    
    method: 训练方法
    alpha：步长    
    maxCycles：大迭代次数 
返回：    
    retest:预测好标签的测试集
'''
def get_acc(train , test ,method, alpha = 0.001 , maxCycles = 5000):
    weights = method(train,alpha=alpha , maxCycles=maxCycles)
    xMat = np.mat(test.iloc[:,:-1].values)
    xMat = regularize(xMat)
    result = []
    for inX in xMat:
        label = classify(inX,weights)
        result.append(label)
    retest = test.copy()
    retest['predict'] = result
    acc = (retest.iloc[:,-1] == retest.iloc[:,-2]).mean()
    #print(f'模型准确率：{acc}')
    return acc
    #return retest
        
if __name__ == '__main__':
    test = pd.read_table('horseColicTest.txt',header=None)
    train = pd.read_table('horseColicTraining.txt',header=None)
    for i in range(20):
        print(f"第{i+1}次训练：")
        print("使用BGD方法：准确率{%.6f}"%get_acc(train,test,BGD_LR,alpha=0.001,maxCycles=5000))
        print("使用SGD方法：准确率{%.6f}"%get_acc(train,test,SGD_LR,alpha=0.001,maxCycles=5000))

'''
数据如果多的话一般随机梯度下降法（SGD）要比批次梯度下降法（BGD）要好
'''