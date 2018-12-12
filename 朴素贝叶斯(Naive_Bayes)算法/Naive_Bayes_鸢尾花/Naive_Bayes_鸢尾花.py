import numpy as np
import pandas as pd
import random



'''
函数功能：随机切分训练集和测试集
参数说明：
    dataSet: 输入的数据集
    rate：训练集所占比例
返回：
    train，test：切分好的训练集和测试集
'''
def randSplit(dataSet,rate):
    l = list(dataSet.index) #提取出索引
    random.shuffle(l) # 随机打乱索引
    dataSet.index = l # 将打乱后的索引重新赋值给原数据集
    n = dataSet.shape[0] # 总行数
    m = int(n * rate) # 训练集的数量
    #print(dataSet)
    train = dataSet.loc[range(m),:] # 提取前m个记录作为训练集
    #print(train)
    test = dataSet.loc[range(m,n),:] # 剩下的作为测试集
    dataSet.index = range(dataSet.shape[0]) # 更新原始数据集的索引
    test.index = range(test.shape[0]) # 更新测试集的索引
    return train , test

'''
函数功能：构建高斯朴素贝叶斯分类器
参数说明:
    train:训练集
    test: 测试集
返回：
    test: 最终预测结果
'''
def gnb_classify(train,test):
    labels = train.iloc[:,-1].value_counts().index # 提取训练集的标签种类
    #print(train.iloc[:,-1].value_counts()) --->Iris-virginica     43
                                            #   Iris-versicolor    41
                                           #    Iris-setosa        36
    # print(labels)--->Index(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], dtype='object')
    mean = [] # 存放每个类别的均值
    std = [] # 存放每个类别的方差
    result = [] # 存放每测试集的预测结果
    for i in labels:
        item = train.loc[train.iloc[:,-1] == i ,:] # 分别提取出每一种类别
        #print(item)
        m = item.iloc[:,: -1].mean() # 当前类别的平均值
        #print(m) --->求每一列的均值
        s = np.sum((item.iloc[:,: -1] - m)**2)/(item.shape[0]) # 当前类别的方差
        mean.append(m) # 将当前类别的平均值追加到列表
        std.append(s) # 将当前类别的方差追加到列表
    means = pd.DataFrame(mean,index = labels)
    #print(means)
    stds = pd.DataFrame(std , index = labels) # 变成DF格式，索引为类标签
    for j in range(test.shape[0]): 
        iset = test.iloc[j,: -1].tolist()  # 当前测试样例
        iprob = np.exp(-1 * (iset - means)**2/(stds*2))/(np.sqrt(2*np.pi*stds)) # 正态分布公式
        #print(iprob)
        prob = 1 # 初始化当前实例总概率
        for k in range(test.shape[1] - 1): # 遍历每个特征
            prob*= iprob[k] # 特征概率之和即为当前实例概率
            cla = prob.index[np.argmax(prob.values)] # 返回最大概率类别
        result.append(cla)
    test['predict'] = result
    acc = (test.iloc[:,-1] == test.iloc[:,-2]).mean() # 计算预测准确率
    print("模型预测准确率{%f}"%acc)
    return test


if __name__ == '__main__':
	#导入数据
    dataSet = pd.read_csv("iris.txt",header = None)
    for i in range(20):
        train,test= randSplit(dataSet, 0.8)
        gnb_classify(train,test)