import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
'''
函数功能：输入DF数据集（最后一列为标签），返回特征矩阵和标签矩阵
'''
def get_Mat(dataSet):
    xMat = np.mat(dataSet.iloc[:,: -1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat

'''
函数功能：计算局部加权线性回归的预测值
参数说明：
    testMat：测试集
    xMat：训练集的特征矩阵
    yMat：训练集的标签矩阵
返回：
    yHat：函数预测值
'''
def LWLR(testMat,xMat,yMat,k=1.0):
    n = testMat.shape[0]
    m = xMat.shape[0]
    # np.eye(m)，生成m行m列的对角矩阵，对角线为1
    weights = np.mat(np.eye(m))
    yHat = np.zeros(n)
    for i in range(n):
        for j in range(m):
            # 根据公式求出权重值
            diffMat = testMat[i] - xMat[j]
            weights[j,j] = np.exp(diffMat*diffMat.T/(-2*k**2))
        
        xTx = xMat.T*(weights*xMat)
        if np.linalg.det(xTx) == 0:
            print('矩阵为奇异矩阵，不能求逆')
            return
        ws = xTx.I*(xMat.T*(weights*yMat))# 加权后的回归系数
        yHat[i] = testMat[i]*ws
    return yHat


'''
函数功能：切分训练集和测试集
参数说明：
    dataSet：原始数据集
    rate：训练集比例
返回：
    train，test：切分好的训练集和测试集
'''
def randSplit(dataSet,rate):
    m = dataSet.shape[0]
    n = int(m*rate)
    train = dataSet.iloc[:n,:]
    test = dataSet.iloc[n:m,:]
    test.index = range(test.shape[0])# 更改测试集的索引
    return train,test

'''
函数功能：计算误差平方和SSE
参数说明：
    yMat:真实值
    yHat：估计值
返回：
    SSE：误差平方和
'''
def sseCal(yMat,yHat):
    sse = ((yMat.A.flatten()-yHat)**2).sum()
    return sse

'''
画图

函数功能：绘制不同k取值下，训练集和测试集的SSE
'''
def showPlot(abalone):
    abX,abY = get_Mat(abalone)
    train_sse = []
    test_sse = []
    for k in np.arange(0.5,10.1,0.1):
        yHat1 = LWLR(abX[:99],abX[:99],abY[:99],k)
        sse1 = sseCal(abY[:99],yHat1)
        train_sse.append(sse1)
        
        yHat2 = LWLR(abX[100:199],abX[:99],abY[:99],k)
        sse2 = sseCal(abY[100:199],yHat2)
        test_sse.append(sse2)
        
    plt.plot(np.arange(0.5,10.1,0.1),train_sse,color='b')
    plt.plot(np.arange(0.5,10.1,0.1),test_sse,color='r') 
    
    plt.xlabel('不同k取值')
    plt.ylabel('SSE')
    
    plt.legend(['train_sse','test_sse'])

'''
由上图得到的训练集和测试集的SSE曲线可得，两曲线的交点在k=2处左右。
所有可设k=2

因为数据量太大，计算速度极慢，所以上图方法选择前100个数据作为训练集，第100-200个数据作为测试集
'''
def compare(abalone):
    abX,abY = get_Mat(abalone)
    # 预测出的年龄
    yHat2 = LWLR(abX[100:199],abX[:99],abY[:99],k=2)

if __name__ =='__main__':
    abalone = pd.read_table('abalone.txt',header = None)
    train,test = randSplit(abalone,0.8)