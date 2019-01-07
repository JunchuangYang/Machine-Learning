import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['simhei']
#%matplotlib inline

def get_Mat(dataSet):
    xMat = np.mat(dataSet.iloc[:,: -1].values)
    yMat = np.mat(dataSet.iloc[:,-1].values).T
    return xMat,yMat

# 岭回归

'''
函数功能：求回归系数
参数说明：
    xMat：训练数据特征
    yMat：训练数据标签
    lam：公式中lambda的值
返回：
    ws：回归系数
'''
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0:
        print('行列式为0，奇异矩阵，不能做逆')
        return
    ws = denom.I * (xMat.T * yMat)
    return ws

'''
函数功能：获取在不同λ下的回归系数
参数说明：
    xArr：训练数据的特征
    yArr：训练数据的标签
返回：
    wMat：每个特征在不同λ下的回归系数
'''
def ridgeTest(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yMean = np.mean(yMat,0) # 该函数第二个参数（压缩）=0表示对各列求平均值得到1*n的矩阵，=1表示对给行求平均值m*1矩阵
    yMat = yMat - yMean
    xMeans = np.mean(xMat,0) 
    xVar = np.var(xMat,0) # 每一列 var方差，第二个参数=0表示求样本的无偏估计值(除以N-1)，=1求方差(除以N)   cov协方差
    xMat = (xMat - xMeans)/xVar
    # 上述代码是对xArr和yArr做标准化处理
    
    numTestPts = 30
    wMat = np.zeros((numTestPts,np.shape(xMat)[1]))
    
    for i in range(numTestPts): # λ值改变
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))# 行列格式一样但处理了的数据集 ，行列格式一样但处理了的目标值 ， e的i-10次方
        wMat[i,:] = ws.T # 将第i次每个特征的回归系数向量按行保存到30次测试的第i行
    return wMat


'''
函数功能：真实值和预测值 的平方误差和
参数：
    yArr：真实值
    yHatArr：预测值
返回：
    平方误差和
'''
def rssError(yArr,yHatArr):
    return ((yArr-yHatArr)**2).sum()


'''
函数功能：将数据标准化
参数：
    xMat：训练数据
返回：
    inMat：标准化后的数据
'''
def regularize(xMat):
    inMat = xMat.copy()
    inMeans = np.mean(inMat,0) #计算平均数，然后减去它
    inVar = np.var(inMat,0)
    inMat = (inMat-inMeans)/inVar
    return inMat


'''
函数功能：前向逐步回归，不断地在新获得的权重上更新
参数：
     xArr：训练数据特征
     yArr：训练数据标签
     eps：更新步长
     numIt：循环次数
返回：
    ws：回归系数
'''
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yMean = np.mean(yMat,0) # 按列求均值
    yMat = yMat - yMean
    xMat = regularize(xMat) # 调用函数标准化数据 在岭回归中同样的处理但直接在函数中
    m,n = np.shape(xMat)
    ws = np.zeros((n,1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    
    
    for i in range(numIt): # 不断更新
        #print(ws.T)  #打印出来便于观察每次权重的变化
        lowerError = float('inf') # 初始化最大误差
        for j in range(n): # 循环每个特征
            for sign in [-1,1]: # 增大或减小
                wsTest = ws.copy()
                wsTest[j] += eps*sign # eps每次迭代的步长
                yTest = xMat*wsTest
                rssE = rssError(yMat.A,yTest.A)   # 预测值和真实值的误差
                if rssE < lowerError:
                    lowerError = rssE
                    wsMax = wsTest # 更新wsMax，ws、wsMax、wsTest三者之间相互copy来保证每次结果的保留和更改

        ws = wsMax.copy() # 当所有特征循环完了，找到错误最小的wsMax赋值给新的ws
    return ws


if __name__ =='__main__':
    abalone = pd.read_table('abalone.txt',header = None)
    abX,abY = get_Mat(abalone)
    
    # 岭回归lambda的回归系数和惩罚项的关系图
    ridgeWeights = ridgeTest(abX,abY)   # 返回 此处30次改变λ值后，得到的30行回归系数
    fig = plt.figure()              # 为了看到缩减（惩罚项）的效果而画图
    ax = fig.add_subplot(111)       # 及回归系数和惩罚项的关系
    ax.plot(ridgeWeights)       # 每列
    plt.show()
    

    # 前向逐步回归的回归系数
    xArr,yArr = get_Mat(abalone)
    w=stageWise(xArr,yArr,0.01,300) 

    #下面是用第1,24个lambda算出来的回归系数，去预测鲍鱼年龄
    #因为在使用岭回归时数据进行了标准化，所以在与预测应该也进行一些处理
    df1  = pd.read_table('abalone.txt',header = None)
    df1.columns=['性别','长度','直径','高度','整体重量','肉重量','内脏重量','壳重','年龄']
    df1['岭回归第1个λ预测']= regularize(abX)*np.mat(ridgeWeights[0,:]).T+ np.mean(abY)
    df1['岭回归第24个λ预测']= regularize(abX)*np.mat(ridgeWeights[23,:]).T+ np.mean(abY)
    

    #使用前向逐步回归返回的回归系数去预测鲍鱼年龄，同样数据需要做响应的处理
    df1['前向逐步回归预测']= regularize(abX)*w + np.mean(abY)

    print(df1)