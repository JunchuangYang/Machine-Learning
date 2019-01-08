# 参考 : https://cuijiahua.com/blog/2017/12/ml_12_regression_2.html


# -*-coding:utf-8 -*-
import numpy as np
from bs4 import BeautifulSoup
import random

'''
函数功能：从页面读取数据，生成retX和retY列表
参数说明：
    retX：数据特征
    retY: 数据标签
    inFile：HTML文件
    yr：年份
    numPce：乐高部件数目
    origPrc：原价
返回：
    无
'''
def scrapePage(retX,retY,inFile,yr,numPce,origPrc):
    # 打开并读取HTML界面
    with open(inFile,encoding = 'utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    
    i=1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table',r="%d"%i)
    while(len(currentRow)!=0):
        currentRow = soup.find_all('table',r="%d"%i)
        title = currentRow[0].find_all('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            #print("商品 #%d 没有出售" % i)
            pass
        else:
            
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) >1:
                priceStr = priceStr.replace('Free shipping','')
            sellingPrice = float(priceStr)
            
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                #print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr,numPce,newFlag,origPrc])
                retY.append(sellingPrice)
        i += 1
        
        # 找下一个套装信息
        currentRow = soup.find_all('table',r = '%d' %i)

def setDataCollect(retX,retY):
    scrapePage(retX,retY,'setHTML/lego8288.html',2006,800,49.99)
    scrapePage(retX,retY,'setHTML/lego10030.html',2002,3096,269.99)    
    scrapePage(retX,retY,'setHTML/lego10179.html',2007,5195,499.99)    
    scrapePage(retX,retY,'setHTML/lego10181.html',2007,3428,199.99)    
    scrapePage(retX,retY,'setHTML/lego10189.html',2008,5922,199.99)    
    scrapePage(retX,retY,'setHTML/lego10196.html',2009,3263,249.99)    

'''
函数说明：数据标准化
参数说明：
    xMat：数据集的特征
    yMat：数据集的标签
    
返回：
    标准化后的数据集
'''

def regularize(xMat,yMat):
    inxMat = xMat.copy() #数据拷贝
    inyMat = yMat.copy()
    yMean = np.Mean(yMat,0) # 压缩行，对各列求均值，返回 1* n 矩阵
    inyMat = yMat - yMean  # 据减去均值
    inMeans = np.mean(inxMat,0)
    inVar = np.Var(inxMat,0) # 求方差
    inxMat = (inxMat - inMeans) / inVar #数据减去均值除以方差实现标准化
    
    return inxMat,inyMat
'''
函数说明：计算平方误差和
参数说明：
    yArr：预测值
    yHatArr：真实值
返回：
    预测值与真实值的误差平方和
'''
def rssError(yArr,yHatArr):
    return ((yArr - yHatArr)**2).sum()

'''
函数说明：使用线性回归方法求回归系数
参数说明：
    xArr：数据集特征
    yArr：数据集标签
返回：
    ws：特征的回归系数
'''
def standRegress(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat   #根据推导的公示计算回归系数
    if np.linalg.det(xTx) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = xTx.I * (xMat.T*yMat)
    return ws


'''
函数说明：使用简单的线性回归
参数说明：
    无
'''
def useStandRegress():
    lgX = []
    lgY = []
    setDataCollect(lgX,lgY)
    
    data_num , features_num = np.shape(lgX)
    
    lgX1 = np.mat(np.ones((data_num,features_num + 1)))
    
    lgX1[:,1:5] = np.mat(lgX)
    ws = standRegress(lgX1,lgY)
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0],ws[1],ws[2],ws[3],ws[4]))    

'''
函数功能：岭回归求回归系数
参数说明：
    xMat：数据集特征
    yMat：数据集标签
    lam：缩减系数
返回：
    ws：回归系数
'''
def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(np.shape(xMat)[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("矩阵为奇异矩阵,不能转置")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws    

'''
函数功能：岭回归预测
参数说明：
    xMat：数据集特征
    yMat：数据集标签
    lam：缩减系数
返回：
    ws：回归系数矩阵
'''
def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    #数据标准化
    yMean = np.mean(yMat, axis = 0)                        #求均值
    yMat = yMat - yMean                                    #数据减去均值
    xMeans = np.mean(xMat, axis = 0)                    #行与行操作，求均值
    xVar = np.var(xMat, axis = 0)                        #行与行操作，求方差
    xMat = (xMat - xMeans) / xVar                        #数据减去均值除以方差实现标准化
    numTestPts = 30                                        #30个不同的lambda测试
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))    #初始回归系数矩阵
    for i in range(numTestPts):                            #改变lambda计算回归系数
        ws = ridgeRegres(xMat, yMat, np.exp(i - 10))    #lambda以e的指数变化，最初是一个非常小的数，
        wMat[i, :] = ws.T                                 #计算回归系数矩阵
    return wMat


 """
 函数说明:交叉验证岭回归
参数说明：
    xMat：数据集特征
    yMat：数据集标签
    numVal： 交叉验证次数
 """
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)                                                                        #统计样本个数                       
    indexList = list(range(m))                                                            #生成索引值列表
    errorMat = np.zeros((numVal,30))                                                    #create error mat 30columns numVal rows
    for i in range(numVal):                                                                #交叉验证numVal次
        trainX = []; trainY = []                                                        #训练集
        testX = []; testY = []                                                            #测试集
        
        random.shuffle(indexList)                                                        #打乱次序
        
        for j in range(m):                                                                #划分数据集:90%训练集，10%测试集
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
                
        wMat = ridgeTest(trainX, trainY)                                                #获得30个不同lambda下的岭回归系数
        for k in range(30):                                                                #遍历所有的岭回归系数
            matTestX = np.mat(testX); matTrainX = np.mat(trainX)                        #测试集
            meanTrain = np.mean(matTrainX,0)                                            #测试集均值
            varTrain = np.var(matTrainX,0)                                                #测试集方差
            matTestX = (matTestX - meanTrain) / varTrain                                 #测试集标准化
            
            yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)                        #根据ws预测y值
            errorMat[i, k] = rssError(yEst.T.A, np.array(testY))                            #统计误差
            
    meanErrors = np.mean(errorMat,0)                                                    #计算每次交叉验证的平均误差
    minMean = float(min(meanErrors))                                                    #找到最小误差
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]                                #找到最佳回归系数
    
    xMat = np.mat(xArr); yMat = np.mat(yArr).T
    meanX = np.mean(xMat,0); varX = np.var(xMat,0)
    unReg = bestWeights / varX                                                            #数据经过标准化，因最佳回归系数需要还原
    print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))    
 

 if __name__ == '__main__':
    lgX = []
    lgY = []
    # 简单线性回归
    useStandRegress()

    # 岭回归
    setDataCollect(lgX, lgY)
    print("缩减过程中回归系数是如何变化的\n",ridgeTest(lgX, lgY))
    crossValidation(lgX, lgY)