# -*- encoding:utf-8 -*-
from numpy.ma import multiply

from kmeans import *
import matplotlib.pyplot as plt

# 从文件集构建数据
dataMat = file2matrix("training_4k2_far.txt","\t")
# 提取数据中的特征列
dataSet = dataMat[:,0:]

# 指定有k个聚类中心
k=4
# 获取数据集的行数
m = shape(dataSet)[0]

# 初始化第一个聚类中心：每一列的均值
centroid0 = mean(dataSet,axis=0).tolist()[0]
# 把均值聚类中心加入到中心表
centList = [centroid0]

# 初始化聚类距离表，距离方差
# 列1：数据集对应的聚类中心，列2：数据集行向量到聚类中心的距离
ClustDist = mat(zeros((m,2)))

for j in range(m):
    ClustDist[j,1] = distEclud(centroid0,dataSet[j,:])**2


# 初始情况
'''
color_cluster(ClustDist[:, 0:1], dataSet, plt)
drawScatter(plt, mat(centList), size=60, color='red', mrkr='D')
plt.show()
'''

# 依次生成k个聚类中心
while(len(centList) < k):
    # 初始化最小误差平方和，核心参数，这个值越小就说明聚类的效果越好
    lowestSSE = inf
    #遍历centList的每一个向量

    # ---1.使用ClustDist计算lowestSSE,以此确定：bestCentToSplit，bestNewCents，bestClusAss
    # 尝试划分每一簇
    for i in range(len(centList)):
        # 从dataSet中提取类别号为i的数据构成一个新数据集

        ptsInCurrClust = dataSet[nonzero(ClustDist[:,0].A == i)[0], :]

        # 应用标准kMeans算法(k=2)，将ptsInCurrClust划分出两个聚类中心，以及对应的聚类距离表
        # if len(ptsInCurrClust) == 0:
        #     continue
        centroidMat,splitClusAss = kMeans(ptsInCurrClust,2)

        # 计算splitClusAss的距离平方和
        sseSplit = sum(multiply(splitClusAss[:,1],splitClusAss[:,1]))
        # 此处求欧式距离的平方和
        # 计算ClustDist[ClustDist第一列！=i 的距离平方和]
        sseNoSplit = sum(ClustDist[nonzero(ClustDist[:,0].A != i)[0],1])

        # 算法公式: lowestSSE = sseSplit + sseNotSplit
        if (sseSplit + sseNoSplit) < lowestSSE:
            bestCentToSplit = i  #  确定聚类中心的最优分隔点
            bestNewCents = centroidMat # 用新的聚类中心更新最优聚类中心
            bestClustAss = splitClusAss.copy() # 深拷贝聚类距离表为最优聚类距离表
            lowestSSE = sseSplit + sseNoSplit # 更新lowserSSE

    # 外循环
    # ---2. 计算新的ClustDist---
    # 计算bestClustAss 分了两部分
    # 第一部分为bestClusAss[bIndex0,0]赋值为聚类中心的索引
    bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)

    # 第二部分 用最优分隔点的指定聚类中心索引
    bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit
    # 以上计算bestClusAss

    # ---3.用最优分隔点来重构聚类中心
    # 覆盖：bestNewCents[0,:].tolist()[0]附加到原有聚类中心的bestCentToSplit位置
    # 增加：聚类中心增加一个新的bestNewCents[1,:].tolist()[0]向量
    centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
    centList.append(bestNewCents[1,:].tolist()[0])
    # 以上为计算centList
    # 将bestCentToSplit所对应的类重新更新类别
    ClustDist[nonzero(ClustDist[:, 0].A==bestCentToSplit)[0],:] = bestClustAss

    # 进行每一次迭代后
    '''
    color_cluster(ClustDist[:, 0:1], dataSet, plt)
    drawScatter(plt, mat(centList), size=60, color='red', mrkr='D')
    plt.show()
    '''


# 输出生成的ClustDist：对应的聚类中心(列1)，到聚类中心的距离(列2)，行与dataSet一一对应
color_cluster(ClustDist[:,0:1],dataSet,plt)
print ("cenList:\n",mat(centList))
# 绘制聚类中心图形
drawScatter(plt, mat(centList), size=60, color='black', mrkr='D')
plt.show()



