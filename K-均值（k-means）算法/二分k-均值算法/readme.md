## Bisecting KMeans (二分K均值）

参考[片片云飘过](https://www.cnblogs.com/eczhou/p/7860435.html)

### 算法原理

由于传统的KMeans算法的聚类结果易受到初始聚类中心点选择的影响，因此在传统的KMeans算法的基础上进行算法改进，对初始中心点选取比较严格，各中心点的距离较远，这就避免了初始聚类中心会选到一个类上，一定程度上克服了算法陷入局部最优状态。

二分KMeans(Bisecting KMeans)算法的**主要思想**是：首先将所有点作为一个簇，然后将该簇一分为二。之后选择能最大限度降低聚类代价函数（也就是误差平方和）的簇划分为两个簇。以此进行下去，直到簇的数目等于用户给定的数目k为止。

以上隐含的一个原则就是：因为聚类的误差平方和能够衡量聚类性能，该值越小表示数据点越接近于他们的质心，聚类效果就越好。

所以我们就需要对误差平方和最大的簇进行再一次划分，因为误差平方和越大，表示该簇聚类效果越不好，越有可能是多个簇被当成了一个簇，所以我们首先需要对这个簇进行划分。

**二分K-means聚类的优点**：

1. 二分K均值算法可以加速K-means算法的执行速度，因为它的相似度计算少了
 
2. 不受初始化问题的影响，因为这里不存在随机点的选取，且每一步都保证了误差最小

### 代码

聚类算法实现过程中默认的类别数量为4。其中辅助函数存于myUtil.py文件和K均值核心函数存于kmeans.py文件。([参见](https://github.com/JunchuangYang/Machine-Learning/tree/master/K-%E5%9D%87%E5%80%BC%EF%BC%88k-means%EF%BC%89%E7%AE%97%E6%B3%95/k-%E5%9D%87%E5%80%BC%E7%AE%97%E6%B3%95%E5%AE%9E%E7%8E%B0))

```python
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



```

### 知识点

**1.mean()**

```
mean()函数功能：求取均值

经常操作的参数为axis，以m * n矩阵举例：

axis 不设置值，对 m*n 个数求均值，返回一个实数

axis = 0：压缩行，对各列求均值，返回 1* n 矩阵

axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵

```

**2.numpy中矩阵名.A的含义**

python中一个matrix矩阵名.A 代表将 矩阵转化为array数组类型

![](https://i.imgur.com/3MDh0dm.png)

上图中，a被定义为矩阵，但a.A赋值给b之后，b的数据类型转化为array。

[来源](https://blog.csdn.net/andyleo0111/article/details/78918285)