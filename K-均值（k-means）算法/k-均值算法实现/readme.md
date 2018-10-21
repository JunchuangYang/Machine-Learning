## k-means算法代码实现

本文来自[片片云飘过](https://www.cnblogs.com/eczhou/p/7860424.html),代码根据实际情况做了少许修改

Python版本：3.6.5

**算法原理**

KMeans算法是典型的基于距离的聚类算法，采用距离作为相似性的评价指标，即认为两个对象的距离越近，其相似度就越大。该算法认为簇是由距离靠近的对象组成的，因此把得到紧凑且独立的簇作为最终目标。

**K个初始聚类中心点的选取对聚类结果具有较大的影响**，因为在该算法第一步中是**随机地**选取任意k个对象作为初始聚类中心，初始地代表一个簇。该算法在每次迭代中对数据集中剩余的每个对象，根据其与各个簇中心的距离赋给最近的簇。当考查完所有数据对象后，一次迭代运算完成，新的聚类中心被计算出来。

算法过程如下：

（1）从N个数据文档（样本）随机选取K个数据文档作为质心（聚类中心）。

本文在聚类中心初始化实现过程中采取在样本空间范围内随机生成K个聚类中心。

（2）对每个数据文档测量其到每个质心的距离，并把它归到最近的质心的类。

（3）重新计算已经得到的各个类的质心。

（4）迭代（2）~（3步直至新的质心与原质心相等或小于指定阈值，算法结束。

本文采用所有样本所属的质心都不再变化时，算法收敛。

本文在实现过程中采用数据集4k2_far.txt，聚类算法实现过程中默认的类别数量为4。

**(1)辅助函数myUtil.py**

```新乡县
# -*- coding:utf-8 -*-
from mpmath import zeros
import numpy as np
from numpy import *

# 数据文件转矩阵
# path:数据文件路径
# delimiter：行内字段分隔符 （数据以空格分隔）
def file2matrix(path,delimiter):
    # 读取文件内容
    recordlist = []
    fp = open(path)
    for line in fp.readlines():
        curLine = line.strip().split(delimiter)
        fltLine=list(map(float,curLine))
        recordlist.append(fltLine)
    # 返回转换后的矩阵形式
    return mat(recordlist)

# 随机生成聚类中心
def randCenters(dataSet,k):
    #数据集总列数
    n = shape(dataSet)[1]
    # mat(zeros(k,n)):初始化聚类中心矩阵
    clustercents = mat(zeros((k,n)))
    for col in range(n):
        # dataSet[:,col]:第col列的全部值
        mincol = min(dataSet[:,col]) #最大值和最小值矩阵
        maxcol = max(dataSet[:,col])
        # random.rand(k,1) : 产生一个0~1之间的随机数向量（k,1表示产生k行1列 的随机数）
        clustercents[:,col]=mat(mincol + float(maxcol - mincol) * random.rand(k,1))
    return clustercents

# 计算欧式距离
def distEclud(vecA,vecB):
    return linalg.norm(vecA-vecB)

# 绘制散点图
def drawScatter(plt,mydata,size=20,color='blue',mrkr='o'):
    for i in range(len(mydata)):
        plt.scatter(mydata[i,0],mydata[i,1],s=size,c=color,marker=mrkr)

# 以不同颜色绘制数据集里的点
def color_cluster(dataindx,dataSet,plt):
    datalen = len(dataindx)
    for indx in range(datalen):
        if int(dataindx[indx]) == 0:
            plt.scatter(dataSet[indx,0],dataSet[indx,1],c='blue',marker='o')
        elif int(dataindx[indx]) == 1:
            plt.scatter(dataSet[indx,0],dataSet[indx,1],c='green',marker='o')
        elif int(dataindx[indx]) == 2:
            plt.scatter(dataSet[indx,0],dataSet[indx,1],c='red',marker='o')
        elif int(dataindx[indx]) == 3:
            plt.scatter(dataSet[indx,0],dataSet[indx,1],c='cyan',marker='o')




```

**(2)KMeans实现核心函数kmeans.py**

```python
from myUtil import *

def kMeans(dataSet,k):
    m = shape(dataSet)[0]

    # 本算法核心数据结构：行数与数据集相同
    # 列1：数据集对应的聚类中心
    # 列2：数据集行向量到聚类中心的距离
    ClustDist = mat(zeros((m,2)))

    # 随机生成一个数据集的聚类中心：本例为4*2的矩阵
    # 确保该聚类中心位于min(dataSet[:,j]),max(dataSet[:,j])之间
    # 随机生成聚类中心
    clustercents = randCenters(dataSet,k)
    flag = True # 初始化标志位，迭代开始
    counter = [] # 计数器

    # 循环迭代直至终止条件为false
    # 算法停止的条件：dataSet的所有向量都能找到某个聚类中心，到此中心的距离均小于其他k-1个中心的距离

    while flag:
        # 预警标志位为False
        flag = False
        # ---1.构建ClusDist: 遍历DataSet数据集，计算DataSet每行与聚类中心的最小欧式距离
        # 将此结果赋值ClustDist=[minIndex,minDist]
        for i in range(m):

            #遍历k个聚类中心，获得最短距离
            distlist = [distEclud(clustercents[j,:],dataSet[i,:]) for j in range(k)]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)

            #找到了一个新的聚类中心
            if ClustDist[i,0] != minIndex:
                flag = True #重置标志位，继续迭代

            # 将minIndex和minDist赋予ClustDist第i行
            # 含义：数据集i行对应的聚类中为minIndex，最短距离为minDist
            ClustDist[i,:] = minIndex,minDist


        #  ---2.如果执行到此处，说明还有需要更新clustercents值：循环变量为cent（0，k-1）
        #  用聚类中心cent切分ClusDist，返回dataSet的行索引
        #  并以此从dataSet中提取对应的行向量构成新的ptsInClust
        #  计算分隔后ptsInClust各列均值，从此更新聚类中心clustercents的各项值
        for cent in range(k):
            # 从ClustDist的第一列中筛选出等于cent值的下标
            dInx = nonzero(ClustDist[:,0].A == cent)[0]
            # dataSet中提取行下标==dInx构成一个新的数据集
            ptsInClust = dataSet[dInx]
             # 计算ptsInClust各列的均值: mean(ptsInClust, axis=0):axis=0 按列计算
            clustercents[cent,:] = mean(ptsInClust,axis=0)

    return clustercents,ClustDist



```
**(3)KMeans算法运行主函数kmeans_test.py**

```python
# -*- encoding:utf-8 -*-

from kmeans import *
import matplotlib.pyplot as plt


# 从文件构建的数据集
dataMat = file2matrix('training_4k2_far.txt',"\t")
# 提取数据集中的特征列
dataSet = dataMat[:,0:]

# 指定有k个聚类中心
k=4
Clustercents,ClustDist = kMeans(dataSet,k)

# 返回计算完成的聚类中心
print('Clustercents:\n',Clustercents)

# 输出生成的ClustDist：对应的聚类中心（列1），到聚类中心的距离（列2），行与dataset对应一一对应
color_cluster(ClustDist[:,0:1],dataSet,plt)
# 绘制聚类中心
drawScatter(plt,Clustercents,size=60,color='black',mrkr='D')

plt.show()

```

由于初始时是随机的选取k个聚簇中心，所以每次运行的结果可能不相同,KMeans并不是总能够找到正确的聚类

**正确的分类输出**

output1:

```
Clustercents:
 [[2.95832148 2.98598456]
 [3.02211698 6.00770189]
 [6.99438039 5.05456275]
 [8.08169456 7.97506735]]
```

![](https://i.imgur.com/KM51zV7.png)


**错误输出：局部最优收敛**

output2：

```
Clustercents:
 [[8.09885286 7.60593286]
 [8.0660961  8.31064416]
 [2.9750599  3.77881139]
 [6.99438039 5.05456275]]

```

![](https://i.imgur.com/gTNkZNI.png)


**错误输出:只收敛到三个聚类中心**

output3:

```
Clustercents:
 [[8.08169456 7.97506735]
 [       nan        nan]
 [6.99438039 5.05456275]
 [2.9750599  3.77881139]]
```

![](https://i.imgur.com/e7NwVfu.png)


