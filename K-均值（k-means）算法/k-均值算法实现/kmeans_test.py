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