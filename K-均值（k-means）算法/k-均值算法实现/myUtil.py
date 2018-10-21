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



