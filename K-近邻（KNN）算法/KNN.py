#原代码网址：https://www.cnblogs.com/ybjourney/p/4702562.html
#coding:utf-8
from numpy import *
import operator


#给出训练数据以及对应的类别
def createDataSet():
    group=array([[1.0,2.0],[1.2,0.1],[0.1,1.4],[0.3,3.5]])#训练数据
    lables=['A','A','B','B']#训练数据所对应的标签
    return group,lables

#通过KNN进行分类
#input：输入的待分类数据
#dataSet，已经训练的数据集
#label：数据分类的标签
#K
def classify(input,dataSet,label,k):
    dataSize=dataSet.shape[0] #dataset这个array的行数
    #numpy 创建的数组都有一个shape属性，它是一个元组，返回各个维度的维数。
    #当shape[0]返回矩阵行数，shape[1]返回矩阵列数
    ##计算欧式距离
    diff=tile(input,(dataSize,1))-dataSet #计算矩阵的差
    sqdiff=diff**2 #矩阵中每一个元素的平方
    squardist=sum(sqdiff,axis=1)
    dist=squardist**0.5
    #axis=1，表示求矩阵中每一行的和，形成一个新的矩阵
    #axis=0，表示求列的和

    #对距离进行排序
    sortedDistIndex=argsort(dist)#对矩阵中的值进行升序排列，返回元素下标


    classCount={}
    for i in range(k):
        #通过距离排序得到K个距离最小的点
        voteLabel=label[sortedDistIndex[i]]
        ##对选取的K个样本进行所属类别的个数进行统计
        classCount[voteLabel]=classCount.get(voteLabel,0)+1

    #选取出现类别次数最多的类别
    maxCount=0
    for key,value in classCount.items():
        if value > maxCount:
            maxCount=value
            classes=key

    return classes


#测试
dataSet,lables=createDataSet()
input=array([1.1,0.3])
K=3
output=classify(input,dataSet,lables,K)
print("测试数据为：",input,"分类结果为：",output)






















#
