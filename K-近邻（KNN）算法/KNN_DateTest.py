#代码来自 icamera0 的CSDN 博客 ，https://blog.csdn.net/icamera0/article/details/77973150?utm_source=copy

# encoding:utf-8
from numpy import *
import operator
# inX: 1行M列的向量，为测试样本，用来跟输入变量dataSet中的数据进行比较(1xM)
# dataSet: 已知的N行M列的数组，为训练样本集，一行为一个训练样本，即N个训练样本，每个训练样本M个元素(NxM)
# labels: 1行N列的向量，存储各个训练样本对应分类的标签信息 (1xN)
# k: 用来比较的近邻元素的个数，选择奇数，且小于20
def knnClassify(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0] #获取训练样本集dataSet的个数，即矩阵dataSet的行数N
	#tile(inX, (dataSetSize, 1))是建立一个N行1列的数组，但其元素是一个1行M列的向量，最后返回一个N行M列的数组
	#N行M列的数组中每一行都是输入的测试样本，它与训练样本集相减，则得到NxM的数组值之差
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis = 1) #axis=1是对存储NxM个元素对应的平方差的数组，将每一行的值累加起来，返回一个Nx1的数组
	distances = sqDistances**0.5 #求得测试样本与各个训练样本的欧式距离

	#对distances中N个元素进行从小到大排序，之后返回一个N个元素的一维数组，存储distances排序后各个元素在原来数组中的index
	#eg. distances=[2,1,3,0], argsort的返回值为[3，1，0，2]
	sortedDistIndicies = distances.argsort()

	classCount={} #定义一个空的字典变量
	for i in range(k): #i的取值为[0，k - 1]
		voteLabel = labels[sortedDistIndicies[i]] #返回测试样本与训练样本欧式距离第i小的训练样本所对应的类的标签
		#classCount.get(voteLabel, 0)获取classCount中voteLabel为index的元素的值，找不到则返回0
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

	#classCount.iteritems()返回字典classCount中所有项，按照迭代器的方式
    #python3.5中，iteritems（）变为items()
	#operator.itemgetter()用于获取对象的哪些维的数据，参数为一些序号；此处参数为1，即按照字典变量classCount各项的第二个元素进行排序
	#reverse = True表示按照逆序排序，即从大到小的次序；False表示按照从小到大的次序排序
	sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True)

	#返回k个近邻元素中所对应类最多那个类的标签，即测试样本所属那个类
	return sortedClassCount[0][0]


#从训练样本文件中读取样本信息，并将匪类信息与其他数据分别存储在不同数组中
def file2matrix(filename):
    fd=open(filename)
    #获取文件中样本数目
    numberOfLines=len(fd.readlines())
    #创建numberOfLines*3的二维数组，并初始化为0，存储文件中读取样本数据的前三列
    returnMat=zeros((numberOfLines,3))
    #定义存储样本类别的列表
    classLabelVector=[]
    #重新打开文件，因为前面为了获取样本中读取了所有样本，文件指针不在文件最前面
    fd=open(filename)
    index=0
    for line in fd.readlines():#从文件中循环读取每一行数据进行处理
        line=line.strip() #去掉这一行前面和后面的空格
        listFromLine=line.split('\t') #将数据按空格分割，本实例中将所读数据分为4个独立数据
        returnMat[index, :]=listFromLine[0:3] #将前3个数据存储到returnMat数组中
        #将最后一个数据，也就是第四个数据，转换成int类型后存储在classLabelVector中
        classLabelVector.append(int(listFromLine[-1]))
        index+=1
    #returnMat存储样本的前三列数据，classLabelVector存储样本对应的分类信息
    return returnMat,classLabelVector

#对输入数组进行归一化数值处理，也叫特征函数，用于将特征缩放到同一个范围内
#缩放公式为：newValue=（oldValue-min）/（max-min）
def auotoNorm(dataSet):#输入数组dataSet为N*3的二维数组
    minVals=dataSet.min(0)#获取数组中每一列的最小值，minVals为1*3的数组，min（1）获取每一行的最小值
    maxVals=dataSet.max(0)#。。。。。。。。。。大。。maxVals。。。。。。max（1）获取每一行的最大值
    range=maxVals-minVals #获取特征范围差，ranges也是1*3的数组
    normDataSet=zeros(shape(dataSet))#创建与dataSet的维数，类型完全一样的数组，初始化为0，用于存储归一化后的结果
    m=dataSet.shape[0]#获取输入数组的行数
    #tile(minVals,(m,1))创建m*1的数组，数组元素为1*3的最小值数组，其返回值为m*3的数组
    #从原始数组的各个元素中，减去对应列的最小值
    normDataSet=dataSet-tile(minVals,(m,1))
    #tile(range,(m,1))创建m*1的数组，数组元素为1*3的特征范围差，其返回值为m*3的数组
    #原始数组的各个元素除以对应列的特征范围差，完成归一化
    normDataSet=normDataSet/tile(range,(m,1))
    #返回归一化后的N*3的数组，1*3的特征范围差和1*3的每列最小值
    return normDataSet,range,minVals


#根据训练样本datingTestSet2.txt中数据对knn算法进行测试
def datingClassTest():
    hoRatio=0.50#用于分割样本，将文件中获取的样本前面一半作为测试样例，后面一半作为训练样例
    #从样本文件datingTestSet2.txt中读取所有样例数据及分类信息
    datingDatamat,datingLabels=file2matrix('datingTestSet2.txt')
    #将样本数据进行归一化处理
    normMat,ranges,minVals=auotoNorm(datingDatamat)
    m=normMat.shape[0]#获取归一化后二维数组的行数，即所有样本的数目
    numTestVecs=int(m*hoRatio)#获取测试样本的数目，为所有样本的一半
    errorCount=0.0 #记录分类错误的次数
    for i in range(numTestVecs):#依次循环，从前一半样本中获得每一个样本，跟后面一半样本进行比对，寻找最近邻样本
        classifierResult=knnClassify(normMat[i, :],normMat[numTestVecs : m, :], datingLabels[numTestVecs : m], 3)
        if(classifierResult!=datingLabels[i]):#如果分类结果与从文件中读取的值不一致，则判为分类错误
            errorCount+=1.0
            print ("the classifier case back with: %d, the real answer is: %d, index: %d" % (classifierResult, datingLabels[i], i))
    print ("the total error rate is: %f" % (errorCount / float(numTestVecs))) #打印出错误率
    print (errorCount) #返回错误次数


#测试
datingClassTest()














#
