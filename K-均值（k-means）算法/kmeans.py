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


