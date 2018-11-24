# encoding=utf-8
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def randCent(dataSet, k):
    n = shape(dataSet)[1]
    centroids = mat(zeros((k,n)))#create centroid mat
    for j in range(n):#create random cluster centers, within bounds of each dimension
        minJ = min(dataSet[:,j])
        rangeJ = float(max(dataSet[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ * random.rand(k,1))
    return centroids

def distEclud(vecA, vecB):
    return sqrt(sum(power(vecA - vecB, 2))) #la.norm(vecA-vecB)
# k-means 聚类算法
# 该算法会创建k个质心，然后将每个点分配到最近的质心，再重新计算质心。
# 这个过程重复数次，直到数据点的簇分配结果不再改变位置。
# 运行结果（多次运行结果可能会不一样，可以试试，原因为随机质心的影响，但总的结果是对的， 因为数据足够相似，也可能会陷入局部最小值）
def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = shape(dataSet)[0]    # 行数
    clusterAssment = mat(zeros((m, 2)))    # 创建一个与 dataSet 行数一样，但是有两列的矩阵，用来保存簇分配结果
    centroids = createCent(dataSet, k)    # 创建质心，随机k个质心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):    # 循环每一个数据点并分配到最近的质心中去
            minDist = inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])    # 计算数据点到质心的距离
                if distJI < minDist:    # 如果距离比 minDist（最小距离）还小，更新 minDist（最小距离）和最小质心的 index（索引）
                    minDist = distJI; minIndex = j
            if clusterAssment[i, 0] != minIndex:    # 簇分配结果改变
                clusterChanged = True    # 簇改变
                clusterAssment[i, :] = minIndex,minDist**2    # 更新簇分配结果为最小质心的 index（索引），minDist（最小距离）的平方
        print (centroids)
        for cent in range(k): # 更新质心
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A==cent)[0]] # 获取该簇中的所有点
            centroids[cent,:] = mean(ptsInClust, axis=0) # 将质心修改为簇中所有点的平均值，mean 就是求平均值的
    return centroids, clusterAssment

# 二分k--均值聚类算法
def biKmeans(dataSet, k, distMeas=distEclud):
    m = shape(dataSet)[0]
    clusterAssment = mat(zeros((m,2))) # 存储数据集中每个点的簇分配结果及平方误差
    centroid0 = mean(dataSet, axis=0).tolist()[0] # 计算整个数据集的质心：1*2的向量
    centList =[centroid0] # []的意思是使用一个列表保存所有的质心,簇列表,[]的作用很大
    for j in range(m):  # 遍历所有的数据点，计算到初始质心的误差值，存储在第1列
        clusterAssment[j,1] = distMeas(mat(centroid0), dataSet[j,:])**2
    while (len(centList) < k):  # 不断对簇进行划分，直到k
        lowestSSE = inf  # 初始化SSE为无穷大
        for i in range(len(centList)): # 遍历每一个簇
            #print 'i:',i               # 数组过滤得到所有的类别簇等于i的数据集
            ptsInCurrCluster = dataSet[nonzero(clusterAssment[:,0].A==i)[0],:]
            # 得到2个簇和每个簇的误差，centroidMat：簇矩阵  splitClustAss：[索引值,误差]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas) # centroidMat是矩阵
            sseSplit = sum(splitClustAss[:,1])  # 求二分k划分后所有数据点的误差和
                                             # 数组过滤得到整个数据点集的簇中不等于i的点集
            #print nonzero(clusterAssment[:,0].A!=i)[0]
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])# 所有剩余数据集的误差之和
            #print "sseSplit and notSplit: ",sseSplit,',',sseNotSplit
            if (sseSplit + sseNotSplit) < lowestSSE: # 划分后的误差和小于当前的误差，本次划分被保存
                #print 'here..........'
                bestCentToSplit = i  # i代表簇数
                bestNewCents = centroidMat  # 保存簇矩阵
                #print 'bestNewCents',bestNewCents
                bestClustAss = splitClustAss.copy() # 拷贝所有数据点的簇索引和误差
                lowestSSE = sseSplit + sseNotSplit  # 保存当前误差和
        # centList是原划分的簇向量，bestCentToSplit是i值
        #print 'len(centList) and  bestCentToSplit ',len(centList),',',bestCentToSplit
                  # 数组过滤得到的是新划分的簇类别是1的数据集的类别簇重新划为新的类别值为最大的类别数
        bestClustAss[nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList)
                  # 数组过滤得到的是新划分的簇类别是0的数据集的类别簇重新划为新的类别值为i
        bestClustAss[nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #print 'the bestCentToSplit is: ',bestCentToSplit   # 代表的是划分的簇个数-1
        #print 'the len of bestClustAss is: ', len(bestClustAss) # 数据簇的数据点个数
                                   # 新划分簇矩阵的第0簇向量新增到当前的簇列表中
        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]
        #print 'centList[bestCentToSplit]:',centList[bestCentToSplit]
                        # 新划分簇矩阵的第1簇向量添加到当前的簇列表中
        centList.append(bestNewCents[1,:].tolist()[0]) # centList是列表的格式
        #print 'centList',centList
                    # 数组过滤得到所有数据集中簇类别是新簇的数据点
        clusterAssment[nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss
    return mat(centList), clusterAssment # 返回质心列表和簇分配结果



# 球面距离计算，这里是利用球面余弦定理
def distSLC(vecA, vecB):  # 经度和纬度用角度作为单位，这里用角度除以180然后乘以pi作为余弦函数的输入
    a = sin(vecA[0,1]*pi/180) * sin(vecB[0,1]*pi/180)
    b = cos(vecA[0,1]*pi/180) * cos(vecB[0,1]*pi/180) * cos(pi * (vecB[0,0]-vecA[0,0]) /180)
    return arccos(a + b)*6371.0  # 返回地球表面两点之间的距离


#绘图函数
#numClust：希望得到的簇数
def clusterClubs(numClust=5):
    datList = [] # 创建一个空列表
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]),float(lineArr[3])]) # 对应的是纬度和经度

    datMat = mat(datList) # 创建一个矩阵
    myCentroids,clustAssing = biKmeans(datMat,numClust,distMeas=distSLC)

    fig = plt.figure() # 创建一幅图
    rect = [0.1,0.1,0.8,0.8] # 创建一个矩形来决定绘制图的哪一部分
    scatterMarkers=['s','o','^','8','p','d','v','h','>','<']  # 构建一个标记形状的列表来绘制散点图
    axprops = dict(xticks=[],yticks=[])
    ax0 = fig.add_axes(rect,label='ax0',**axprops) # 创建一个子图
    imgP = plt.imread('Portland.png') # imread()函数基于一幅图像来创建矩阵
    ax0.imshow(imgP) # imshow()绘制该矩阵
    ax1=fig.add_axes(rect, label='ax1', frameon=False) # 在同一张图上又创建一个子图
    for i in range(numClust): ## 遍历每一个簇
        ptsInCurrCluster = datMat[nonzero(clustAssing[:,0].A==i)[0],:]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:,0].flatten().A[0], ptsInCurrCluster[:,1].flatten().A[0], marker=markerStyle, s=90)
    ax1.scatter(myCentroids[:,0].flatten().A[0], myCentroids[:,1].flatten().A[0], marker='+', s=300)
    plt.show()

if __name__ =='__main__':
    clusterClubs(4)













