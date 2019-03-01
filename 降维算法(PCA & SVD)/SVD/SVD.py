from numpy import *
from numpy import linalg as la
'''
函数功能：计算欧式距离
参数说明：
    inA和inB都是列向量
'''
def ecludSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

'''
函数功能：计算皮尔斯相关系数
'''
def pearsSim(inA , inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA , inB , rowvar = 0)[0][1]

'''
函数功能：计算余弦相似度
'''
def cosSim(inA , inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denom)

'''
函数功能：用来计算在给定相似度计算方法的条件下，用户对物品的估计评分值
参数说明：
    dataMat：数据矩阵
    user：用户编号
    simMeas：相似度计算方法
    item：物品编号
返回：预测评分值
'''
def standEst(dataMat , user , simMeas , item):
    # 数据集中物品的数目
    n = shape(dataMat)[1]
    # 对两个用于计算估计评分值的变量进行初始化
    simTotal = 0.0
    ratSimTotal = 0.0
    # 遍历行中的每个物品
    for j in range(n):
        userRating = dataMat[user,j]
        # 评分值为0，意味着用户没有对该物品评分
        if userRating == 0:
            continue
        #寻找两个用户都评级过的商品，变量overlap给出的是两个物品当中已经评分的那个元素
        overLap = nonzero(logical_and(dataMat[:,item].A > 0 , dataMat[:,j].A) > 0)[0]
        # 若两者没有任何重合元素，则相似度为0且终止本次循环0
        if len(overLap) == 0:
            similarity = 0
        #如果存在重合的物品，则基于这些重合物品计算相似度
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        # 随后相似度不断累加
        simTotal += similarity
        ratSimTotal += similarity * userRating
        
    if simTotal ==0:
        return 0
    else:
        # 通过除以所有的评分总和，对上述相似度评分的乘积进行归一化。这使得评分值在0-5之间，
        # 而这些评分值则用于对预测值进行排序
        return ratSimTotal / simTotal
        
        
 '''
函数功能：推荐引擎，产生相似度最高的N个结果
参数说明：
    dataMat：数据集（用户*物品）
    uset：给定的用户
    N：几个推荐结果
    simMeas: 相似度计算
    estMethod：估计方法
返回：
    相似度最高的N个结果
'''
def recommend(dataMat , user , N=3, simMeaas = cosSim , estMethod = standEst):
    # 寻找未评级的物品，对给定用户建立一个未评级的物品列表
    unratedItems = nonzero(dataMat[user,:]== 0)[1]
    # 如果不存在未评级的物品，退出函数，否则在所有未评分物品上进行循环
    if len(unratedItems) == 0:
        return  'you rated everything'
    itemScores = []
    for item in unratedItems:
        # 对于每个未评分的物品，通过调用standEst（）来产生该物品的预测评分
        estimatedScore = estMethod(dataMat , user , simMeaas , item)
        # 该物品的编号和估计得分值会放在一个元素列表itemScores中
        itemScores.append((item,estimatedScore))
        
    return sorted(itemScores,key = lambda j:j[1],reverse= True)[:N]

def loadExData() :
    return [[1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [5, 5, 5, 0, 0],
            [1, 1, 0, 2, 2],
            [0, 0, 0, 3, 3],
            [0, 0, 0, 1, 1]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

'''
函数功能：基于SVD的评分估计
参数说明：
    dataMat：数据矩阵
    user：用户编号
    simMeas:相似度计算
    item：物品编号
    

在recommend()中，svdEst用户替换对standEst()的调用，该函数对给定用户物品构建一个评分估计值。
与standEst()非常相似，不同之处就在于它在第3行对数据集进行了SVD分解。在SVD分解后，只利用包含
90%能量值的奇异值，这些奇异值以Numpy数组的形式得以保存。
'''

def svdEst(dataMat, user, simMeas, item) :
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat)
    # 使用奇异值构建一个对角矩阵
    Sig4 = mat(eye(4)*Sigma[:4])
    # 利用U矩阵将物品转换到低维空间中
    # 转置后变成 物品*用户
    xformedItems = dataMat.T * U[:, :4] * Sig4.I
    # 对于给定的用户，for循环在用户对应行的所有元素上进行遍历，与standEst()函数中的for循环目的一样
    # 不同的是，这里的相似度是在低维空间下进行的。相似度的计算方法也会作为一个参数传递给该函数
    for j in range(n) :
        userRating = dataMat[user,j]
        if userRating == 0 or j == item : continue
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)# 计算相似度时在转置回来
        # print便于了解相似度计算的进展情况
        print ('the %d and %d similarity is : %f' % (item, j, similarity))
        # 对相似度求和
        simTotal += similarity
        # 对相似度及评分值的乘积求和
        ratSimTotal += similarity * userRating
    if simTotal == 0 : return 0
    else : return ratSimTotal/simTotal

'''
函数功能： 图像压缩函数，用于打印矩阵
由于矩阵含有浮点数，因此必须定义浅色和深色。这里通过一个阈值来界定。
该函数遍历所有的矩阵元素，当元素大于阈值时打印1，否则打印0
参数说明：
    intMat：数据矩阵
    thresh：阈值
'''
def printMat(intMat , thresh = 0.8):
    for i in range(32):
        for k in range(32):
            if float(intMat[i,k]) > thresh:
                print ('1')
            else:
                print ('0')
            print (' ')

'''
函数功能：实现了图像的压缩。它允许基于任意给定的奇异值数目来重构图像
参数说明：
    numSV：奇异值数目
    filename:文件路径
    thresh：阈值
'''
def imgCompress(numSV = 3,filename,thresh = 0.8):
    myl = [] # 构建一个列表myl
    # 打开文本文件，以数值方式读入字符
    for line in open(filename).readlines():
        newRow = [] 
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = mat(myl)
    # 输入矩阵
    print ('********original matrix*******')
    printMat(myMat,thresh)
    # 对原始图像进行SVD分解并重构图像，通过将Sigma重构成SigRecon来实现
    U , Sigma , VT = la.svd(myMat)
    # Sigma是一个对角矩阵，需要建立一个全0矩阵，然后将前面的那些奇异值填充到对角线上。
    SigRecon = mat(zeros((numSV,numSV)))
    
    for k in range(numSV):
        SigRecon[k,k] = Sigma[k]
        
    # 通过截断的U和VT矩阵，用SigRecon得到重构后的矩阵
    reconMat = U[:,:numSV]*SigRecon*VT[:numSV,:]
    print ("******reconstructed matrix using %d singular values******" % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':

    # 计算该矩阵的SVD来了解其到底需要多少维特征。
    U , Sigma , VT = la.svd(mat(loadExData2()))
    # 对Sigma中的值求平方
    Sig2 = Sigma**2
    #接着看看到底多少个奇异值能达到总能量的90%。
    print("总能量：",sum(Sig2))
    print("总能量的90%：",sum(Sig2)*0.9)
    print("前两个元素的能量：",sum(Sig2[:2]))
    print("前三个元素的能量：",sum(Sig2[:3]))

    # 利用SVD提高推荐的效果
    myMat=mat(loadExData2())
    recommend(myMat, 1, estMethod=svdRec.svdEst)

    # 基于SVD的图像压缩
    imgCompress(2,'xxx.txt')