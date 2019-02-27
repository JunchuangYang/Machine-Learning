#-*- coding:utf-8 -*
import numpy as np
import pandas as pd
'''
简单测试数据集
'''
def loadDataSet():
    dataSet = [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
    return dataSet

'''
函数功能：生成第一个候选集合C1
参数说明：
    dataSet：原始数据集
返回：
    frozenset形式的候选集合C1
'''
def createC1(dataSet):
    C1 = []
    for transaction  in dataSet:
        for item in transaction:
            if not {item} in C1:
                C1.append({item})
    C1.sort()
    return list(map(frozenset,C1))

'''
函数功能：生成满足最小支持度的频繁项集L1
参数说明：
    D：原始数据集
    ck：候选项集
    minSupport：最小支持度
返回：
    retList：频繁项集
    supportData：候选项集的支持度
'''
def scanD(D,ck,minSupport):
    ssCnt = {}
    for tid in D:
        for can in ck:
            if can.issubset(tid): # 判断can是否是tid的子集，返回bool类型
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {} # 候选项集Ck的支持度字典（key：候选项，value：支持度）
    for key in ssCnt:
        support = ssCnt[key] / numItems
        supportData[key] = support
        if support >= minSupport:
            retList.append(key)
    return retList,supportData


'''
函数功能：创建候选项集Ck
参数说明：
    Lk：频繁项集列表Lk
    k：项集元素个数
返回：
    Ck：候选项集(无重复)
'''
def aprioriGen(Lk,k):
    Ck = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            # 前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L1.sort()
            L2 = list(Lk[j])[:k-2]
            L2.sort()
            if L1 == L2:
                Ck.append(Lk[i] | Lk[j])
    return Ck

'''
参数说明：
    D：原始数据集
    minSupport：最小支持度
返回：
    L：所有项集
    supportData：项集：支持度
'''
def apriori(D,minSupport = 0.5):
    C1 = createC1(D)
    L1 , supportData = scanD(D,C1,minSupport)
    L = [L1]
    k = 2
    while(len(L[k-2]) > 0):
        ck = aprioriGen(L[k-2],k)
        Lk , supK = scanD(D,ck,minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L,supportData

'''
函数功能：计算规则的可信度以及找到满足最小可信度要求的规则
'''
def calcConf(freqSet,H,supportData,br1,minConf=0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf > minConf:
            print (freqSet - conseq , '--->',conseq , 'conf:'  , conf)
            br1.append((freqSet - conseq,conseq,conf))
            prunedH.append(conseq)
    return prunedH

'''
函数功能：从最初的项集中生成更多的规则
'''
def rulesFromConseq(freqSet,H,supportData,br1,minConf = 0.7):
    m = len(H[0])
    if (len(freqSet) > (m+1)):
        Hmp1 = aprioriGen(H , m+1)
        Hmp1 = calcConf(freqSet,Hmp1,supportData,br1,minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet,Hmp1,supportData,br1,minConf )

'''
参数说明：
    L:频繁项集列表
    SupportData：频繁项集支持数据的字典
    minConf：最小可信度阈值
返回：
    bigRuleList：包含可信度的规则列表
'''
def generateRules(L,supportData,minConf = 0.7):
    bigRuleList = []
    for i in range(1,len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1:
                rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf )
            else :
                calcConf(freqSet,H1,supportData,bigRuleList,minConf)
    return bigRuleList

if __name__ == '__main__':
    # 毒蘑菇数据测试(只用了毒蘑菇数据中的前100条数据)
    mushDatSet = [line.split() for line in open('mushroom.dat').readlines()]
    M1,MsuppData = apriori(mushDatSet,minSupport = 0.3)
    #与毒蘑菇频繁出现的一项特征
    for item in M1[1]:
        if item.intersection('2'):
            print (item)
    #与毒蘑菇频繁出现的二项特征
    for item in M1[2]:
        if item.intersection('2'):
            print (item)