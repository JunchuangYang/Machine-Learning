from myUtil import *

def kMeans(dataSet,k):
    m = shape(dataSet)[0]

    # ���㷨�������ݽṹ�����������ݼ���ͬ
    # ��1�����ݼ���Ӧ�ľ�������
    # ��2�����ݼ����������������ĵľ���
    ClustDist = mat(zeros((m,2)))

    # �������һ�����ݼ��ľ������ģ�����Ϊ4*2�ľ���
    # ȷ���þ�������λ��min(dataSet[:,j]),max(dataSet[:,j])֮��
    # ������ɾ�������
    clustercents = randCenters(dataSet,k)
    flag = True # ��ʼ����־λ��������ʼ
    counter = [] # ������

    # ѭ������ֱ����ֹ����Ϊfalse
    # �㷨ֹͣ��������dataSet���������������ҵ�ĳ���������ģ��������ĵľ����С������k-1�����ĵľ���

    while flag:
        # Ԥ����־λΪFalse
        flag = False
        # ---1.����ClusDist: ����DataSet���ݼ�������DataSetÿ����������ĵ���Сŷʽ����
        # ���˽����ֵClustDist=[minIndex,minDist]
        for i in range(m):

            #����k���������ģ������̾���
            distlist = [distEclud(clustercents[j,:],dataSet[i,:]) for j in range(k)]
            minDist = min(distlist)
            minIndex = distlist.index(minDist)

            #�ҵ���һ���µľ�������
            if ClustDist[i,0] != minIndex:
                flag = True #���ñ�־λ����������

            # ��minIndex��minDist����ClustDist��i��
            # ���壺���ݼ�i�ж�Ӧ�ľ�����ΪminIndex����̾���ΪminDist
            ClustDist[i,:] = minIndex,minDist


        #  ---2.���ִ�е��˴���˵��������Ҫ����clustercentsֵ��ѭ������Ϊcent��0��k-1��
        #  �þ�������cent�з�ClusDist������dataSet��������
        #  ���Դ˴�dataSet����ȡ��Ӧ�������������µ�ptsInClust
        #  ����ָ���ptsInClust���о�ֵ���Ӵ˸��¾�������clustercents�ĸ���ֵ
        for cent in range(k):
            # ��ClustDist�ĵ�һ����ɸѡ������centֵ���±�
            dInx = nonzero(ClustDist[:,0].A == cent)[0]
            # dataSet����ȡ���±�==dInx����һ���µ����ݼ�
            ptsInClust = dataSet[dInx]
             # ����ptsInClust���еľ�ֵ: mean(ptsInClust, axis=0):axis=0 ���м���
            clustercents[cent,:] = mean(ptsInClust,axis=0)

    return clustercents,ClustDist


