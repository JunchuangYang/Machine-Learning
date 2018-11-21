## 使用k-近邻算法识别手写数字

为了简单起见，这里构造的系统只能识别0到9，需要识别的数字已经使用图形处理软件，处理成具有相同的色彩和大小：宽高是32像素*32像素的黑白图像。尽管采用文本格式存储的图像不能有效地利用内从空间，但是为了方便，我们还是将图像转换为文本格式。

**classify0()**: 用于计算待测数据与训练数据的欧式距离

**img2Vector()**: 将图像（32*32的二进制模拟图像）转化为测试向量（1*1024的向量）

**handwritingClassTest()** : 测试代码

```python

def handwritingClassTest():
    hwLables = []
    # 获取目录trainingDigits目录下的所有文件
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        # 下面三行获取训练文件的分类标签
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s'%fileNameStr)
    testFileList = listdir('testDigits')
    errcount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        # 同上
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s'% fileNameStr)
        classifierResult = classify0(vectorUnderTest,trainingMat,hwLables,3)
```

handwritingClassTest()：将trainingDigits目录中的文件内容存储在列表中，然后可以得到目录中有多少文件，将其存储在变量m中。接着，代码创建一个m行1024列的训练矩阵，该矩阵的每行数据存储一个图像。我们可以从文件名中解析分析分类文字。该目录下的文件按照规则命名，如文件9_45.txt的分类是9，他是数字9的第45个实例。然后我们可以将类代码存储在hwLables向量中，使用前面讨论的img2Vector函数载入图像。在下一步中我们对testDigits目录中的文件执行相似的操作，不同之处是我们并不将这个目录下的文件载入矩阵中，而是使用classifiy0（）函数测试该目录下的文件


实际使用这个算法时，算法的执行效率并不高。因为算法需要为每个测试向量做2000次距离计算，每个距离计算包括了1024个维度浮点运算，总计要执行900次，此外，我们还需要为测试向量准备2MB的存储空间。




