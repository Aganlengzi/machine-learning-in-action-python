# -*- coding:utf-8 -*-

from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

#从文件读取特定格式的数据
def file2matrix(filename):
	fr = open(filename)						#打开文件
	arrayOfLines = fr.readlines()			#按行读取文件
	numberOfLines = len(arrayOfLines)		#计算行数
	retrunMat = zeros((numberOfLines,3))	#numberOfLines * 3的零矩阵
	classLabelVector = []					#list 存放的是分类标签
	index = 0
	for line in arrayOfLines:				#对每一行
		line = line.strip()					#删除头尾字符，可以用参数指定，默认是空格换行等
		listFormLine = line.split('\t')		#返回一个list
		retrunMat[index,:] = listFormLine[0:3]	#第index行数据
		classLabelVector.append(int(listFormLine[-1]))	#取出这一行的最后一列作为分类标签
		index += 1
	return retrunMat, classLabelVector



#某一列数值的归一化
def autoNorm(dataSet):
	minVals = dataSet.min(0)						#每列数据的最小值 min如果不带参数 表示取某行中的最小值
	maxVals = dataSet.max(0)						#每列数据的最大值
	ranges = maxVals - minVals						#每列范围
	normDataSet = zeros(shape(dataSet))				#理解为与dataset相同尺寸的零矩阵
	m = dataSet.shape[0]							#行数
	normDataSet = dataSet - tile(minVals, (m,1))	#两个列矩阵相减
	normDataSet = normDataSet/tile(ranges, (m,1))	#两个列矩阵相同位置元素相除
	return normDataSet, ranges, minVals				#返回归一化结果



#KNN分类器
def classify0(intX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]					#numpy.core.fromnumeric中的函数，矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度,可以理解为行数 shape[1]理解为列数
	diffMat = tile(intX, (dataSetSize,1)) - dataSet	#欧式距离中的减法操作 tile是按照后面的参数对第一个参数进行扩展，(X,(a,b))生成a*b*X的array
	sqDiffMat = diffMat**2							#平方操作
	sqDistances = sqDiffMat.sum(axis=1)				#相加操作,如果不写就是普通相加操作，如果axis=1就是按照行向量相加
	distances = sqDistances**0.5					#开方操作（这个有必要吗？反正是比较大小，开不开方好像没所谓）
	sortedDistIndicies = distances.argsort()		#排序操作 返回元素排序的索引值，理解为原数组中的下标值
	classCount = {}									#字典
	
	for i in range(k):								#前k个距离，根据上文这里得到的是排序下标的数组，下标与原label中的类对应
		voteIlabel = labels[sortedDistIndicies[i]]	#第i个的标签
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 	#第i个的标签的统计加1 dict的get方法的第一个参数是key第二个参数是key不存在时返回的参数
	#此sorted函数返回副本，注意此处返回的是dict
	#原输入不变，第一个参数是迭代器，第二个参数是获取矩阵(数组？)维数(从0开始数的第几列)
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse = True)
	return sortedClassCount[0][0]					#得到最前面的一个元素值的key而不是value



#测试程序 测试程序的可用行 根据检验集的错误率进行判断和分析
def datingClassTest():
	hoRatio = 0.10			#设置测试集大小
	datingDataMat, datingLabels = file2matrix("data.txt")	#从文件中读取数据
	normDataSet, ranges, minVals = autoNorm(datingDataMat)	#进行归一化处理
	m = normDataSet.shape[0]		#行数
	numTestVecs = int(m*hoRatio)	#测试数据行数
	errorCount = 0.0				#测试中的出错个数
	for i in range(numTestVecs):	#简单起见选取了了前面的hoRatio%的数据作为测试集进行测试
		classifierResult = classify0(normDataSet[i,:], normDataSet[numTestVecs:m,:],datingLabels[numTestVecs:m],3) #返回分类结果
		print "index %d the classifierResult and the real classifierResult is: %d, %d" %(i,classifierResult,datingLabels[i])
		if(classifierResult != datingLabels[i]):	#分类结果和真是分类进行比较
			errorCount += 1.0;
	print "the total error ratio is %f" %(errorCount/float(numTestVecs)) 	#返回最终结果



#实际使用的程序
def classifyPerson():
	resultList = ["not at all", "small does", "large does"]
	para1 = float(raw_input("please in put the para1\n"))
	para2 = float(raw_input("please in put the para2\n"))
	para3 = float(raw_input("please in put the para3\n"))
	datingDataMat, datingLabels = file2matrix("data.txt")
	normDataSet, ranges, minVals = autoNorm(datingDataMat)
	intArr = array([para1, para2,para3])
	res = classify0((intArr - minVals)/ranges, normDataSet, datingLabels, 3)
	print resultList[res - 1]


#KNN简单生成数据
def createDataSet():
	group = array([[1.0, 1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels

# datingClassTest()
classifyPerson()















# retrunMat, classLabelVector = file2matrix("data.txt")
# print retrunMat, '\n', classLabelVector
# print
# normDataSet, ranges, minVals = autoNorm(retrunMat)
# print normDataSet, '\n'





# X = random.rand(10,3) 
# Y = random.rand(10,3)

# fig = plt.figure()			#生成图像
# ax = fig.add_subplot(111)
# #横坐标 纵坐标 大小 颜色	
# ax.scatter(normDataSet[:,1], normDataSet[:,2], 15.0 * array(classLabelVector),  15.0 * array(classLabelVector) )
# plt.show()					



# group,labels = createDataSet()

# print group
# print
# print 'group.min()\n', group.min()
# print
# print 'group.min(0)\n', group.min(0)
# print
# print classify0([0,0],group,labels,3)


#########################New Get#################################
# print group
# print
# print labels

# print group.shape[0]			#理解为行数
# print group.
# tmp =  tile([0,0], (group.shape[0],1))		#
# print tmp
# print tmp - group


