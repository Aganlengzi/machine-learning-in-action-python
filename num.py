# -*- coding:utf-8 -*-

from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt

# print "hello"

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

#将32*32的文件中的数字转换成1*1024的数据 也就是相当于是1条数据
def image2vector(filename):
	returnVect = zeros((1,1024))		#声明一个1*1024的的对象
	fr = open(filename)					#打开文件
	for i in range(32):
		lineStr = fr.readline()			#读取一行,一共32行
		for j in range(32):				#对这一行进行操作
			returnVect[0,32*i+j] = int(lineStr[j])	#拼接到这一行中,每个数字都要进行处理后并放在后面拼接上
	return returnVect					#返回


# print image2vector("/home/sjc/python/Ch02/trainingDigits/1_16.txt")
print listdir("/home/sjc/python/Ch02/trainingDigits/")	#列出给定目录下的所有文件名称,list

def handWritingClassTest():
	hwLabels = []
	dirStr = "/home/sjc/python/Ch02/trainingDigits/"
	trainingFileList = listdir("/home/sjc/python/Ch02/trainingDigits")
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		filenameStr = trainingFileList[i]
		fileStr = filenameStr.split('.')[0]	#应该是得到前缀字符串
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = image2vector(dirStr + filenameStr)
	# return trainingMat
	testFileList = listdir("/home/sjc/python/Ch02/testDigits/")
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		filenameStr = testFileList[i]
		fileStr = filenameStr.split('.')[0]
		trueClass = int(fileStr.split('_')[0])
		testVect = image2vector("/home/sjc/python/Ch02/testDigits/" + filenameStr)
		classfierResult = classify0(testVect, trainingMat, hwLabels, 3)
		if(classfierResult != trueClass):
			errorCount += 1.0
	print "the error ratio is %f\n" %(errorCount/float(mTest))


handWritingClassTest()











