# -*- coding:utf-8 -*-

from numpy import *
from os import listdir
from math import log
import operator
import matplotlib
import matplotlib.pyplot as plt


#计算给定数据集的香农熵  数据集的最后一列是分类信息
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)		#数据集中数据条数
	labelCount = {}					#字典用于统计分类
	for featVec in dataSet:
		currLabel = featVec[-1]		#取出分类标签
		if currLabel not in labelCount.keys():
			labelCount[currLabel] = 0
		labelCount[currLabel] += 1
	# print labelCount
	shannonEnt = 0.0				#香农熵
	for key in labelCount:
		prob = float(labelCount[key]) / float(numEntries)	#概率
		shannonEnt -= prob * log(prob,2)
		# print shannonEnt					#计算香农熵累加
	return shannonEnt 

#按照给定特征划分数据集,参数分别是数据集,维度和维度中的值
def splitDataSet(dataSet, axis, value):
	retDataSet = []					#python传的是引用,避免对原来的值造成影响
	for featVec in dataSet:
		# print featVec
		if featVec[axis] == value:	#将已经用完的当前属性去掉,剩下其它属性列
			retFeatVec = featVec[:axis]
			retFeatVec.extend(featVec[axis + 1:])	#extend同类扩展元素个数
			retDataSet.append(retFeatVec)			#append将参数作为一个元素放到原来的里面
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	nunEntries = len(dataSet)
	numFeature = len(dataSet[0]) - 1 #去掉最后的标签
	baseShannonEnt = calcShannonEnt(dataSet)
	bestInfoGain = 0.0
	bestFeature = -1
	for  i in range(numFeature):		#队每一个属性进行处理 :按照这个属性进行分类 并计算信息熵,然后比较所有属性的信息熵并找出最好的一个
		featList = [example[i] for example in dataSet]	#得到当前列的所有取值  列表推导 python
		uniqueVal = set(featList)		#去掉重复项
		newShannonEnt = 0.0
		#按照每个属性进行划分
		#信息增益:熵的减少或者称作无序度的减少 原本的信息熵减去划分后的信息熵得到信息增益的度量  熵减少越多越好,也就是信息增益越大越好
		for value in uniqueVal:
			subDataSet = splitDataSet(dataSet, i, value)		#得到按照当前属性值的分类方式
			prob = float(len(subDataSet)) /  float(nunEntries)
			newShannonEnt += prob * calcShannonEnt(subDataSet)
		infoGain = baseShannonEnt - newShannonEnt				#信息熵衡量的是信息的无序程度,越大越无序,信息增益是指数据的无序度减少的程度,所以 老减新 越大越好
		if infoGain > bestInfoGain:
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

#当所有的属性全部都用完了进行划分,得到的最后的数据集仍然不是全部属于一个类的情况
#需要进行少数服从多数的选举
def majorCnt(classList):
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	#dict的内容项类似  分类1:12
	#排序函数 迭代器 以统计值排序(12)  逆序  返回排序好的dict 原dict不变
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]	#返回的是分类名称


#递归方式创建生成树  参数是数据集 和 属性列表
def createTree(dataSet, labels):
	#首先是出口
	classList = [example[-1] for example in dataSet]	#当前数据集的类别
	#list 的 count函数返回的是参数在list中出现的次数 
	if classList.count(classList[0]) == len(classList):	#当前数据集中所有都是属于一个类别
		return classList[0]
	if len(dataSet[0]) == 1:	#只有1列  也就是第0行只有一列	#当前数据集中不都是属于同一个类别但是只有一个用于划分的属性
		return majorCnt(classList)

	bestFeat = chooseBestFeatureToSplit(dataSet)		#还需要继续划分 选出一个信息增益最大的属性 这里返回的是列下标

	bestFeatLabel = labels[bestFeat]					#属性名称
	myTree = {bestFeatLabel:{}}							#此属性为当前子树的根
	del(labels[bestFeat])
	featValues = [example[bestFeat] for example in dataSet] #此属性的所有值
	uniqueVals = set(featValues)						#此属性的所有不同值
	for value in uniqueVals:
		subLabels = labels[:]							#复制label列表
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree




#简单生成数据
def createDataSet():
	#下面这两种方式是有区别的
	# group = array([[1.0, 1.0, 'yes'], [1.0, 1.0, 'yes'],[1.0, 0.0, 'yes'],[0.0, 0.0, 'no'],[0.0, 1.0, 'no']])
	group = [[1.0, 1.0, 'yes'], [1.0, 1.0, 'yes'],[1.0, 0.0, 'no'],[0.0, 1.0, 'no'],[0.0, 1.0, 'no']]
	labels = ['A','B']
	return group, labels

dataSet, labels = createDataSet()
# print calcShannonEnt(dataSet)
# print splitDataSet(dataSet, 0, 1)
# print splitDataSet(dataSet, 0, 0)
# print [example[0] for example in dataSet]
#print chooseBestFeatureToSplit(dataSet)
print createTree(dataSet,labels)

