# -*- coding:utf-8 -*-

from numpy import *
from os import listdir
from math import log
import operator
import matplotlib
import matplotlib.pyplot as plt

#定义文本框和箭头格式
# decisonNode = dict(boxstyle="sawtooth", fc="0.8")
decisonNode = dict(boxstyle="round4", fc="0.8")
leafNode = dict(boxstyle="round4", fc = "0.8")
arrow_args = dict(arrowstyle="->")

#绘制箭头的注解
def plotNode(nodeTex, centerPt, parentPt, nodeType):
	createPlot.ax1.annotate(nodeTex, xy=parentPt,\
		xycoords='axes fraction',\
		xytext=centerPt,\
		textcoords='axes fraction',\
		va = "center", \
		ha="center", \
		bbox=nodeType, \
		arrowprops=arrow_args)
#绘制点和标签
def createPlot():
	fig = plt.figure(1, facecolor='white')	#创建一个绘制区
	fig.clf()								#清空
	createPlot.ax1 = plt.subplot(111, frameon=False)
	plotNode('a decison node', (0.5,0.1),(0.1,0.5),decisonNode)
	plotNode('a leaf node', (0.8,0.1),(0.3,0.8),leafNode)
	plt.show()


def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			numLeafs += getNumLeafs(secondDict[key])
		else:
			numLeafs += 1
	return numLeafs


def getTreeDept(myTree):
	maxDept = 0
	firstStr = myTree.keys()[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDept = 1 + getTreeDept(secondDict[key])
		else:
			thisDept = 1
		if thisDept > maxDept: maxDept = thisDept
	return maxDept




createPlot()