# -*- coding:utf-8 -*-

from numpy import *
from os import listdir
from math import log
import operator
import matplotlib
import matplotlib.pyplot as plt
import re

# 贝叶斯
# 理解的关键点是 条件概率 贝叶斯是条件概率意义相反的用法
# 可以这样理解:条件概率是指在大类确定后判断是某值的可能性,贝叶斯是反着来的,确定的某值属于某个类的可能性
# 这样理解起来,公式也就便于理解了
# 怀疑在实现中,公式中的分母因为是一样的,所以并没有实现
# 可以参照理解 http://blog.csdn.net/jinshengtao/article/details/39532043?utm_source=tuicool&utm_medium=referral
# 朴素贝叶斯之所以称为朴素是因为其假设特征之前是独立的,所以计算概率的时候可以进行简化,而不用考虑特征之间的关系
print "hello python"

def loadData():
	postingList=[['my','dog', 'has', 'flea', 'problems', 'help', 'please'],
				 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
				 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
				 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
				 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
				 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
	]
	classVec = [0,1,0,1,0,1]	#标称
	return postingList, classVec

# 获得文档中所有出现次的集合
def createVocabList(dataSet):
	vocabSet=set([])							#空集
	for docunment in dataSet:
		vocabSet = vocabSet | set(docunment)	#直接将某一个元素(是list)转化成为set,然后执行并集操作
	return list(vocabSet)						#再转换为list



# vocabList是词汇表,inputSet是给定的某文档
# 得到的结果是某个文档中在vocabList中词汇的出现情况
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)				# 和词汇表等长的向量
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word %s is not in my vocabulary\n" %s(word)
	return returnVec


#这里的trainCategory就是0,1值的list,就是classVec
#这里的trainMatrix应该是指原始的文档数据已经处理过得到的矩阵,一个很整齐的矩阵
def trainNB0(trainMatrix, trainCategory):
	numTrainDocs = len(trainMatrix)			#行数
	numWords = len(trainMatrix[0])			#列数
	print 'numWords', numWords,'\n'
	pAbusive = sum(trainCategory)/float(numTrainDocs)	#包含脏词的分类 二分问题,另一类就是 1 - Pabusive
	
	# p0Num = zeros(numWords)
	# p1Num = zeros(numWords)
	# p0Denom = 0.0
	# p1Denom = 0.0
	p0Num = ones(numWords)
	p1Num = ones(numWords)
	p0Denom = 2.0
	p1Denom = 2.0

	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num += trainMatrix[i]			#向量相加
			p1Denom += sum(trainMatrix[i])	
		else:
			p0Num += trainMatrix[i]
			p0Denom += sum(trainMatrix[i])
	# p1Vect = p1Num/p1Denom					#向量/float常数
	# p0Vect = p0Num/p0Denom
	p1Vect = [log(x) for x in p1Num/p1Denom]
	p0Vect = [log(x) for x in p0Num/p0Denom]
	return p0Vect, p1Vect, pAbusive

#都转成log计算方式,log(a*b) = loga + logb
def classfyNB(vec2Classify, p0Vec, p1Vec, pClass):
	p1 = sum(vec2Classify * p1Vec) + log(pClass)
	p0 = sum(vec2Classify * p0Vec) + log(1.0-pClass)
	if p1 > p0:
		return 1
	else:
		return 0

def testingNB():
	listOPOsts, listClass = loadData()
	myVocabList = createVocabList(listOPOsts)
	trainMat = []
	for postinDoc in listOPOsts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V,p1V,pAb = trainNB0(array(trainMat), array(listClass))

	tesEntry = ['love','my','dalmation']
	thisdoc = array(setOfWords2Vec(myVocabList, tesEntry))
	res = classfyNB(thisdoc, p0V, p1V, pAb)
	print res,'\n'

	tesEntry = ['stupid','garbage']
	thisdoc = array(setOfWords2Vec(myVocabList, tesEntry))
	res = classfyNB(thisdoc, p0V, p1V, pAb)
	print res,'\n'


#词袋
def bagOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
	return returnVec

#正则表达式进行文本字符串的单词切分,使用正则表达式可以方便进行标点符号的处理和空格的过滤,另外将单词统一成大写或者小写
def textParse(bigString):
	listOfWords = re.split(r'\W*', bigString)
	return [word.lower() for word in listOfWords if len(word) > 2]


def spamTest():
	docList = []; classList = []; fullText = []	#用于录入原始文件	文件矩阵,分类列表和全部文本
	
	for i in range(1,26):
		wordList = textParse(open('/home/sjc/python/Ch04/email/spam/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(1)

		wordList = textParse(open('/home/sjc/python/Ch04/email/ham/%d.txt' % i).read())
		docList.append(wordList)
		fullText.extend(wordList)
		classList.append(0)
	
	vocabList = createVocabList(docList)		#获得单词集合

	trainingSet = range(50); testSet = []		#训练集和测试集
	
	for i in range(10):							#随机选择其中的10个
		randIndex = int(random.uniform(0,len(trainingSet)))
		testSet.append(trainingSet[randIndex])
		del(trainingSet[randIndex])
	
	trainMat = []; trainClasses = []			#训练用的  矩阵和分类
	for docIndex in trainingSet:
		trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
		trainClasses.append(classList[docIndex])
	p0V,p1V,pSpam = trainNB0(array(trainMat), array(trainClasses))

	errCount = 0
	for docIndex in testSet:
		wordVector = setOfWords2Vec(vocabList, docList[docIndex])
		if classfyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
			errCount += 1
	print 'the error rate is %f' % (float(errCount) / len(testSet))

spamTest()



# fr = open('/home/sjc/python/Ch04/email/ham/1.txt')
# oneLine = fr.read()
# print oneLine.split()		#这样直接split出来的结果是有标点符号的,标点符号和其前一个单词在一起
# 
# regEx = re.compile('\\W*')	#使用正则表达式来进行文本切分
# print [word.lower() for word in regEx.split(oneLine) if len(word) > 0]

#test
# postingList, classVec = loadData()
# trainNB0(postingList,classVec)
# vocabList = createVocabList(postingList)
# print vocabList,'\n'
# returnVec = setOfWords2Vec(vocabList, postingList[1])
# print postingList[1],'\n'
# print returnVec





#testingNB()

