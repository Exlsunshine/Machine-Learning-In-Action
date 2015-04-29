from numpy import *
import operator
from os import listdir

def createDataSet():
	group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels

def classify0(inX, dataSet, labels, k):
	#compute distances
	rows = dataSet.shape[0]
	diffMat = tile(inX, (rows, 1)) - dataSet
	sqrtMat = diffMat ** 2
	sqrtDistances = sqrtMat.sum(axis = 1)
	distances = sqrtDistances ** 0.5
	sortedDisIndicies = distances.argsort()

	#select k nearest items
	classCount = {}
	for i in range(k):
		voteLabel = labels[sortedDisIndicies[i]]
		classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

	sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedClassCount[0][0]

def img2vector(filename):
	vec = zeros((1, 1024))
	fp = open(filename)

	for i in range(32):
		line = fp.readline()
		for j in range(32):
			vec[0, 32 * i + j] = int(line[j])

	return vec

def handwritingClassTest():
	handwriting_labels = []
	training_files = listdir('trainingDigits')
	training_size = len(training_files)

	training_data = zeros((training_size, 1024))
	for i in range(training_size):
		filename = training_files[i]
		fileprefix = filename.split('.')[0]
		classifyTargetNum = int(fileprefix.split('_')[0])
		handwriting_labels.append(classifyTargetNum)
		training_data[i, :] = img2vector('trainingDigits/%s' % filename)

	test_files = listdir('testDigits')
	error_cnt = 0
	test_size = len(test_files)
	for i in range(test_size):
		filename = test_files[i]
		fileprefix = filename.split('.')[0]
		classifyTargetNum = int(fileprefix.split('_')[0])
		test_data = img2vector('testDigits/%s' % filename)

		classify_result = classify0(test_data, training_data, handwriting_labels, 3)
		print 'Classifier prediction is %d, real value is %d' % (classify_result, classifyTargetNum)

		if (classify_result != classifyTargetNum):
			error_cnt += 1

	print 'Error count is %d' % error_cnt
	print 'Error rate is %f' % (error_cnt * 1.0 / float(test_size))