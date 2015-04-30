from numpy import *

def loadDataSet():
	postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
	['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
	['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
	['stop', 'posting', 'stupid', 'worthless', 'garbage'],
	['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
	['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
	classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
	return postingList,classVec
	
def createVocabList(dataSet):
	vocabSet = set([])
	for document in dataSet:
		vocabSet = vocabSet | set(document)
	return list(vocabSet)
	
def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] = 1
		else:
			print "the word: %s is not in my Vocabulary!" % word
	return returnVec

def trainNB0(training_data, labels):
	rows = len(training_data)
	columns = len(training_data[0])

	#abusive
	probability_label1 = sum(labels) * 1.0 / rows
	#unabusive
	probability_label0 = (columns - sum(labels)) * 1.0 / rows
	label_0_freq = ones(columns)
	label_1_freq = ones(columns)
	label_0_sum = 2
	label_1_sum = 2

	for i in range(rows):
		if labels[i] == 1:
			label_1_freq += training_data[i]
			label_1_sum += sum(training_data[i])
		else :
			label_0_freq += training_data[i]
			label_0_sum += sum(training_data[i])

	probability_label0_vect = log(label_0_freq * 1.0 / label_0_sum)
	probability_label1_vect = log(label_1_freq * 1.0 / label_1_sum)

	return probability_label1_vect, probability_label0_vect, probability_label1

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
	p1 = sum(vec2Classify * p1Vec) + log(pClass1)
	p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)

	if p1 > p0 :
		return 1
	else :
		return 0

def testingNB():
	posts, labels = loadDataSet()
	words = createVocabList(posts)
	
	training_data = []
	for post in posts:
		training_data.append(setOfWords2Vec(words, post))
	
	p1Vec,p0Vec,pAb = trainNB0(array(training_data),array(labels))

	testEntry = ['love', 'my', 'dalmation']
	thisDoc = array(setOfWords2Vec(words, testEntry))
	print testEntry,'classified as: ', classifyNB(thisDoc,p0Vec,p1Vec,pAb)
	testEntry = ['stupid', 'garbage']
	thisDoc = array(setOfWords2Vec(words, testEntry))
	print testEntry,'classified as: ', classifyNB(thisDoc,p0Vec,p1Vec,pAb)



######################################################################################
#Spam classifier
def bagOfWords2VecMN(vocabList, inputSet):
	returnVec = [0]*len(vocabList)
	for word in inputSet:
		if word in vocabList:
			returnVec[vocabList.index(word)] += 1
		else:
			print "the word: %s is not in my Vocabulary!" % word
	return returnVec

def textParse(text):
	import re
	tokens = re.split(r'\w*', text)
	tokens = [tok.lower() for tok in tokens if len(tok) > 2]

	return tokens
# def textParse(text):
# 	import re
	
# 	tokens = re.compile('\\W*').split(text)
# 	tokens = [tok.lower() for tok in tokens if len(tok) > 0]

# 	return tokens



def spamTest():
	email_bodys = []
	labels = []
	
	for i in range(1, 26):
		email_body = textParse(open('email/spam/%d.txt' % i).read())
		email_bodys.append(email_body)
		labels.append(1)

		email_body = textParse(open('email/ham/%d.txt' % i).read())
		email_bodys.append(email_body)
		labels.append(0)

	#randomly pick up 10 data, set them to testing data
	#the remain 40 data are used for training data
	training_data_index = range(50)
	testing_data = []
	for i in range(10):
		randIndex = int(random.uniform(0,len(training_data_index)))
		testing_data.append(training_data_index[randIndex])
		del(training_data_index[randIndex])

	training_data_mat = []
	training_data_labels = []
	words = createVocabList(email_bodys)
	for i in training_data_index:
		training_data_mat.append(setOfWords2Vec(words, email_bodys[i]))
		training_data_labels.append(labels[i])
	
	p1V, p0V, pSpam = trainNB0(array(training_data_mat), array(training_data_labels))
	errorCount = 0
	for docIndex in testing_data:
		wordVector = setOfWords2Vec(words, email_bodys[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != labels[docIndex]:
			errorCount += 1

	print 'the error rate is: ',float(errorCount) / len(testing_data)
	print 'the error count is: ',errorCount

######################################################################################
#RSS feed
def calcMostFreq(words, text):
	import operator
	freqDict = {}
	for word in words:
		freqDict[word] = text.count(word)

	sortedFreq = sorted(freqDict.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sortedFreq[:30]

def localWords(feed1, feed0):
	import feedparser
	tokens_list = []
	labels = []
	full_text =[]
	min_length = min(len(feed1['entries']), len(feed0['entries']))
	for i in range(min_length):
		tokens = advanced_reg_parse(feed1['entries'][i]['summary'])
		tokens_list.append(tokens)
		full_text.extend(tokens)
		labels.append(1)

		tokens = advanced_reg_parse(feed0['entries'][i]['summary'])
		tokens_list.append(tokens)
		full_text.extend(tokens)
		labels.append(0)

	#get and delte top-30 most frequent words
	words = createVocabList(tokens_list)
	top30Words = calcMostFreq(words, full_text)
	for word in top30Words:
		if word[0] in words:
			words.remove(word[0])
	
	#randomly pick up 20 data, set them to testing data
	#the remain data are used for training data
	training_data_index = range(2 * min_length)
	testing_data_index = []
	for i in range(20):
		randIndex = int(random.uniform(0,len(training_data_index)))
		testing_data_index.append(training_data_index[randIndex])
		del(training_data_index[randIndex])

	#train
	training_data_mat = []
	training_data_labels = []
	for docIndex in training_data_index:
		training_data_mat.append(bagOfWords2VecMN(words, tokens_list[docIndex]))
		training_data_labels.append(labels[docIndex])
	p1V,p0V,pSpam = trainNB0(array(training_data_mat),array(training_data_labels))

	#test
	errorCount = 0
	for docIndex in testing_data_index:
		wordVector = bagOfWords2VecMN(words, tokens_list[docIndex])
		if classifyNB(array(wordVector),p0V,p1V,pSpam) != labels[docIndex]:
			errorCount += 1
	print 'the error rate is: ',float(errorCount)/len(testing_data_index)
	return words,p0V,p1V

def getTopWords(ny,sf):
	import operator

	words, p0, p1 = localWords(ny,sf)
	top1 = []
	top0 = []
	for i in range(len(p0)):
		if p0[i] > -6.0 :
			top0.append((words[i], p0[i]))
		if p1[i] > -6.0 :
			top1.append((words[i], p1[i]))
	
	sortedSF = sorted(top0, key = lambda pair: pair[1], reverse = True)
	print "\nSF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF\n"
	
	cnt = 0
	for item in sortedSF:
		print item[0]
		if cnt == 10:
			break
		else :
			cnt += 1
	
	sortedNY = sorted(top1, key=lambda pair: pair[1], reverse=True)
	print "\nNY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY\n"
	
	cnt = 0
	for item in sortedNY:
		print item[0]
		if cnt == 10:
			break
		else :
			cnt += 1

def advanced_reg_parse(text):
	import re
	tokens = re.compile('\\W*').split(text)
	tokens = [tok.lower() for tok in tokens if len(tok) > 2]

	return tokens
