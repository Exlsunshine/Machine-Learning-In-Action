from math import log
import operator

def create_dataset():
	dataset = [	[1, 1, 'yes'],
				[1, 1, 'yes'],
				[1, 0, 'no'],
				[0, 1, 'no'],
				[0, 1, 'no']]
	labels = ['no surfacing', 'flippers']
	return dataset, labels

def calc_shannon(dataset):
	rows = len(dataset)
	labels = {}

	for row in dataset:
		label = row[-1]
		if label not in labels.keys():
			labels[label] = 0
		labels[label] += 1
	
	shannon_ent = 0
	for key in labels:
		probability = labels[key] * 1.0 / rows
		shannon_ent -= probability * log(probability, 2)

	return shannon_ent

def split_dataset(dataset, axis, value):
	split_dataset = []

	for row in dataset:
		if row[axis] == value:
			reduced_row = row[:axis]
			reduced_row.extend(row[axis + 1:])
			split_dataset.append(reduced_row)

	return split_dataset

def choose_best_feature(dataset):
	#the last coulumn is the class that this row belongs to
	#so we have to remove this coulumn
	feature_size = len(dataset[0]) - 1
	base_entropy = calc_shannon(dataset)
	best_info_gain = 0;
	best_feature = 0;

	for i in range(feature_size):
		feature_list = [example[i] for example in dataset]
		values = set(feature_list)
		entropy = 0;

		for value in values:
			sub_dataset = split_dataset(dataset, i, value)
			entropy += calc_shannon(sub_dataset)

		if base_entropy - entropy > best_info_gain:
			best_info_gain = base_entropy - entropy
			best_feature = i

	return best_feature

def majority_count(class_list):
	cnt = {}

	for vote in class_list:
		if vote not in cnt.keys():
			cnt[vote] = 0
		cnt[vote] += 1

	sorted_cnt = sorted(cnt.iteritems(), key = operator.itemgetter(1), reverse = True)
	return sorted_cnt[0][0]

def create_decision_tree(dataset, labels):
	class_list = [example[-1] for example in dataset]

	#if all classes are the same, just return the class
	if class_list.count(class_list[0]) == len(class_list):
		return class_list[0]

	#if reach to the last feature and still not reach to a conclusion
	#then just vote for the class and then return the result
	if len(dataset[0]) == 1:
		return majority_count(class_list)

	best_feature_index = choose_best_feature(dataset)
	best_feature = labels[best_feature_index]
	decision_tree = {best_feature: {}}

	del(labels[best_feature_index])
	values = set([example[best_feature_index] for example in dataset])
	for value in values:
		child_labels = labels[:]
		decision_tree[best_feature][value] = create_decision_tree(split_dataset(dataset, best_feature_index, value), child_labels)

	return decision_tree

def classify(inputTree,labels,testVec):
	root_key = inputTree.keys()[0]
	root_dictioinary = inputTree[root_key]
	label_index = labels.index(root_key)
	for key in root_dictioinary.keys():
		if testVec[label_index] == key:
			if type(root_dictioinary[key]).__name__=='dict':
				classLabel = classify(root_dictioinary[key],labels,testVec)
			else:
				classLabel = root_dictioinary[key]
	
	return classLabel

def store_decision_tree(decision_tree, file_name):
	import pickle
	fp = open(file_name, 'w+')
	pickle.dump(decision_tree, fp)
	fp.close()

def retrive_decision_tree(file_name):
	import pickle
	fp = open(file_name)
	return pickle.load(fp)