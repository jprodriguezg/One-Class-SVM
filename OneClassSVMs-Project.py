from __future__ import print_function
from glob import glob
import itertools
import os.path
import re
import sys
import tarfile
import time
import copy
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.sparse import csr_matrix

from sklearn.externals.six.moves import html_parser
from sklearn.externals.six.moves import urllib
from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import Binarizer
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import train_test_split

 


###############################################################################
# Reuters Dataset related routines
###############################################################################


class ReutersParser(html_parser.HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding='latin-1'):
        html_parser.HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


###############################################################################
# Useful functions
###############################################################################

## This function calls the dataset previously saved on the machine
def stream_reuters_documents(data_path=None):
	parser = ReutersParser()

	# If none data_path is define the entire Reuters dataset is loaded (The articles of the 22 files)
	if data_path is None:
		print("Loading the entire Reuters dataset ... ")
		data_path = os.path.join(get_data_home(), "reuters")
		for filename in glob(os.path.join(data_path, "*.sgm")):
			for doc in parser.parse(open(filename, 'rb')):
				yield doc

	# If an specific file is selected, only the articles of that file are loaded
	else:
		print("Loading the dataset (articles) of the specified file ... ")
		for doc in parser.parse(open(data_path, 'rb')):
			yield doc

## From the dataset takes only the articles which correspond to the topic on pos_class. It also returns a list of tulpes
def get_topic_articles(doc_iter, size, pos_class):
	data = []
	for doc in itertools.islice(doc_iter, size):
		for i in doc['topics']:
			if i == pos_class:
				# Note that the addition of the tuple of ones, is only used to put the label to +1
				data.append(((u'{title}\n\n{body}'.format(**doc))))
	print('The number of files found with the topic: ' + pos_class + ' is  ' + str(len(data)) + "\n")
	return data

# Function to binarize the matrix X
def binarize_set(X):
	binarize = Binarizer(threshold=0.0, copy = False)
	out = binarize.transform(X, y=None, copy=None)
	return out
	
# Select the n_features more frequent on the observations
def reduce_features(X, n_features = 10):

	# It's only working based on the features which higher frequency among all the observations
	X_bin = copy.copy(X)
	binarize_set(X_bin)
	weights = np.ravel(X_bin.sum(axis = 0))
	f_index = (-weights).argsort()[:n_features]
	
	X_features = np.zeros((np.size(X,0),f_index.size))
	for i in range(f_index.size):
		X_features[:,i] = np.ravel(X[:,f_index[i]])
	return X_features

# Find the outliers given the threshold
def find_outliers(X, threshold = 3):
	y = -np.ones(np.size(X,0))
	X_aux = copy.copy(X)
	binarize_set(X_aux)
	for i in range(y.size):
		if np.sum(X_aux[i,:]) >= threshold:
			y[i] = 1
	return y

# Fiter the outliers in order to obtain a matrix with only observations which satisfied the threshold
def filter_outliers(X, y_outliers):
	u, times = np.unique(y_outliers, return_counts = True)
	if times.size > 1:
		X_filtered = np.zeros((times[1],np.size(X,1)))
		count = 0
		for i in range(y_outliers.size):
			if y_outliers[i] == 1:
				X_filtered[count,:] = X[i,:]
				count += 1
		return X_filtered	
	else:
		print("The matrix does not have outliers ")
		X_filtered = X
		return X_filtered


def hadamard_product(X):
	#ve = np.sum(X,axis=0)
	X_train, X_train_filtered, X_test, y_outliers_train, y_outliers_test = dataset_split(X,5, on_loop = 1)
	ve = np.sum(X_train,axis=0)
	ve = np.array(ve).flatten()
	for i in range(np.size(X,0)):
		for j in range(np.size(X,1)):
			X[i,j] = X[i,j]*ve[j]

	return X

# Transforms the dataset of words to an numeric representation and select the most important features.
# There are 3 types of representations 1) Binary, 2) Normalized Frequency representation and 3) tf-idf representation. 
# The output matrix has number of feautures equal to the n most important features on the dataset.
def dataset_vectorization(X_text, n_important_features, doc_representation_type ='binary'):

	# Generates the vectorizer and the limit number of features to a reasonable # maximum
 	# Notes:
 	# If you want to know the vocabulary or/and the stop_words please use the attributes vectorizer.vocabulary_ and vectorizer.stop_words_
 	# If you want to have a big set of stop_words please add "stop_words = 'english'" (It reduces considerably the dimension output of the 
 	# vectorized function)  
	vectorizer = CountVectorizer(min_df = 1, analyzer = 'word', stop_words = 'english')
	
	# This is an important opertaion! Here we pass from having a document of words to have an sparce matrix of integers (document x
	# features -> Tokens -> words). The result represets the X matrix of the dataset. In order to handle the sparse as an array please use
	# X.toarray() for an array of array or X.todense() to obtain a matrix.
	X_text_vec = vectorizer.fit_transform(X_text)
	# Transform an sparse matrix into a full matrix
	X_text_vec = X_text_vec.todense()

	# Find the n more frequent features
	# Here we finally have the matrix X of observations x features
	X = reduce_features(X_text_vec, n_features = n_important_features)
	print("The number of features per observation has been reduced to " +str(np.size(X,1)))
	print("Type of document representation: " +doc_representation_type+"\n")

	if doc_representation_type == 'binary':
		binarize_set(X)
	elif doc_representation_type == 'Nfrequency':
		transformer = TfidfTransformer(use_idf = 'False')
		X = transformer.fit_transform(X)
		X = X.todense()
	elif doc_representation_type == 'tf-idf':
		transformer = TfidfTransformer(use_idf = 'True')
		X = transformer.fit_transform(X)
		X = X.todense()
	elif doc_representation_type == 'hadamard':
		X = hadamard_product(X)

	return X

# Splits the X dataset into training and testing sets.
# The outliers labels are computed based on the threshold_outliers input. It determines whether an observation is an outlier or not
# The X_train_filtered set is a training set used only by the OneClassSVM.
def dataset_split(X, threshold_outliers, on_loop = 0):

	# Split X in training set and testing set
	# NOTE: The train_size is fixed to the 25% of the original X by following the approach used on the guide paper
	X_train, X_test, y_train, y_test = train_test_split(X, np.ones(np.size(X,0)), train_size=0.25, random_state=42)

	# Given the threshold, defines the observations considered as outliers 
	y_outliers_train = find_outliers(X_train,threshold = threshold_outliers)
	y_outliers_test = find_outliers(X_test,threshold = threshold_outliers)	

	# Create a X  training matrix only with the observations which are not outliers (Only used by OneClassSVM)
	X_train_filtered = filter_outliers(X_train, y_outliers_train)

	if on_loop == 0:
		print("The threshold for determine the outliers is:  " + str(threshold_outliers))
		print(str(np.size(X_train_filtered,0)) +"  document(s) for training the One-Class SVM")	
		print(str(np.size(X_train,0)) +"  document(s) for training the Outliers SVM")
		print(str(np.size(X_test,0)) +"  document(s) for testing the SVM \n")

	return X_train, X_train_filtered, X_test, y_outliers_train, y_outliers_test
	
# Determines the accuracy of the OneClassSVM
def ScoreOneClassSVM(y, y_predicted):
	count = 0
	if len(y) == len(y_predicted):
		for i in range(len(y)):
			if y[i] == y_predicted[i]:
				count +=1
		return float(count)/len(y)
	else:
		print("ERROR: y and y_predicted have not the same lenght")
		return -1

# An special function to perform the crossvalidation process on OneClassSVM
def OneClass_cross_val_score (svm_classiffier, X, y, n_folds = 5):

	factor = int(float(len(y))/n_folds)

	X_folded = list()
	y_folded = list()
	for i in range(n_folds):
	
		# Condition to take all the data of X and y
		if i == n_folds-1:
			X_folded.append(X[i*factor:len(y)-1,:]) 
			y_folded.append(y[i*factor:len(y)-1])
		else:
			X_folded.append(X[i*factor:(i+1)*factor-1,:]) 
			y_folded.append(y[i*factor:(i+1)*factor-1])

	# Initialize X_test and X_train
	X_train = np.asarray([], dtype=int)
	X_test = np.asarray([], dtype=int)

	scores = np.zeros(n_folds)
	# This for Perform the crossvalidation n (n_folds) times
	for i in range(n_folds):
		# Select one (i) of the folds of X_folds as X_test. Same process for y
		X_test = X_folded[i]
		y_test = y_folded[i]
		# The flag is only used to perform the concatenations
		flag = 1
		# This for is used to create the X_train with the n-1 folds
		for j in range(n_folds):
			# Condition to don't select the X_test on the X_train folds
			if i != j:		
				if flag == 1:
					X_train = X_folded[j]
					y_train = y_folded[j]
					flag = 0
				else:
					X_train = np.concatenate((X_train,X_folded[j]),axis=0)
					y_train = np.concatenate((y_train,y_folded[j]))

		u, n_times = np.unique(y_train, return_counts = True)
		#print ('y_train on round ',str(i),' has :',' u  ', u, ' times  ', n_times,"  The value of C is:  ",str(svm_classiffier.C))

		X_train_filtered = filter_outliers(X_train, y_train)
		svm_classiffier.fit(X_train_filtered)
		y_predicted = svm_classiffier.predict(X_test)
		# Change X_train. Only a caution!
		X_train = np.asarray([], dtype=int)
		X_test = np.asarray([], dtype=int)

		scores[i] = ScoreOneClassSVM(y_test, y_predicted)
	#print(scores)
	return scores

# Tunes the value of C which maximize the accuracy of the Outliers-SVM
def CTune_SVM(svm_classiffier, X_train, y_train, n_folds = 5, inf_pow = -2, sup_pow = 2, SVM ='none'):

	# Generates an array with the powers of the Cs
	power = range(inf_pow,sup_pow+1,1)
	# Initialize the array where will be safe the possible values of C
	C_range = np.zeros(len(power))
	# Initialize the array with the accuracies of each crossvalidation proccess (one for each value of C)
	accuracies = np.zeros(len(power))
	# This for is used to create an array with the possible values of C
	for i in range(len(power)):
			C_range[i] = math.pow(10,power[i])

	# Evaluate the accuracy of the classifier using the different values of C
	for i in range(len(C_range)):
		# Change the C value of the SVM
		svm_classiffier.C = C_range[i]
		# Perform a crossvalidation on the X_train set using number of folds given by n_folds
		if SVM == "OneClass":
			#print("Test with C:  ",str(clf.C))
			scores = OneClass_cross_val_score (svm_classiffier, X_train, y_train, n_folds = n_folds)
		else:
			scores = cross_val_score(svm_classiffier, X_train, y_train, cv=n_folds)
		# Determine the result of the crossvalidation making the average of the result of each validation test
		accuracies[i] = np.average(scores)

	print(C_range)
	print(accuracies)
	C_tunned = C_range[np.argmax(accuracies)]
	print("The tunned value of C for the Outliers-SVM is:  " + str(C_tunned))
	return C_tunned

# Computes the performance parameter comparing the y_real and y_predicted arrays
def performance_parameters(y_real, y_predicted, on_loop):
	# array of statistics [TP TN FP FN]
	out = np.zeros(4)
	if y_real.size == y_predicted.size:
		for i in range(y_real.size):
			# True positive
			if y_real[i] == 1 and y_real[i] == y_predicted[i]:
				out[0] += 1
			# True negative
			elif y_real[i] == -1 and y_real[i] == y_predicted[i]:
				out[1] += 1	
			# False positive
			elif y_real[i] == -1 and y_real[i] != y_predicted[i]:
				out[2] += 1	
			# False negative
			elif y_real[i] == 1 and y_real[i] != y_predicted[i]:
				out[3] += 1
	else: 
		print(" The input vectors real and prediction don't have the same lenght")	
		return out

	if on_loop == 0:
		print("Performance parameters:")
		print("TP: %d  TN: %d  FP: %d  FN: %d" %(out[0],out[1],out[2],out[3]))	

	return out
		
# Returns the performance of the classifer
def clf_performance(y_real,y_predicted, on_loop = 0):
	
	# Obtain the statistics of the results [TP TN FP FN]
	param = performance_parameters(y_real,y_predicted, on_loop)
	
	# Recall -> R = TP/(TP+FN)
	R = param[0]/(param[0]+param[3])	
	# Precision -> P = TP/(TP+FP)	
	P = param[0]/(param[0]+param[2])
	# Accuracy  -> acc = TP+TN / TP+TN+FP+FN
	acc =(param[0]+param[1])/(param[0]+param[1]+param[2]+param[3])
	# True positive rate TPR 
	TPR = R
	# False positive Rate -> FPR = FP/FP+TN (FNR = FN/FN+TP)
	FPR = param[2]/(param[2]+param[1])
	# F1 measure
	F1 = 2*R*P/(R+P)

	if on_loop == 0:
		print("Performance results :")	
		print("F1: %.3f  Recall: %.3f  Precision: %.3f  Accuracy: %.3f  TPR: %.3f  FPR: %.3f \n" %(F1,R,P,acc,TPR,FPR))	
		return F1,P,R
	else:
		return F1,P,R,acc,FPR,param[0],param[1],param[2],param[3]


def loop_test(document_topic,X_text):

	address = '/home/juan-laptop/Dropbox/AIRO/Neural Networks/Results/'
	textfile = open(address+document_topic+'.txt','w')
	textfile.write('SVM,n_features,representation,kernel,F1,P,R,Acc,FPR,TP,TN,FP,FN\n')

	n_features = np.array([10,25,50,100])
	representations = np.array(['binary', 'Nfrequency', 'tf-idf','hadamard'])
	kernel = np.array(['linear','poly','rbf','sigmoid'])

	for i in n_features:

		print('Evualating n_features : '+str(i))
		vectorizer = CountVectorizer(min_df = 1, analyzer = 'word', stop_words = 'english')
		X_text_vec = vectorizer.fit_transform(X_text)
		# Transform an sparse matrix into a full matrix
		X_text_vec = X_text_vec.todense()
		X = reduce_features(X_text_vec, n_features = i)

		for j in representations:
			print('Document representation : '+j)
			if j == 'binary':
				binarize_set(X)
			elif j == 'Nfrequency':
				transformer = TfidfTransformer(use_idf = 'False')
				X = transformer.fit_transform(X)
				X = X.todense()
			elif j == 'tf-idf':
				transformer = TfidfTransformer(use_idf = 'True')
				X = transformer.fit_transform(X)
				X = X.todense()
			elif j == 'hadamard':
				X = hadamard_product(X)

			X_train, X_train_filtered, X_test, y_outliers_train, y_outliers_test = dataset_split(X,5, on_loop = 1)

			for k in kernel:
				print('Kernel : '+k)
				clf = svm.OneClassSVM(kernel=str(k), max_iter = 1000000, cache_size = 200, nu = 0.2)
				clf.fit(X_train_filtered)
				y_predicted_OneClass = clf.predict(X_test)
				F1,P,R,Acc,FPR,TP,TN,FP,FN = clf_performance(y_outliers_test, y_predicted_OneClass, on_loop = 1)
				textfile.write('OneClass'+','+str(i)+','+j+','+k+','+str(F1)+','+str(P)+','+str(R)+','+str(Acc)+','+str(FPR)+','+str(TP)+','+str(TN)+','+str(FP)+','+str(FN)+'\n')

				clc = SVC(kernel = str(k))
				clc.C = 100
				clc.fit(X_train, y_outliers_train)
				y_predicted_outliers = clc.predict(X_test)
				F1,P,R,Acc,FPR,TP,TN,FP,FN = clf_performance(y_outliers_test, y_predicted_outliers, on_loop = 1)
				textfile.write('Outliers'+','+str(i)+','+j+','+k+','+str(F1)+','+str(P)+','+str(R)+','+str(Acc)+','+str(FPR)+','+str(TP)+','+str(TN)+','+str(FP)+','+str(FN)+'\n')
			

	textfile.close()
	print('finish!!')
	
	



###############################################################################
# Main
###############################################################################


###############################################################################
# Loading and preparting the dataset of documents
###############################################################################

# Iterator over parsed Reuters SGML files. In other words it loads the Reuters dataset
data_stream = stream_reuters_documents()

# Define the topic of the document to select
#['earn','acq','money-fx','grain','crude','trade','interest','ship','wheat','corn']
document_topic = 'earn'
#Note that n_test_docuement is the number of documents to extract from the dataset. The maximum is 1000 (given by the Reuters dataset)
n_documents = 20000
# Call the dataset of Reuters on "data_stream", with a number of documents equal to n_documents
X_text = get_topic_articles(data_stream, n_documents, document_topic)

# Use this to make the loop test (Generates txt a file with the results of the text changing the n_features, kernel and doc_represetation)
#loop_test(document_topic,X_text)
###############################################################################
# Vectorizing and spliting
###############################################################################

most_important_features = 25
# Transforms the X_text set into a vectorized representation given the desired doc_representation.  
X = dataset_vectorization(X_text, most_important_features, doc_representation_type ='binary')
# Split and filter the X dataset. The resulting subsets are used for training and testing the SVMs
X_train, X_train_filtered, X_test, y_outliers_train, y_outliers_test = dataset_split(X,5, on_loop = 0)

###############################################################################
# One-Class SVM Classification
###############################################################################

# Call the OneClassSVM. 
#Note the important parameters of the OneClassSVM 'nu' corresponds to the 'v' parameter
clf = svm.OneClassSVM(kernel='rbf', max_iter = 1000000, cache_size = 200, nu = 0.2)

# Train the OneSVM with the X_train_filtered_bin matrix
clf.fit(X_train_filtered)
print("OneClass-SVM trained .... ")
# Test the OneSVM with the X_test_bin sparce matrix
y_predicted_OneClass = clf.predict(X_test)
print("OneClass-SVM tested  .... ")
F1 , R, P = clf_performance(y_outliers_test, y_predicted_OneClass, on_loop = 0)

###############################################################################
# Outlier SVM Classification
###############################################################################

# Initialize a classic SVM	
clc = SVC(kernel = 'rbf')
# Use this only for tunning C of the SVM (From the experiments the best value of C is 100)
#C_tuned = CTune_SVM(clc, csr_matrix(X_train_bin), y_outliers_train, n_folds = 5, inf_pow = -4,  sup_pow = 4)
clc.C = 100
print("The selected value of C for the OutliersSVM is:  ",str(clc.C))
# Train the SVM with X_train
clc.fit(X_train, y_outliers_train)
print("Outliers-SVM trained .... ")
# Predict the labels of X_test
y_predicted_outliers = clc.predict(X_test)
print("Outliers-SVM test .... ")
F1 , R, P = clf_performance(y_outliers_test, y_predicted_outliers, on_loop = 0)
