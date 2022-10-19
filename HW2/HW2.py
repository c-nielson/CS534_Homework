import time
import math
import re
import time

from numpy import where
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import multiprocessing

# n_jobs = math.floor(multiprocessing.cpu_count() * 0.25)
n_jobs = 20


def load_files():
	"""
	Loads the training and validation sets from selected files and returns the corresponding DataFrames.
	:return: df_train, df_validation DataFrames
	"""
	train_file = fd.askopenfilename(title="Select Training File", filetypes=[("CSV", "*.csv")])
	validation_file = fd.askopenfilename(title="Select File to Use for Scoring", filetypes=[("CSV", "*.csv")])

	df_train = pd.read_csv(train_file)
	df_validation = pd.read_csv(validation_file)

	return df_train, df_validation


# Optimum Parameters:
# 	alpha=0.09
def NB(train, validation, vec):
	"""
	Used to optimize and train a Multinomial Naive Bayes model. Commented out code was previously used for parameter optimization.
	:param train: Training DataFrame
	:param validation: Validation DataFrame
	:param vec: Vectorizer for tokenizing tweets
	:return: None
	"""
	# params = {
	# 	'alpha': np.linspace(0, 1000, 100001)
	# }

	nb = MultinomialNB(alpha=0.09)
	# gs = GridSearchCV(nb, params, n_jobs=n_jobs)
	# print('\nOptimizing Naive Bayes...')
	# gs.fit(vec.transform(train['tweet']), train['target'])
	# print(gs.best_params_)
	# print(gs.best_score_)

	# nb = gs.best_estimator_
	nb.fit(vec.transform(train['tweet']), train['target'])
	nb_y_pred = nb.predict(vec.transform(validation['tweet']))

	print(
		f'\nNaive Bayes Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], nb_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], nb_y_pred)}\n'
	)


# Optimum Parameters:
# 	n_estimators: 49
#	min_samples_leaf: 1
def RF(train, validation, vec):
	"""
	Used to optimize and train a Random Forest. Commented out code was previously used for parameter optimization.
	:param train: Training DataFrame
	:param validation: Validation DataFrame
	:param vec: Vectorizer for tokenizing tweets
	:return: None
	"""
	# params = {
	# 	'n_estimators': np.arange(1, 51),
	# 	'min_samples_leaf': np.arange(11)
	# }

	rf = RandomForestClassifier(min_samples_leaf=1, n_estimators=49)
	# gs = GridSearchCV(rf, params, n_jobs=n_jobs)
	# print('\nOptimizing Random Forest...')
	# gs.fit(vec.transform(train['tweet']), train['target'])
	# print(gs.best_params_)
	# print(gs.best_score_)

	# rf = gs.best_estimator_
	rf.fit(vec.transform(train['tweet']), train['target'])
	rf_y_pred = rf.predict(vec.transform(validation['tweet']))

	print(
		f'\nRandom Forest Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], rf_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], rf_y_pred)}\n'
	)


# Optimum Parameters:
# 	n_estimators: 58
# 	min_samples_leaf: 11
# 	learning_rate: 0.8
def GBDT(train, validation, vec):
	"""
	Used to optimize and train a Gradient Boosted Decision Tree. Commented out code was previously used for parameter optimization.
	:param train: Training DataFrame
	:param validation: Validation DataFrame
	:param vec: Vectorizer for tokenizing tweets
	:return: None
	"""
	# params = {
	# 	'n_estimators': np.arange(50, 61),
	# 	'min_samples_leaf': np.arange(7, 12),
	# 	'learning_rate': np.arange(0.8, 1.1, 0.1)
	# }

	gbdt = GradientBoostingClassifier(n_estimators=58, min_samples_leaf=11, learning_rate=0.8)
	# rs = RandomizedSearchCV(gbdt, params, n_jobs=n_jobs, n_iter=50)
	# print('\nOptimizing GBDT...')
	# rs.fit(vec.transform(train['tweet']), train['target'])
	# print(rs.best_params_)
	# print(rs.best_score_)
	# print(rs.cv_results_)

	# gbdt = rs.best_estimator_
	gbdt.fit(vec.transform(train['tweet']), train['target'])
	gbdt_y_pred = gbdt.predict(vec.transform(validation['tweet']))

	print(
		f'\nGradient Boosted Classifier Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], gbdt_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], gbdt_y_pred)}\n'
	)


# Optimum Parameters:
# 	kernel: linear
# 	C: 2.0
def SVM(train, validation, vec):
	"""
	Used to optimize and train an SVC. Commented out code was previously used for parameter optimization.
	:param train: Training DataFrame
	:param validation: Validation DataFrame
	:param vec: Vectorizer for tokenizing tweets
	:return: None
	"""
	# params = {
	# 	'kernel': ['linear', 'poly', 'sigmoid'],
	# 	'C': np.linspace(0, 100, 51)
	# }

	svm = SVC(C=2, kernel='linear')
	# gs = GridSearchCV(svm, params, n_jobs=n_jobs)
	# print('\nOptimizing SVC...')
	# gs.fit(vec.transform(train['tweet']), train['target'])
	# print(gs.best_params_)
	# print(gs.best_score_)

	# svm = gs.best_estimator_
	svm.fit(vec.transform(train['tweet']), train['target'])
	svm_y_pred = svm.predict(vec.transform(validation['tweet']))

	print(
		f'\nSVC Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], svm_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], svm_y_pred)}\n'
	)


if __name__ == "__main__":
	# Load data files
	df_train, df_validation = load_files()

	# Create corpus from all tweets
	corpus = []
	for tweet in df_train['tweet']:
		corpus.append(tweet)
	for tweet in df_validation['tweet']:
		corpus.append(tweet)

	# Preprocess and tokenize corpus of tweets
	vec = CountVectorizer(ngram_range=(1, 1), preprocessor=lambda t: re.sub(r'[^\w\s]', '', t))
	vec.fit_transform(corpus)

	# Transform labels into numeric; real==1, fake==0
	df_train['target'] = where(df_train['label'] == 'real', 1, 0)
	df_validation['target'] = where(df_validation['label'] == 'real', 1, 0)

	NB(df_train, df_validation, vec)
	RF(df_train, df_validation, vec)
	GBDT(df_train, df_validation, vec)
	SVM(df_train, df_validation, vec)
