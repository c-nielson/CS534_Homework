import math
import re
from numpy import where
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import multiprocessing


n_jobs = math.floor(multiprocessing.cpu_count() * 0.25)


def load_files():
	train_file = fd.askopenfilename(title="Select training file", filetypes=[("CSV", "*.csv")])
	validation_file = fd.askopenfilename(title="Select validation file", filetypes=[("CSV", "*.csv")])
	test_file = fd.askopenfilename(title="Select testing file (with labels)", filetypes=[("CSV", "*.csv")])

	df_train = pd.read_csv(train_file)
	df_validation = pd.read_csv(validation_file)
	df_test = pd.read_csv(test_file)

	return df_train, df_validation, df_test


def NB(train, validation, test, vec):
	params = {
		'alpha': np.linspace(0, 1000, 100001)
	}

	nb = MultinomialNB()
	gs = GridSearchCV(nb, params, n_jobs=n_jobs)
	print('\nOptimizing Naive Bayes...')
	gs.fit(vec.transform(train['tweet']), train['target'])
	print(gs.best_params_)
	print(gs.best_score_)

	nb = gs.best_estimator_
	nb.fit(vec.transform(train['tweet']), train['target'])
	nb_y_pred = nb.predict(vec.transform(validation['tweet']))

	print(
		f'\nNaive Bayes Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], nb_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], nb_y_pred)}\n'
	)


def RF(train, validation, test, vec):
	params = {
		'n_estimators': np.arange(0, 1001),
		'min_samples_leaf': np.linspace(0, 20, 101)
	}

	rf = RandomForestClassifier(random_state=0)
	gs = GridSearchCV(rf, params, n_jobs=n_jobs)
	print('\nOptimizing Random Forest...')
	gs.fit(vec.transform(train['tweet']), train['target'])
	print(gs.best_params_)
	print(gs.best_score_)

	rf = gs.best_estimator_
	rf.fit(vec.transform(train['tweet']), train['target'])
	rf_y_pred = rf.predict(vec.transform(validation['tweet']))

	print(
		f'\nRandom Forest Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], rf_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], rf_y_pred)}\n'
	)


def GBDT(train, validation, test, vec):
	params = {
		'n_estimators': np.arange(0, 1001),
		'min_samples_leaf': np.linspace(0, 20, 101),
		'learning_rate': np.arange(0, 101, 0.1)
	}

	gbdt = GradientBoostingClassifier(random_state=0)
	gs = GridSearchCV(gbdt, params, n_jobs=n_jobs)
	print('\nOptimizing GBDT...')
	gs.fit(vec.transform(train['tweet']), train['target'])
	print(gs.best_params_)
	print(gs.best_score_)

	gbdt = gs.best_estimator_
	gbdt.fit(vec.transform(train['tweet']), train['target'])
	gbdt_y_pred = gbdt.predict(vec.transform(validation['tweet']))

	print(
		f'\nGradient Boosted Classifier Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], gbdt_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], gbdt_y_pred)}\n'
	)


def SVM(train, validation, test, vec):
	params = {
		'kernel': ['linear', 'polynomial', 'sigmoid'],
		'C': np.linspace(0, 100, 1001)
	}

	svm = SVC(random_state=0)
	gs = GridSearchCV(svm, params, n_jobs=n_jobs)
	print('\nOptimizing SVC...')
	gs.fit(vec.transform(train['tweet']), train['target'])
	print(gs.best_params_)
	print(gs.best_score_)

	svm = gs.best_estimator_
	svm.fit(vec.transform(train['tweet']), train['target'])
	svm_y_pred = svm.predict(vec.transform(validation['tweet']))

	print(
		f'\nSVC Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], svm_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], svm_y_pred)}\n'
	)


if __name__ == "__main__":
	# Load data files
	df_train, df_validation, df_test = load_files()

	# Create corpus from all tweets
	corpus = []
	for tweet in df_train['tweet']:
		corpus.append(tweet)
	for tweet in df_validation['tweet']:
		corpus.append(tweet)
	for tweet in df_test['tweet']:
		corpus.append(tweet)

	# Preprocess and tokenize corpus of tweets
	vec = CountVectorizer(ngram_range=(1, 1), preprocessor=lambda t: re.sub(r'[^\w\s]', '', t))
	vec.fit_transform(corpus)

	# Transform labels into numeric; real==1, fake==0
	df_train['target'] = where(df_train['label'] == 'real', 1, 0)
	df_validation['target'] = where(df_validation['label'] == 'real', 1, 0)
	df_test['target'] = where(df_test['label'] == 'real', 1, 0)

	NB(df_train, df_validation, df_test, vec)
	RF(df_train, df_validation, df_test, vec)
	GBDT(df_train, df_validation, df_test, vec)
	SVM(df_train, df_validation, df_test, vec)
