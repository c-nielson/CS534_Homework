import re
from numpy import where
from tkinter import filedialog as fd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


def load_files():
	train_file = fd.askopenfilename(title="Select training file", filetypes=[("CSV", "*.csv")])
	validation_file = fd.askopenfilename(title="Select validation file", filetypes=[("CSV", "*.csv")])
	test_file = fd.askopenfilename(title="Select testing file (with labels)", filetypes=[("CSV", "*.csv")])

	df_train = pd.read_csv(train_file)
	df_validation = pd.read_csv(validation_file)
	df_test = pd.read_csv(test_file)

	return df_train, df_validation, df_test


def NB(train, validation, test, vec):
	nb = MultinomialNB()
	nb.fit(vec.transform(train['tweet']), train['target'])
	nb_y_pred = nb.predict(vec.transform(validation['tweet']))
	print(
		f'\nNaive Bayes Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], nb_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], nb_y_pred)}\n'
	)
	# TODO optimization


def RF(train, validation, test, vec):
	rf = RandomForestClassifier()
	rf.fit(vec.transform(train['tweet']), train['target'])
	rf_y_pred = rf.predict(vec.transform(validation['tweet']))

	print(
		f'\nRandom Forest Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], rf_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], rf_y_pred)}\n'
	)
	# TODO optimize


def GBDT(train, validation, test, vec):
	gbdt = GradientBoostingClassifier()
	gbdt.fit(vec.transform(train['tweet']), train['target'])
	gbdt_y_pred = gbdt.predict(vec.transform(validation['tweet']))

	print(
		f'\nGradient Boosted Classifier Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], gbdt_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], gbdt_y_pred)}\n'
	)
	# TODO optimize


def SVM(train, validation, test, vec):
	svm = SVC()
	svm.fit(vec.transform(train['tweet']), train['target'])
	svm_y_pred = svm.predict(vec.transform(validation['tweet']))

	print(
		f'\nSVC Validation Scores:\n'
		f'\tAccuracy: {accuracy_score(validation["target"], svm_y_pred)}\n'
		f'\tF1: {f1_score(validation["target"], svm_y_pred)}\n'
	)
	# TODO optimize


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
