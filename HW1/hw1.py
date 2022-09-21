import io
import math

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


def load_data() -> dict[str: pd.DataFrame]:
	"""
	Used to load data from current directory
	:return: dictionary of data splits
	"""
	train_data = pd.read_csv('train.csv')
	validation_data = pd.read_csv('validation.csv')
	test_data = pd.read_csv('test.csv')
	return {'train': train_data, 'validation': validation_data, 'test': test_data}


def print_scores(f: io.TextIOWrapper, data: dict[str: pd.DataFrame], columns_to_drop: list[str], target: str, model: linear_model) -> None:
	"""
	Prints the R2 and RMSE values for each data split in 'data' using 'model' to predict
	:param f: file object for writing results
	:param data: dictionary of data splits
	:param model: trained linear_model
	:param columns_to_drop: list of columns to drop from 'data' for training models
	:param target: name of target column in 'data'
	:return: None
	"""
	for data_name in data:  # Iterate through data sets and calculate RMSE, R2 scores
		prediction = model.predict(data[data_name].drop(columns_to_drop, axis=1))  # Calculate prediction of model
		rmse = math.sqrt(mean_squared_error(data[data_name][target], prediction))  # Calculate RMSE
		r2 = r2_score(data[data_name][target], prediction)  # Calculate R2
		f.write(f'\tScores for {data_name}:\n\t\tR2={r2}\n\t\tRMSE={rmse}\n')  # Report values
	f.write('\n')


def LinearRegression(f: io.TextIOWrapper, data: dict[str: pd.DataFrame], columns_to_drop: list[str], target: str) -> None:
	"""
	Used for Problem 1 of HW1. Generates LinearRegression, Ridge, and Lasso models, comparing results and testing various parameters.
	:param f: file object used for writing
	:param data: dictionary of dataframes representing different data splits
	:param columns_to_drop: list of columns to drop from 'data' for training models
	:param target: name of target column in 'data'
	:return: None
	"""
	train_valid_join = pd.concat([data['train'], data['validation']])  # DataFrame containing both the training and validation datasets

	# P1.1, LinearRegression on training data:
	# 	Scores for train:
	# 		R2=0.18674709354493324
	# 		RMSE=9650.203054172724
	# 	Scores for validation:
	# 		R2=0.002645573525469902
	# 		RMSE=9513.284018866663
	# 	Scores for test:
	# 		R2=-0.2119519180786953
	# 		RMSE=10008.256002148079
	f.write('P1.1, LinearRegression on training data:\n')
	lin_model = linear_model.LinearRegression()
	lin_model.fit(data['train'].drop(columns_to_drop, axis=1), data['train'][target])  # Fit model to training data
	print_scores(f, data, columns_to_drop, target, lin_model)

	# P1.2, LinearRegression on training and validation data:
	# 	Scores for train:
	# 		R2=0.16754354896477064
	# 		RMSE=9878.075716031653
	# 	Scores for validation:
	# 		R2=0.17512826483285027
	# 		RMSE=7868.054612760931
	# 	Scores for test:
	# 		R2=0.09588838543361966
	# 		RMSE=7466.121681989215
	#
	# These scores are better than P1.1, for both validation and test, although slightly worse for the training set; maybe P1.1 was slightly overfit
	# for the training set?
	lin_model = linear_model.LinearRegression()
	lin_model.fit(
		train_valid_join.drop(columns_to_drop, axis=1), train_valid_join[target]
	)  # Fit to both training and validation data
	f.write('\nP1.2, LinearRegression on training and validation data:\n')
	print_scores(f, data, columns_to_drop, target, lin_model)

	# P1.3, Ridge and Lasso using various alpha (lambda) values:
	# Ridge, alpha=0.0:
	# 	Scores for train:
	# 		R2=0.18674709354493313
	# 		RMSE=98.23544703503275
	# 	Scores for validation:
	# 		R2=0.0026455735254680146
	# 		RMSE=97.5360652213666
	# 	Scores for test:
	# 		R2=-0.21195191807869818
	# 		RMSE=100.0412714940594
	#
	# Lasso, alpha=15:
	# 	Scores for train:
	# 		R2=0.12260606988545875
	# 		RMSE=102.03583587825472
	# 	Scores for validation:
	# 		R2=0.06748971196472864
	# 		RMSE=94.31207229529994
	# 	Scores for test:
	# 		R2=0.06711402202273953
	# 		RMSE=87.77094824377073
	f.write('\nP1.3, Ridge and Lasso using various alpha (lambda) values:\n')
	for alpha in np.arange(0, 1.1, 0.1):
		lin_model = linear_model.Ridge(alpha=alpha)
		lin_model.fit(data['train'].drop(columns_to_drop, axis=1), data['train'][target])  # Fit Ridge to training data
		f.write(f'Ridge, alpha={alpha}:\n')
		print_scores(f, data, columns_to_drop, target, lin_model)

	for alpha in np.arange(0, 21, 1):
		lin_model = linear_model.Lasso(alpha=alpha)
		lin_model.fit(data['train'].drop(columns_to_drop, axis=1), data['train'][target])  # Fit Lasso to training data
		f.write(f'Lasso, alpha={alpha}:\n')
		print_scores(f, data, columns_to_drop, target, lin_model)

	# P1.4, Ridge and Lasso using alpha=0 for ridge, and alpha=15 for lasso (from P1.3), model trained on training and validation data:
	# Ridge, alpha=0.0:
	# 	Scores for train:
	# 		R2=0.16754354896477053
	# 		RMSE=99.38850897378255
	# 	Scores for validation:
	# 		R2=0.1751282648328506
	# 		RMSE=88.70205529051132
	# 	Scores for test:
	# 		R2=0.09588838543362033
	# 		RMSE=86.40672243517405
	#
	# Lasso, alpha=15:
	# 	Scores for train:
	# 		R2=0.10793309087785619
	# 		RMSE=102.8854896064948
	# 	Scores for validation:
	# 		R2=0.08120641800705086
	# 		RMSE=93.61586361931055
	# 	Scores for test:
	# 		R2=0.06876888435157158
	# 		RMSE=87.69306449844295
	#
	# These results are again better for validation and testing sets, and slightly worse for the training set. Very similar to P1.2.
	f.write('\nP1.4, Ridge and Lasso using alpha=0 for ridge, and alpha=15 for lasso (from P1.3), model trained on training and validation data:\n')
	alpha = 0.0
	lin_model = linear_model.Ridge(alpha=alpha)
	lin_model.fit(
		train_valid_join.drop(columns_to_drop, axis=1), train_valid_join[target]
	)  # Fit Ridge to training and validation data
	f.write(f'Ridge, alpha={alpha}:\n')
	print_scores(f, data, columns_to_drop, target, lin_model)

	alpha = 15
	lin_model = linear_model.Lasso(alpha=alpha)
	lin_model.fit(
		train_valid_join.drop(columns_to_drop, axis=1), train_valid_join[target]
	)  # Fit Lasso to training and validation data
	f.write(f'Lasso, alpha={alpha}:\n')
	print_scores(f, data, columns_to_drop, target, lin_model)


# P2, using lambda=10
# LinearRegression (fake) coefficients:
# [  2.49284448 -12.52338979  11.10284545 -10.49253611 -18.583615
#   22.35180157  15.34052285  -1.46144575  -1.57519006  -0.65109693
#   -0.23510003   5.43466862  -0.0682671    0.98664692   0.4060793
#    2.44227443  -4.42358158  -0.89960344  -5.40385219 -17.29441212
#    0.45279313  -2.24204553   0.14356633   0.23403793  16.16481039]
#
# Ridge coefficients:
# [  2.49284447 -12.52338977  11.10284548 -10.49253617 -18.58361505
#   22.3518016   15.34052278  -1.46144571  -1.57519004  -0.65109704
#   -0.23510002   5.43466858  -0.06826712   0.98664691   0.4060793
#    2.44227449  -4.42358159  -0.89960376  -5.40385213 -17.294413
#    0.45279312  -2.2420457    0.14356628   0.23403793  16.16481133]
def FakeRidgeRegression(f: io.TextIOWrapper, data: dict[str: pd.DataFrame], columns_to_drop: list[str], target: str) -> None:
	"""
	Method for Problem 2 of homework 1.
	:param f: File object to write results to.
	:param data: dict of data splits
	:param columns_to_drop: list of columns to drop for fitting
	:param target: target to fit to
	:return: None
	"""
	alpha = 10
	augmented_data = data['train'].copy()
	for label, _ in augmented_data.items():
		if label != 'date':
			augmented_data[label] = augmented_data[label] - augmented_data[label].mean()
	augment = np.identity(augmented_data.shape[1] - len(columns_to_drop)) * math.sqrt(alpha)
	augment = np.concatenate((np.zeros((augment.shape[0], len(columns_to_drop))), augment), axis=1)
	augment = pd.DataFrame(augment)
	augment.columns = augmented_data.columns.values
	augmented_data = pd.concat([augmented_data, augment], ignore_index=True)
	lin_model = linear_model.LinearRegression()
	lin_model.fit(augmented_data.drop(columns_to_drop, axis=1), augmented_data[target])
	f.write(f'\n\nP2, using lambda={alpha}\n')
	f.write(f'LinearRegression (fake) coefficients:\n{lin_model.coef_}')

	lin_model = linear_model.Ridge(alpha=alpha)
	lin_model.fit(data['train'].drop(columns_to_drop, axis=1), data['train'][target])
	f.write(f'\n\nRidge coefficients:\n{lin_model.coef_}')


def LinearRegressionWithKnowledge():
	...


if __name__ == '__main__':
	data = load_data()
	columns_to_drop = ['date', 'Appliances']
	target = 'Appliances'
	filename = 'hw1_Nielson.txt'
	print(f'All calculated values will be written to {filename}')
	with open(filename, 'w') as f:
		# LinearRegression(f, data, columns_to_drop, target)
		# FakeRidgeRegression(f, data, columns_to_drop, target)
		LinearRegressionWithKnowledge()
