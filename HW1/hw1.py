import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def load_data():
	train_data = pd.read_csv('energy_complete/train.csv')
	test_data = pd.read_csv('energy_complete/test.csv')
	validation_data = pd.read_csv('energy_complete/validation.csv')
	return {'train': train_data, 'test': test_data, 'validation': validation_data}


def LinearRegression():
	data = load_data()	# Dictionary of train, test, validation
	lin_reg = linear_model.LinearRegression()
	lin_reg.fit(data['train'].iloc[:, 3:], data['train']['Appliances'])
	for data_name in data:
		rmse = mean_squared_error(data[data_name]['Appliances'], lin_reg.predict(data[data_name].iloc[:, 3:]))
		print(f'Scores for {data_name}:\n\tR2={lin_reg.score(data[data_name].iloc[:, 3:], data[data_name]["Appliances"])}\n\tRMSE={rmse}')


if __name__ == '__main__':
	LinearRegression()
