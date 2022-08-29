# Assignment taken from https://github.com/yubin-park/cs534/blob/master/homework/hw0.md

import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def problem1():
	df = pd.read_csv('hw0_p1.csv')
	dfm = df.melt('y', var_name='xn', value_name='x')  # Combine x1, x2 into single column
	sns.set_theme()
	fig, axs = plt.subplots(1, 3)
	sns.lineplot(ax=axs[0], x='x1', y='y', data=df, label='x1 v y')  # Lineplot of x1 vs y
	sns.lineplot(ax=axs[0], x='x2', y='y', data=df, label='x2 v y')  # Lineplot of x2 vs y
	sns.lineplot(ax=axs[0], x='x1', y='x2', data=df, label='x1 v x2')  # Lineplot of x1 vs x2
	sns.kdeplot(ax=axs[1], x='x', y='y', data=dfm)  # KDE plot of x vs y with x1, x2 coupled
	sns.kdeplot(ax=axs[2], x='x', y='y', data=dfm, hue='xn', legend=True)  # KDE plot of x vs y with x1, x2 decoupled
	axs[0].legend()
	plt.show()
	print(df.describe())


def problem2():
	df = pd.read_csv('AAPL.csv')
	df['Date'] = pd.to_datetime(df['Date'])  # Convert Date column to datetime object

	close_shift = df['Close'].shift()  # Create shifted version of Close column
	df['Daily_Return'] = df['Close'] / close_shift  # Calculate Daily Return
	df['Year'] = df['Date'].dt.year
	print(df.describe())  # Print descriptive information (including mean of Close and Daily Return

	fig, axs = plt.subplots(1, 2)
	sns.set_theme()

	sns.lineplot(ax=axs[0], x='Date', y='Close', data=df)  # Lineplot of Close values
	sns.lineplot(ax=axs[1], x='Date', y='Daily_Return', data=df, hue=df['Date'].dt.year, legend=True)  # LinePlot of Daily Return, colored by year

	# fg = sns.FacetGrid(data=df, col='Year')
	# fg.map(sns.lineplot, 'Date', 'Daily_Return')

	plt.show()


def problem3():
	# Milk is first index, juice is second
	milk_to_juice = np.array([[0.7, 0], [0.3, 1]])
	juice_to_milk = np.array([[1, 0.2], [0, 0.8]])
	mixtures = np.array([1, 1])
	cycles = np.zeros((2, 101))
	cycles[:, 0] = mixtures
	for cycle in range(1, cycles.shape[1]):
		mixtures = juice_to_milk @ milk_to_juice @ mixtures  # Pour milk into juice first, then juice into milk
		cycles[:, cycle] = mixtures

	df = pd.DataFrame({'Chocolate_Milk_Cup': cycles[0, :], 'Orange_Juice_Cup': cycles[1, :]})
	df['Cycle'] = df.index
	dfm = df.melt(id_vars='Cycle', value_vars=['Chocolate_Milk_Cup', 'Orange_Juice_Cup'])

	sns.set_theme()
	sns.lineplot(data=dfm, x='Cycle', y='value', hue='variable')
	plt.show()


if __name__ == '__main__':
	# problem1()
	# problem2()
	# problem3()
