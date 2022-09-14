# Assignment taken from https://github.com/yubin-park/cs534/blob/master/homework/hw0.md

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def problem1():
	df = pd.read_csv('hw0_p1.csv')
	dfm = df.melt('y', var_name='xn', value_name='x')  # Combine x1, x2 into single column

	sns.set_theme()
	sns.pairplot(data=df)
	plt.suptitle('Problem 1 - Pair Plot')
	plt.tight_layout()
	plt.show()

	fig, axs = plt.subplots(1, 2)
	sns.set_theme()
	sns.kdeplot(ax=axs[0], x='x', y='y', data=dfm).set_title('x1, x2 Coupled')                              # KDE plot of x vs y with x1, x2 coupled
	sns.kdeplot(ax=axs[1], x='x', y='y', data=dfm, hue='xn', legend=True).set_title('x1, x2 Decoupled')     # KDE plot of x vs y with x1, x2 decoupled
	plt.suptitle('Problem 1 - KDE Plot')
	plt.tight_layout()
	plt.show()

	sns.set_theme()
	df.hist(layout=(1, 3))
	plt.suptitle('Problem 1 - Histogram')
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

	sns.lineplot(ax=axs[0], x='Date', y='Close', data=df).set_title('Closing Prices')  # Lineplot of Close values
	sns.lineplot(ax=axs[1], x='Date', y='Daily_Return', data=df, hue=df['Date'].dt.year, legend=True).set_title('Daily Return')  # LinePlot of Daily Return, colored by year

	# Alternate code that splits each year into separate graphs for plotting Daily_Return
	# fg = sns.FacetGrid(data=df, col='Year')
	# fg.map(sns.lineplot, 'Date', 'Daily_Return')

	plt.suptitle('Problem 2')
	plt.tight_layout()
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
	dfm = df.melt(id_vars='Cycle', value_vars=['Chocolate_Milk_Cup', 'Orange_Juice_Cup'], value_name='Volume')

	sns.set_theme()
	sns.lineplot(data=dfm, x='Cycle', y='Volume', hue='variable').set_title('Problem 3')
	plt.show()


if __name__ == '__main__':
	problem1()
	problem2()
	problem3()
