# Assignment taken from https://github.com/yubin-park/cs534/blob/master/homework/hw0.md

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def problem1():
	df = pd.read_csv('hw0_p1.csv')
	dfm = df.melt('y', var_name='xn', value_name='x')   # Combine x1, x2 into single column
	sns.set_theme()
	fig, axs = plt.subplots(1, 3)
	sns.lineplot(ax=axs[0], x='x1', y='y', data=df, label='x1 v y')         # Lineplot of x1 vs y
	sns.lineplot(ax=axs[0], x='x2', y='y', data=df, label='x2 v y')         # Lineplot of x2 vs y
	sns.lineplot(ax=axs[0], x='x1', y='x2', data=df, label='x1 v x2')       # Lineplot of x1 vs x2
	sns.kdeplot(ax=axs[1], x='x', y='y', data=dfm)                          # KDE plot of x vs y with x1, x2 coupled
	sns.kdeplot(ax=axs[2], x='x', y='y', data=dfm, hue='xn', legend=True)   # KDE plot of x vs y with x1, x2 decoupled
	axs[0].legend()
	plt.show()
	print(df.describe())


def problem2():
	df = pd.read_csv('AAPL.csv')
	df['Date'] = pd.to_datetime(df['Date'])         # Convert Date column to datetime object

	close_shift = df['Close'].shift()               # Create shifted version of Close column
	df['Daily_Return'] = df['Close'] / close_shift  # Calculate Daily Return
	print(df.describe())                            # Print descriptive information (including mean of Close and Daily Return

	fig, axs = plt.subplots(1, 2)
	sns.set_theme()

	sns.lineplot(ax=axs[0], x='Date', y='Close', data=df)   # Lineplot of Close values
	sns.lineplot(ax=axs[1], x='Date', y='Daily_Return', data=df, hue=df['Date'].dt.year, legend=True)   # LinePlot of Daily Return, colored by year

	plt.show()


if __name__ == '__main__':
	# problem1()
	# problem2()
