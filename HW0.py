import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def problem1():
	df = pd.read_csv('hw0_p1.csv')
	dfm = df.melt('y', var_name='xn', value_name='x')
	sns.set_theme()
	fig, axs = plt.subplots(1, 3)
	sns.lineplot(ax=axs[0], x='x1', y='y', data=df, label='x1 v y')
	sns.lineplot(ax=axs[0], x='x2', y='y', data=df, label='x2 v y')
	sns.lineplot(ax=axs[0], x='x1', y='x2', data=df, label='x1 v x2')
	sns.kdeplot(ax=axs[1], x='x', y='y', data=dfm)
	sns.kdeplot(ax=axs[2], x='x', y='y', data=dfm, hue='xn', legend=True)
	axs[0].legend()
	plt.show()
	print(df.describe())


	def problem2():



if __name__ == '__main__':
	# problem1()
	problem2()
