import pandas as pd
import seaborn as sns

if __name__ == '__main__':
	df = pd.read_csv('hw0_p1.csv')
	sns.lineplot(x='x1', y='y', data=df)
