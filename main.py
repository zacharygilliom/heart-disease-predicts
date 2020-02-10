import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *


def exploratory_summary(df):
	print(df.head())
	print('***** NA values in each column')
	print(df.isna().sum())
	print('**** Shape of Our Data ****')
	print(df.shape)
	print(df.dtypes)
	print('******** Unique Values in Each Row ********')
	for col in df.columns:
		print(str(col) + ': ' + str(len(df[col].unique())))
	print(df.describe())
	return None

def plot(df):
	a = list(range(1, len(get_cols(df)) + 1))
	print(a)
	b = get_cols(df)
	print(b)
	c = len(get_cols(df))
	print(c)
	figure, axes = plt.subplots(nrows=c, ncols=1)
	figure.tight_layout(pad=1.05)
	for i, j in zip(a, b):
		print(i, j)
		plt.subplot(c,1,i)
		sns.distplot(df[j])
	plt.show()

def slice_data(df):
	df = df[['age', 'trestbps', 'chol', 'fbs', 'restecg',
		'thalach', 'ca']]
	return df

def get_cols(df):
	cols = []
	for col in df.columns:
		if len(df[col].unique()) > 5:
			cols.append(col)
	return cols


df = pd.read_csv('heart.csv')


sliced_data = slice_data(df)
# The column "Unnamed: 0" has no value as it is just a repeat of our index

# Lets get an idea of how our data is shaped
exploratory_summary(df)
# exploratory_summary(sliced_data)
# Lets do some plotting of our variables to see how they are related

plot(df)

# sns.pairplot(sliced_data)
# plt.show()
