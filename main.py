import pandas as pd 
import matplotlib
matplotlib.use('TkAgg', force=True)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler


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

# Want to plot the distribtution of our continuous variables.  
def plot(df):
	a = list(range(1, len(get_cols(df)) + 1))
	b = get_cols(df)
	c = len(get_cols(df))
	figure, axes = plt.subplots(nrows=c, ncols=1)
	figure.tight_layout(pad=1.05)
	colors = ['b', 'r', 'g', 'c', 'm']
	k = 0
	for i, j in zip(a, b):
		print(i, j)
		plt.subplot(c,1,i)
		sns.distplot(df[j], color=colors[k])
		k += 1
	plt.show()

# this function will grab all of the columns that are continuous.
def get_cols(df):
	cols = []
	for col in df.columns:
		if len(df[col].unique()) > 5:
			cols.append(col)
	return cols


df = pd.read_csv('heart.csv')

# Lets get an idea of how our data is shaped.  Looks like our data
# is organized and not missing any values.
exploratory_summary(df)

# Lets do some plotting of our variables to see how they are related.
plot(df)

# Selecting all of our variables as our features.
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
			'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features]


outcome = ['target']
y = df[outcome]

# We want our target variable to be a flattened array.
y = np.ravel(y)

# Since we only have 1 dataset, we will need to split it.
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=.8)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Initialize our model.  Chose LibLinear as this is the suggested model for 
# smaller datasets.
log_model = LogisticRegression(solver='liblinear', random_state=1,
								C=10)

log_model.fit(X_train, y_train)

# want to run some of our metrics on our model
print('*** probability prediction ***')
print(log_model.predict_proba(X_train))

print('** Model Score on Train Data ***')
print(log_model.score(X_train, y_train))

print('*** Model Score on Test Data ***')
print(log_model.score(X_test, y_test))

print('*** The Confusion Matrix ***')
cm = confusion_matrix(y_test, log_model.predict(X_test))
print(cm)

print('Classification Report')
print(classification_report(y_test, log_model.predict(X_test)))


# we can plot our confusion matrix to visualize the expected 
# outcomes vs the actual outcomes
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
