import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

mydata = sns.load_dataset('iris')

mydata.head()
mydata.info()
mydata['species'].value_counts()

from sklearn.model_selection import train_test_split
#X = mydata.drop('species', axis=1)
X = mydata.drop(columns=['species'])
Y = mydata['species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

from sklearn.linear_model import LogisticRegression 
model = LogisticRegression()
model.fit(x_train, y_train)

print("Accuracy:", model.score(x_test, y_test)*100)

import pickle
filename = 'iris-model.sav'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename,'rb'))
load_model.predict([[6.0, 2.2, 4.0, 1.0]])
