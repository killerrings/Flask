import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pickle 
import requests
import json


#Reading the csv file
df = pd.read_csv('student_scores.csv')
df.head()

#Defining our independent and dependent varibales for our plot\n",
x=df['Hours'].values.reshape(-1, 1)
y=df['Scores'].values.reshape(-1, 1)


#Splitting our data into training and test set in a precentage of 80 to 20 respectively\n",
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 42)

#Calling our Linear Regression model from the scikit-learn library
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Defining our prediction variable
y_predictions = regressor.predict(x_test)

#Saving model to disk
pickle.dump(regressor, open('model.pkl', 'wb'))

#Loading model to compare the results
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1.8]]))
