#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Set dataset
startups = pd.read_csv("Data/50_Startups.csv")

#Apply index to X and Y
X = startups.iloc[:,:-1]
y = startups.iloc[:, -1].values

print(startups.head(5))

#Convert categorical values to numeric values with OneHotEnconder 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
X = np.array(ct.fit_transform(X))

#Splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#Training the model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
print(df)

#Evaluating the performance of the regression model
from sklearn import metrics
print('Mean Squared Error', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
