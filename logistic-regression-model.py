#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import seaborn as sb

#Read CSV and set data
user_data = pd.read_csv('Data/user+data.csv')
X = user_data.iloc[:, [2, 4]].values
y = user_data.iloc[:, 4].values

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

#Pre-processing
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#Training the model
classifier = LogisticRegression(random_state=0)
classifier.fit (X_train, y_train)

#Results
y_pred = classifier.predict(X_test)
print(y_pred)

cm = confusion_matrix(y_pred, y_test)
print(cm)

#Heatmap
sb.heatmap(cm)
plt.show()
