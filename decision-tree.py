#Import libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import numpy as np

#Read CSV
data = pd.read_csv('Learning Machine Learning/Data/salaries.csv')

#Encoder labels in columns
label_enconder = preprocessing.LabelEncoder()
data['company'] = label_enconder.fit_transform(data['company'])
data['job'] = label_enconder.fit_transform(data['job'])
data['degree'] = label_enconder.fit_transform(data['degree'])

#Split the dataset in features and target variable
feature_col = ['company', 'job', 'degree']
X = data[feature_col]
y = data['salary_more_then_100k']
print(X)
print(y)

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=100)

#Create decision tree classifier object usign entropy
clf_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=8)

#Train decision tree classifier
clf_entropy = clf_entropy.fit(X_train, y_train)

#Predict response for dataset
y_pred = clf_entropy.predict(X_test)
print('Accuracy', metrics.accuracy_score(y_test, y_pred))

# Define possible values for new data
companies = ['facebook', 'google', 'abc pharma']
jobs = ['sale executive', 'business manager', 'computer programmer']
degrees = ['bachelors', 'masters']

# Create a new DataFrame with all combinations of company, job, and degree
new_data = pd.DataFrame([(c, j, d) for c in companies for j in jobs for d in degrees], 
                        columns=['company', 'job', 'degree'])

# Preprocess new data (encoding using the same label encoders)
new_data['company'] = label_enconder.fit_transform(new_data['company'])
new_data['job'] = label_enconder.fit_transform(new_data['job'])
new_data['degree'] = label_enconder.fit_transform(new_data['degree'])

# Extract features from new data
X_new = new_data[feature_col]

# Predict on new data
y_new_pred = clf_entropy.predict(X_new)

# Print predictions for the new data
new_data['salary_more_then_100k'] = y_new_pred
print('Predictions for new data:')
print(new_data)