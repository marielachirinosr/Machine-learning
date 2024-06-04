#Importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random 

#Load CSV
user_data = pd.read_csv('Data/user+data.csv')

# Encode 'gender' column to numerical values
label_encoder = LabelEncoder()
user_data['Gender'] = label_encoder.fit_transform(user_data['Gender'])

X = user_data.iloc[:, [1, 3]].values
y = user_data.iloc[:, 2].values

#Training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#KKN Classifier 
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier (n_neighbors=5, metric="minkowski", p=2)
classifier.fit(X_train, y_train)

#Making confusion matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
print(cm)

# Display actual vs predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(results)

# Example new data 
new_data_point =[]

for x in list(range(1000)):
    new_data_point.append([random.randint(0, 1), random.randint(25000, 200000)])


# Scale the new data point using the same scaler as the training data
new_data_point_scaled = sc_X.transform(new_data_point)

# Predict the class label for the new data
predicted_label = classifier.predict(new_data_point_scaled)

print("\nPredicted Class Label:", predicted_label)

