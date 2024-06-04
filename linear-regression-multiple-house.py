# Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn import metrics

# Load the dataset
house_data = pd.read_csv("Data/house_prices.csv")

# Define features (X) and target (y)
X = house_data[['Bedrooms', 'Size', 'Location']]
y = house_data['Price']

# Define column transformer for one-hot encoding
column_transformer = ColumnTransformer([('encoder', OneHotEncoder(), [2])], remainder='passthrough')

# Apply one-hot encoding to the 'Location' feature and convert to numpy array
X_encoded = column_transformer.fit_transform(X)

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.4)

# Training the model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Evaluating the performance of the regression model
mse = metrics.mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Displaying actual vs. predicted values
results = pd.DataFrame({'Actual Price': y_test, 'Predicted Price': y_pred})
print(results.head())

# Create new data with bedrooms, size, and city 
new_data = pd.DataFrame({'Bedrooms': [3, 4, 2, 4, 2, 2, 3, 4, 5, 3],
                         'Size': [1500, 2000, 1200, 2030, 1500, 2328, 3000, 2200, 1010, 1250],
                         'Location': ['City', 'Rural', 'City','Suburb', 'Suburb', 'City','Rural', 'City', 'City', 'Suburb']})

# Apply one-hot encoding to the 'Location' feature for new data
new_data_encoded = column_transformer.transform(new_data)

# Predict house prices for the new data
predicted_prices = regressor.predict(new_data_encoded)

# Display the predicted prices
predicted_prices_df = pd.DataFrame({'Predicted Price': predicted_prices})
print(predicted_prices_df)