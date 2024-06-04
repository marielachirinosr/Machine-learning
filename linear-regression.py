# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Set dataset
homeprices = pd.read_csv("Data/homeprices.csv")

plt.xlabel('Area(sqr ft)')
plt.ylabel('Price(US$)')
plt.scatter(homeprices.Area, homeprices.Price, color = 'green', marker = '+')
#plt.show()

# Linear Regression Data
reg = LinearRegression()
reg.fit(homeprices[['Area']], homeprices.Price)

price_prediction = reg.predict([[3300]])

print(price_prediction)

print(reg.coef_)

print(reg.intercept_)