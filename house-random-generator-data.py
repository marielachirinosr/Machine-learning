import pandas as pd
import numpy as np

# Generate synthetic data
np.random.seed(0)  

# Define the number of samples
num_samples = 1000

# Generate features: number of bedrooms, size of the house, and location
bedrooms = np.random.randint(1, 6, size=num_samples)
size = np.random.randint(1000, 3001, size=num_samples) 
location = np.random.choice(['City', 'Suburb', 'Rural'], size=num_samples)

# Generate target variable: house prices
location_prices = {'City': 50000, 'Suburb': 30000, 'Rural': 10000}
prices = (bedrooms * 10000) + (size * 100) + np.array([location_prices[loc] for loc in location])

# Create a DataFrame
house_data = pd.DataFrame({'Bedrooms': bedrooms, 'Size': size, 'Location': location, 'Price': prices})

# Save the DataFrame to a CSV file
house_data.to_csv('house_prices.csv', index=False)

# Display the first few rows of the dataset
print(house_data.head())

