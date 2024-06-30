Basic implementation of a linear regression model for predicting house prices using Python with scikit-learn:

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'house_data.csv' with your dataset path)
data = pd.read_csv('house_data.csv')

# Display the first few rows of the dataset and check columns
print(data.head())
print(data.columns)

# Select features and target variable
X = data[['sqft_living', 'bedrooms', 'bathrooms', 'floors']]  # Example features
y = data['price']  # Target variable

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = model.predict(X_test)

# Print model coefficients and intercept
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Mean Squared Error:', mse)
print('R-squared:', r2)

# Plotting predicted vs actual prices
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
```

### Explanation:

1. **Imports and Data Loading:**
   - Import necessary libraries (`numpy`, `pandas`, `matplotlib`) and scikit-learn components (`LinearRegression`, `train_test_split`, `mean_squared_error`, `r2_score`).
   - Load the dataset using `pd.read_csv()`.

2. **Data Preparation:**
   - Select relevant features (`'sqft_living'`, `'bedrooms'`, `'bathrooms'`, `'floors'`) and the target variable (`'price'`).
   - Split the data into training and testing sets using `train_test_split()`.

3. **Model Initialization and Training:**
   - Create a linear regression model (`LinearRegression()`).
   - Train the model using `model.fit()` with training data (`X_train`, `y_train`).

4. **Prediction and Evaluation:**
   - Make predictions on the testing set using `model.predict()`.
   - Evaluate the model's performance using mean squared error (`mean_squared_error()`) and R-squared (`r2_score()`).

5. **Visualization:**
   - Plot a scatter plot to compare actual prices (`y_test`) against predicted prices (`y_pred`).

This code provides a basic framework for implementing a linear regression model to predict house prices based on selected features. Ensure to replace `'house_data.csv'` with the actual path to your dataset containing house-related features and prices.
