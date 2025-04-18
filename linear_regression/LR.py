import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the Chicago Taxi dataset
csv_file = "chicago_taxi_train.csv"
chicago_taxi_dataset = pd.read_csv(csv_file)

# Select relevant features (e.g., trip_distance as X and fare as Y)
X = chicago_taxi_dataset[['TRIP_MILES']].values
y = chicago_taxi_dataset[['FARE']].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Root Mean Squared Error
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print Model Summary
print("\nModel Summary:")
print("---------------------------")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R-squared Score (R²): {r2:.4f}")
print(f"Intercept: {model.intercept_[0]:.2f}")
print(f"Slope (Coefficient): {model.coef_[0][0]:.2f}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Trip Distance (miles)')
plt.ylabel('Fare (USD)')
plt.title('Linear Regression - Chicago Taxi Dataset')
plt.legend()
plt.show()
