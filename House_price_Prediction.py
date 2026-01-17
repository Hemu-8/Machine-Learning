Area=[1200, 1500, 1800, 2400, 3000, 2000, 2800]
Bedrooms=[2, 3, 3, 4, 5, 3, 4]
Age=[10, 8, 5, 2, 1, 4, 3]
Price=[300000, 4200000, 5000000, 6500000, 8200000, 5400000, 7500000]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create DataFrame
data = pd.DataFrame({
    'Area': Area,
    'Bedrooms': Bedrooms,
    'Age': Age,
    'Price': Price
})

# Features (X) and Target (y)
X = data[['Area', 'Bedrooms', 'Age']]
y = data['Price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Visualization
plt.plot(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Prediction for new house
new_house = np.array([[500000, 3, 4]])  # Area, Bedrooms, Age
predicted_price = model.predict(new_house)
print("Predicted Price for new house:", predicted_price[0])
