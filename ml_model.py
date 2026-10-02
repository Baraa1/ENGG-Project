import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load the cleaned dataset
# Ensure 'cleaned_climate_merged.csv' is in the same directory as this script
df = pd.read_csv('cleaned_climate_merged.csv')

# 2. Define Features (X) and Target (y)
X = df[['% GHG emitting source', '% GHG non emitting source']]
y = df['PM2.5 Exposure']

# 3. Split the dataset into Training and Testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of testing samples: {len(X_test)}\n")

# 4. Initialize and Train the Random Forest model
print("Training the model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predict on the test set
y_pred = model.predict(X_test)

# 6. Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-" * 30)
print("Model Evaluation Results:")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f} µg/m³")
print(f"MAE (Mean Absolute Error): {mae:.2f} µg/m³")
print(f"R-squared (R²): {r2:.4f}")
print("-" * 30)

# 7. Visualize the results: Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Ideal prediction diagonal line
plt.xlabel('Actual PM2.5 Exposure')
plt.ylabel('Predicted PM2.5 Exposure')
plt.title('Actual vs Predicted PM2.5 (Random Forest)')
plt.grid(True)
plt.show()

# 8. View Feature Importances
importances = model.feature_importances_
print(f"\nFeature Importances:")
print(f"- GHG emitting source: {importances[0]*100:.2f}%")
print(f"- GHG non emitting source: {importances[1]*100:.2f}%")