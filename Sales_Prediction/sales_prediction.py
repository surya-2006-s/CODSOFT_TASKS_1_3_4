# CODSOFT Task 4 - Sales Prediction Using Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 1. Load Dataset
df = pd.read_csv("Sales_Prediction/Advertising.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# 2. Data Visualization - Correlation
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Advertising Dataset Correlation")
plt.tight_layout()
plt.savefig("Sales_Prediction/correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved successfully!")


# 3. Define Features and Target
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]


# 4. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 5. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")


# 6. Make Predictions
y_pred = model.predict(X_test)


# 7. Evaluate Model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"MAE      : {mae:.2f}")
print(f"MSE      : {mse:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"R2 Score : {r2:.2f}")


# 8. Actual vs Predicted Sales
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.tight_layout()
plt.savefig("Sales_Prediction/actual_vs_predicted.png")
plt.close()

print("\nActual vs predicted graph saved successfully!")


# 9. TV Advertising vs Sales
plt.figure(figsize=(7, 5))
plt.scatter(df["TV"], df["Sales"])

plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("TV Advertising vs Sales")

plt.tight_layout()
plt.savefig("Sales_Prediction/tv_vs_sales.png")
plt.close()

print("TV vs Sales graph saved successfully!")


# 10. Example Prediction
example = pd.DataFrame({
    "TV": [150],
    "Radio": [30],
    "Newspaper": [20]
})

predicted_sales = model.predict(example)

print("\nExample Prediction:")
print("TV Advertising   : 150")
print("Radio Advertising: 30")
print("Newspaper        : 20")
print(f"Predicted Sales  : {predicted_sales[0]:.2f}")


# 11. Final Summary
print("\n" + "=" * 50)
print("SALES PREDICTION - SUMMARY")
print("=" * 50)

print("Dataset: Advertising Dataset")
print("Model: Linear Regression")
print("Features: TV, Radio, Newspaper")
print("Target: Sales")
print("Train-Test Split: 80% Training, 20% Testing")
print(f"R2 Score: {r2:.2f}")

print("\nProject completed successfully!")