# Car Price Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load the dataset
df = pd.read_csv("Car_Price_Prediction/car data.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Basic information about the dataset
print("\nDataset Shape:")
print(df.shape)

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDescriptive Statistics:")
print(df.describe())

# Check unique values in categorical columns
print("\nFuel Types:")
print(df["Fuel_Type"].unique())

print("\nSeller Types:")
print(df["Seller_Type"].unique())

print("\nTransmission Types:")
print(df["Transmission"].unique())

print("\nNumber of Owners:")
print(df["Owner"].unique())

# Remove unnecessary column
df = df.drop(columns=["Car_Name"])

print("\nColumns after removing Car_Name:")
print(df.columns)


# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nDataset after encoding:")
print(df.head())

# Separate features and target
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Train Linear Regression model
model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

# Make predictions on the test data
y_pred = model.predict(X_test)

print("\nPredictions completed successfully!")
print("First 5 predicted prices:")
print(y_pred[:5])


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Actual vs Predicted Prices
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

# Reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.savefig("actual_vs_predicted.png")
plt.close()

print("\nActual vs Predicted graph saved successfully!")

# Correlation Heatmap
plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Correlation Heatmap - Car Price Dataset")

plt.savefig("Car_Price_Prediction/correlation_heatmap.png")
plt.close()

print("\nCorrelation heatmap saved successfully!")

# Project Summary
print("\n" + "=" * 50)
print("CAR PRICE PREDICTION PROJECT - SUMMARY")
print("=" * 50)

print("Dataset: CarDekho Used Car Dataset")
print("Total records:", len(df))
print("Machine Learning Model: Linear Regression")
print("Train-Test Split: 80% Training, 20% Testing")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

print("\nProject completed successfully!")