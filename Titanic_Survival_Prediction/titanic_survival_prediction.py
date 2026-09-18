# CODSOFT Task 1 - Titanic Survival Prediction

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. Load the dataset
df = pd.read_csv("Titanic_Survival_Prediction/train.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())


# 2. Select useful columns
df = df[
    [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]
].copy()


# 3. Handle missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# 4. Convert categorical columns to numbers
df = pd.get_dummies(
    df,
    columns=["Sex", "Embarked"],
    drop_first=True
)


print("\nDataset after preprocessing:")
print(df.head())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# 5. Separate features and target
X = df.drop(columns=["Survived"])
y = df["Survived"]


# 6. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 7. Train Logistic Regression model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")


# 8. Make predictions
y_pred = model.predict(X_test)


# 9. Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(f"{accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 10. Confusion Matrix Visualization
plt.figure(figsize=(7, 5))

sns.heatmap(
    confusion_matrix(y_test, y_pred),
    annot=True,
    fmt="d"
)

plt.title("Titanic Survival - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "Titanic_Survival_Prediction/confusion_matrix.png"
)

plt.close()

print("\nConfusion matrix graph saved successfully!")


# 11. Survival Distribution Visualization
plt.figure(figsize=(7, 5))

sns.countplot(data=df, x="Survived")

plt.title("Titanic Survival Distribution")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig(
    "Titanic_Survival_Prediction/survival_distribution.png"
)

plt.close()

print("Survival distribution graph saved successfully!")


# 12. Project Summary
print("\n" + "=" * 50)
print("TITANIC SURVIVAL PREDICTION - SUMMARY")
print("=" * 50)

print("Dataset: Titanic Passenger Dataset")
print("Machine Learning Model: Logistic Regression")
print("Train-Test Split: 80% Training, 20% Testing")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nProject completed successfully!")