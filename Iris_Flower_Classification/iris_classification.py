# Iris Flower Classification

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target/species column
df["species"] = iris.target

# Display the first 5 rows
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


# Pairplot to visualize relationships between features
pairplot = sns.pairplot(df, hue="species")
pairplot.fig.suptitle("Iris Dataset - Feature Relationships", y=1.02)
pairplot.savefig("iris_pairplot.png")
plt.close()

# Box plots to identify distribution and outliers
plt.figure(figsize=(10, 6))

sns.boxplot(data=df.drop(columns=["species"]))

plt.title("Box Plot of Iris Features")
plt.xlabel("Features")
plt.ylabel("Measurement (cm)")

plt.savefig("iris_boxplot.png")
plt.close()

# Separate features and target
X = df.drop(columns=["species"])
y = df["species"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Train Logistic Regression model
from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=200)

logistic_model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully!")

# Train K-Nearest Neighbors (KNN) model
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

print("\nKNN model trained successfully!")


# Make predictions on the test data
logistic_predictions = logistic_model.predict(X_test)
knn_predictions = knn_model.predict(X_test)

print("\nPredictions completed successfully!")

# Evaluate model accuracy
from sklearn.metrics import accuracy_score

logistic_accuracy = accuracy_score(y_test, logistic_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nModel Accuracy:")
print("Logistic Regression:", logistic_accuracy)
print("KNN:", knn_accuracy)



# Confusion matrices
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Logistic Regression confusion matrix
logistic_cm = confusion_matrix(y_test, logistic_predictions)
print("\nLogistic Regression Confusion Matrix:")
print(logistic_cm)

# KNN confusion matrix
knn_cm = confusion_matrix(y_test, knn_predictions)
print("\nKNN Confusion Matrix:")
print(knn_cm)

# Save Logistic Regression confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, logistic_predictions)
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("logistic_confusion_matrix.png")
plt.close()

# Save KNN confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, knn_predictions)
plt.title("KNN - Confusion Matrix")
plt.savefig("knn_confusion_matrix.png")
plt.close()

# Classification reports
from sklearn.metrics import classification_report

print("\nLogistic Regression Classification Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=iris.target_names
))

print("\nKNN Classification Report:")
print(classification_report(
    y_test,
    knn_predictions,
    target_names=iris.target_names
))

# Compare model performance
print("\nModel Comparison:")
print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")

if knn_accuracy > logistic_accuracy:
    print("KNN achieved higher accuracy on the test set.")
elif logistic_accuracy > knn_accuracy:
    print("Logistic Regression achieved higher accuracy on the test set.")
else:
    print("Both models achieved the same accuracy on the test set.")



# Feature selection discussion
print("\nFeature Selection:")
print("The Iris dataset contains four features:")
print("1. Sepal length")
print("2. Sepal width")
print("3. Petal length")
print("4. Petal width")

print("\nPetal length and petal width are particularly useful")
print("because they show clear differences between the Iris species.")
print("All four features are used for model training.")


# Project Summary
print("\n" + "=" * 50)
print("IRIS FLOWER CLASSIFICATION - PROJECT SUMMARY")
print("=" * 50)

print("Dataset: Iris dataset from scikit-learn")
print("Total samples: 150")
print("Features: 4")
print("Classes: Setosa, Versicolor, Virginica")
print("Training samples: 120")
print("Testing samples: 30")
print(f"Logistic Regression Accuracy: {logistic_accuracy * 100:.2f}%")
print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
print("Project completed successfully!")


