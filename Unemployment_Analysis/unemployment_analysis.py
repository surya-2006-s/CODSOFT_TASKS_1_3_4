# Unemployment Analysis with Python

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Unemployment_Analysis/Unemployment in India.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Clean column names by removing extra spaces
df.columns = df.columns.str.strip()

# Remove completely empty rows
df = df.dropna(how="all")

# Convert Date column to datetime format
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Display cleaned dataset information
print("\nAfter Cleaning:")
print("Dataset Shape:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\nUpdated Column Names:")
print(df.columns.tolist())

# Average unemployment rate by region
regional_unemployment = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Unemployment Rate by Region:")
print(regional_unemployment)

# Regional Unemployment Rate Graph

plt.figure(figsize=(12, 7))

regional_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=90)

plt.tight_layout()

plt.savefig("Unemployment_Analysis/regional_unemployment.png")
plt.close()

print("\nRegional unemployment graph saved successfully!")


# Monthly Unemployment Trend

monthly_unemployment = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_index()
)

plt.figure(figsize=(12, 6))

plt.plot(
    monthly_unemployment.index,
    monthly_unemployment.values,
    marker="o"
)

plt.title("Unemployment Rate Trend in India")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("Unemployment_Analysis/unemployment_trend.png")
plt.close()

print("\nUnemployment trend graph saved successfully!")


# COVID-19 Impact Analysis

df["Year_Month"] = df["Date"].dt.to_period("M")

# Before COVID: 2019
before_covid = df[
    (df["Date"] >= "2019-05-01") &
    (df["Date"] <= "2020-02-29")
]

# During COVID: March 2020 onwards
during_covid = df[
    (df["Date"] >= "2020-03-01") &
    (df["Date"] <= "2020-06-30")
]

before_rate = before_covid["Estimated Unemployment Rate (%)"].mean()
during_rate = during_covid["Estimated Unemployment Rate (%)"].mean()

print("\nCOVID-19 Impact Analysis:")
print(f"Average unemployment rate before COVID-19: {before_rate:.2f}%")
print(f"Average unemployment rate during COVID-19: {during_rate:.2f}%")

# COVID comparison graph
plt.figure(figsize=(8, 6))

plt.bar(
    ["Before COVID-19", "During COVID-19"],
    [before_rate, during_rate]
)

plt.title("Unemployment Rate Before and During COVID-19")
plt.ylabel("Average Unemployment Rate (%)")

plt.tight_layout()

plt.savefig("Unemployment_Analysis/covid_impact.png")
plt.close()

print("\nCOVID impact graph saved successfully!")


# Rural vs Urban Unemployment Analysis

area_unemployment = (
    df.groupby("Area")["Estimated Unemployment Rate (%)"]
    .mean()
)

print("\nAverage Unemployment Rate by Area:")
print(area_unemployment)

plt.figure(figsize=(8, 6))

area_unemployment.plot(kind="bar")

plt.title("Average Unemployment Rate: Rural vs Urban")
plt.xlabel("Area")
plt.ylabel("Average Unemployment Rate (%)")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("Unemployment_Analysis/rural_vs_urban.png")
plt.close()

print("\nRural vs Urban graph saved successfully!")


# Project Summary

print("\n" + "=" * 55)
print("UNEMPLOYMENT ANALYSIS PROJECT - SUMMARY")
print("=" * 55)

print("Dataset: Unemployment in India")
print("Total records after cleaning:", len(df))
print("Analysis performed:")
print("1. Dataset exploration and cleaning")
print("2. Regional unemployment analysis")
print("3. Unemployment trend over time")
print("4. COVID-19 impact analysis")
print("5. Rural vs Urban comparison")

print("\nGenerated Graphs:")
print("1. regional_unemployment.png")
print("2. unemployment_trend.png")
print("3. covid_impact.png")
print("4. rural_vs_urban.png")

print("\nProject completed successfully!")



