import pandas as pd 
df=pd.read_csv("store_customers.csv")
print(df.head())
print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nData Information:")
print(df.info())

# Fill missing values

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Annual Income (k$)"] = df["Annual Income (k$)"].fillna(
    df["Annual Income (k$)"].median()
)

df["Spending Score (1-100)"] = df["Spending Score (1-100)"].fillna(
    df["Spending Score (1-100)"].median()
)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

from sklearn.cluster import KMeans

# Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Create K-Means model with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)

# Assign each customer to a cluster
df["Cluster"] = kmeans.fit_predict(X)

print("\nCustomer Segments:")
print(df.head(10))

print("\nNumber of Customers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

# Analyze each customer segment

segment_analysis = df.groupby("Cluster").agg({
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean",
    "CustomerID": "count"
}).rename(columns={
    "CustomerID": "Customer Count"
})

print("\nSegment Analysis:")
print(segment_analysis.round(2))

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation")

plt.show()

# Give meaningful names to customer clusters

segment_names = {
    0: "Regular Customers",
    1: "High Income - Low Spending",
    2: "Potential Customers",
    3: "Young High-Spending Customers",
    4: "High Income - At-Risk Customers"
}

df["Customer Segment"] = df["Cluster"].map(segment_names)

# Save final dataset
df.to_csv("customer_segments.csv", index=False)

print("\nFinal Customer Segments:")
print(df[["CustomerID", "Customer Segment"]].head(10))

print("\nFinal dataset saved as customer_segments.csv")