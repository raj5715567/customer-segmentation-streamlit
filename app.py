import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Segmentation", layout="wide")

st.title("🛍️ Customer Segmentation Dashboard")

# Load dataset
df = pd.read_csv("data/Customer Personality Analysis Dataset.csv")

# Clean data
df = df.dropna()

# Convert columns
df['Year_Birth'] = pd.to_numeric(df['Year_Birth'], errors='coerce')
df['Income'] = pd.to_numeric(df['Income'], errors='coerce')

df = df.dropna(subset=['Year_Birth', 'Income'])

# Feature Engineering
df['Age'] = 2024 - df['Year_Birth']

df['Total_Spending'] = (
    df['MntWines'] +
    df['MntFruits'] +
    df['MntMeatProducts'] +
    df['MntFishProducts'] +
    df['MntSweetProducts'] +
    df['MntGoldProds']
)

df['CLV'] = df['Total_Spending'] * df['NumWebPurchases']

# Remove outliers (important)
df = df[df['Income'] < 200000]

# Sidebar controls
st.sidebar.header("Settings")
k = st.sidebar.slider("Number of Clusters", 2, 6, 4)

# Features
X = df[['Income', 'Age', 'Total_Spending']]

# Model
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Visualization
st.subheader("📊 Customer Segments")

fig, ax = plt.subplots()
scatter = ax.scatter(df['Income'], df['Total_Spending'], c=df['Cluster'])

ax.set_xlabel("Income")
ax.set_ylabel("Total Spending")

st.pyplot(fig)

# Cluster Summary
st.subheader("📈 Cluster Analysis")

summary = df.groupby('Cluster')[['Income', 'Age', 'Total_Spending']].mean()
st.dataframe(summary)

# Insights
st.subheader("💡 Insights")

st.write("""
- Customers are grouped based on income and spending behavior.
- High income groups tend to spend more.
- Some clusters show low spending despite high income.
- Businesses can target each segment differently.
""")