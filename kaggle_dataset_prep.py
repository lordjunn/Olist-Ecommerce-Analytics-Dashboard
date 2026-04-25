# 0. Imports
import os
import json
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

# 1. Setup Kaggle API credentials (from environment variables)
kaggle_username = os.getenv("KAGGLE_USERNAME")
kaggle_key = os.getenv("KAGGLE_KEY")

if not kaggle_username or not kaggle_key:
    raise EnvironmentError(
        "Missing Kaggle credentials. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
    )

kaggle_credentials = {
    "username": kaggle_username,
    "key": kaggle_key
}

os.makedirs("/root/.kaggle", exist_ok=True)
with open("/root/.kaggle/kaggle.json", "w") as f:
    json.dump(kaggle_credentials, f)
os.chmod("/root/.kaggle/kaggle.json", 0o600)

# 2. Download & unzip dataset
os.makedirs("./olist_data", exist_ok=True)
subprocess.run([
    "kaggle", "datasets", "download", "-d", "olistbr/brazilian-ecommerce",
    "-p", "./olist_data", "--unzip"
], check=True)

DATA_DIR = "./olist_data"

df_customer     = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
df_geolocation  = pd.read_csv(f"{DATA_DIR}/olist_geolocation_dataset.csv")
df_order_item   = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv")
df_order_payment= pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv")
df_order_review = pd.read_csv(f"{DATA_DIR}/olist_order_reviews_dataset.csv")
df_orders       = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv")
df_products     = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
df_sellers      = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
df_translation  = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")

# display shape of dataset
print("Customer Dataset Shape:", df_customer.shape)
print("Orders Geolocation Shape:", df_geolocation.shape)
print("Order Items Dataset Shape:", df_order_item.shape)
print("Order Payments Dataset Shape:", df_order_payment.shape)
print("Order Reviews Dataset Shape:", df_order_review.shape)
print("Order Dataset Shape:", df_orders.shape)
print("Product Dataset Shape:", df_products.shape)
print("Sellers Dataset Shape:", df_sellers.shape)
print("Category Translation Dataset Shape:", df_translation.shape)

datasets = {
    "1. Customer Dataset": df_customer,
    "2. Geolocation Dataset": df_geolocation,
    "3. Order Items Dataset": df_order_item,
    "4. Order Payments Dataset": df_order_payment,
    "5. Order Reviews Dataset": df_order_review,
    "6. Order Dataset": df_orders,
    "7. Product Dataset": df_products,
    "8. Sellers Dataset": df_sellers,
    "9. Category Translation": df_translation
}

# loop all the unique values
for name, df in datasets.items():
    print(f"===== {name} =====")
    print(df.nunique())
    print("\n" + "-"*30 + "\n")

# preview datasets
print("\n=== Customer Dataset ===")
print(df_customer.head())
print("\n=== Orders Geolocation Dataset ===")
print(df_geolocation.head())
print("\n=== Orders Items Dataset ===")
print(df_order_item.head())
print("\n=== Orders Payments Dataset ===")
print(df_order_payment.head())
print("\n=== Orders Reviews Dataset ===")
print(df_order_review.head())
print("\n=== Orders Dataset ===")
print(df_orders.head())
print("\n=== Product Dataset ===")
print(df_products.head())
print("\n=== Sellers Dataset ===")
print(df_sellers.head())
print("\n=== Category Translation Dataset ===")
print(df_translation.head())

geo_data = df_geolocation.groupby('geolocation_zip_code_prefix').first().reset_index()

# keep only necessary columns to avoid clutter
geo_data = geo_data[['geolocation_zip_code_prefix', 'geolocation_lat', 'geolocation_lng']]
geo_data.rename(columns={'geolocation_zip_code_prefix': 'zip_code_prefix'}, inplace=True)

# aggreagation
# calculate total payment value per order
payment_agg = df_order_payment.groupby('order_id').agg({
    'payment_value': 'sum',
    'payment_installments': 'max'
}).reset_index()
payment_agg.rename(columns={'payment_value': 'order_total_payment', 'payment_installments': 'max_installments'}, inplace=True)

# calculate average review score per order
review_agg = df_order_review.groupby('order_id').agg({
    'review_score': 'mean'
}).reset_index()
review_agg.rename(columns={'review_score': 'avg_review_score'}, inplace=True)

# merge
# Start with Orders  --> Join Items -> Join Customers -> Join Aggregates
df_merged = pd.merge(df_orders, df_order_item, on='order_id', how='left')
df_merged = pd.merge(df_merged, df_customer, on='customer_id', how='left')
df_merged = pd.merge(df_merged, payment_agg, on='order_id', how='left')
df_merged = pd.merge(df_merged, review_agg, on='order_id', how='left')


df_merged = pd.merge(df_merged, df_products, on='product_id', how='left')
df_merged = pd.merge(df_merged, df_translation, on='product_category_name', how='left')
df_merged = pd.merge(df_merged, df_sellers, on='seller_id', how='left')

# for typo lenght-> length
df_merged.rename(columns={
    'product_name_lenght': 'product_name_length',
    'product_description_lenght': 'product_description_length'
}, inplace=True)

# drop since we dont need portuguese version
df_merged.drop(columns=['product_category_name'], inplace=True) 

# Merge Customer Location
df_merged = pd.merge(df_merged, geo_data, 
                     left_on='customer_zip_code_prefix', 
                     right_on='zip_code_prefix', 
                     how='left')
df_merged.rename(columns={'geolocation_lat': 'customer_lat', 'geolocation_lng': 'customer_lng'}, inplace=True)
df_merged.drop(columns='zip_code_prefix', inplace=True) # Clean up join key

# Merge Seller Location
df_merged = pd.merge(df_merged, geo_data, 
                     left_on='seller_zip_code_prefix', 
                     right_on='zip_code_prefix', 
                     how='left')
df_merged.rename(columns={'geolocation_lat': 'seller_lat', 'geolocation_lng': 'seller_lng'}, inplace=True)
df_merged.drop(columns='zip_code_prefix', inplace=True) # Clean up join key


# final check
print(f"Original Orders Shape: {df_orders.shape}")
print(f"Merged Dataset Shape: {df_merged.shape}")
df_merged.head()

# Save to CSV
df_merged.to_csv('olist_merged_dataset.csv', index=False)

print("File saved successfully as 'olist_merged_dataset.csv'")

# read merged dataset file
df = pd.read_csv('olist_merged_dataset.csv')

print("=== 1. DATASET OVERVIEW ===")
print(f"Shape: {df_merged.shape}")
print("\nData Types:")
print(df_merged.dtypes.value_counts())

print("\n=== 2. MISSING VALUES ANALYSIS ===")
# calculate percentage of missing values
missing_percent = df_merged.isnull().sum() / len(df_merged) * 100
missing_percent = missing_percent[missing_percent > 0].sort_values(ascending=False)
print("Columns with missing values (%):")
print(missing_percent)

print("=== 3. SKEWNESS ANALYSIS ===")
numeric_cols = df_merged.select_dtypes(include=['number'])
skewness = numeric_cols.skew().sort_values(ascending=False)
print(skewness)

print("\n=== 4. DUPLICATE CHECK ===")
# check for fully identical rows
duplicates = df_merged.duplicated().sum()
print(f"Total Duplicate Rows: {duplicates}")

#check the id as well if key_dup not equall to 0 need to add more step to solve
key_duplicates = df_merged.duplicated(subset=['order_id', 'order_item_id']).sum()
print(f"Duplicates in Primary Key (order_id + order_item_id): {key_duplicates}")

print("\n=== 5. DESCRIPTIVE STATISTICS (Numerical) ===")
# show stats for key numerical columns
num_cols = df_merged.select_dtypes(include=['number']).columns.tolist()
print(df_merged[num_cols].describe().round(2))


print("\n=== 6. DESCRIPTIVE STATISTICS (Categorical) ===")
# show stats for key categorical columns
cat_cols = df_merged.select_dtypes(include=['object']).columns.tolist()
print(df_merged[cat_cols].describe())


# --- VISUALIZATIONS ---
# histogram and boxplot for all datatype

for col in num_cols:
    # Create a figure with 2 subplots (one for Hist, one for Box)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Histogram (Distribution)
    sns.histplot(data=df_merged, x=col, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Distribution of {col}')
    
    # Plot 2: Boxplot (Outliers)
    sns.boxplot(data=df_merged, x=col, ax=axes[1], color='orange')
    axes[1].set_title(f'Boxplot of {col}')
    
    plt.tight_layout()
    plt.show()

for col in cat_cols:
    plt.figure(figsize=(12, 5))
    
    unique_count = df_merged[col].nunique()
    
    if unique_count > 10: # for huge value
        top_10 = df_merged[col].value_counts().nlargest(10).index
        sns.countplot(data=df_merged[df_merged[col].isin(top_10)], x=col, order=top_10, palette='viridis')
        plt.title(f'Top 10 Categories in {col} (out of {unique_count})')
    else:
        sns.countplot(data=df_merged, x=col, order=df_merged[col].value_counts().index, palette='viridis')
        plt.title(f'Distribution of {col}')

    plt.xticks(rotation=45, ha='right')  
    plt.tight_layout()
    plt.show()
    

plt.figure(figsize=(10, 8))
# Select only numeric columns for correlation
numeric_df = df_merged.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()