"""
This script trains all machine learning models for the Olist E-commerce Analytics Dashboard.

Models Trained:
1. Delivery Time Prediction (RandomForestRegressor)
2. Delivery Risk Prediction (RandomForestClassifier)
3. Customer Satisfaction Prediction (GradientBoostingClassifier)
4. Customer Segmentation (KMeans Clustering)

"""

import pickle
import pandas as pd
import numpy as np
import os
import shutil
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                            accuracy_score, classification_report, confusion_matrix)

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================
def is_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False

IN_COLAB = is_colab()

if IN_COLAB:
    print("Running in Google Colab")
    MODELS_DIR = '/content/models'
else:
    print("Running in local environment")
    MODELS_DIR = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)
print(f"Models will be saved to: {MODELS_DIR}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)
df = pd.read_csv('olist_cleaner_dataset.csv')
print(f"Dataset loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# CREATE SHARED ONE-HOT ENCODER FOR STATES
# ============================================================================
print("\n" + "="*60)
print("CREATING ONE-HOT ENCODER FOR STATES")
print("="*60)

# Fit OHE on all unique states (handles unknown states gracefully)
ohe_states = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
all_states_df = df[['seller_state', 'customer_state']].dropna()
ohe_states.fit(all_states_df)

n_seller_states = len(ohe_states.categories_[0])
n_customer_states = len(ohe_states.categories_[1])
n_ohe_features = n_seller_states + n_customer_states
print(f"Seller states: {n_seller_states}, Customer states: {n_customer_states}")
print(f"Total OHE features: {n_ohe_features}")

# Save OHE encoder
pickle.dump(ohe_states, open(f'{MODELS_DIR}/ohe_states.pkl', 'wb'))
print(f"✓ OHE encoder saved to {MODELS_DIR}/ohe_states.pkl")

# ============================================================================
# MODEL 1: DELIVERY TIME PREDICTION (RandomForestRegressor)
# ============================================================================
print("\n" + "="*60)
print("MODEL 1: DELIVERY TIME PREDICTION")
print("Algorithm: RandomForestRegressor")
print("="*60)

# Feature engineering
features_delivery = ['product_weight_g', 'freight_value', 'estimated_delivery_days', 
                     'seller_state', 'customer_state']
target_delivery = 'actual_delivery_days'

delivery_df = df[features_delivery + [target_delivery]].dropna()
print(f"Training samples: {len(delivery_df):,}")

# Encode states using OHE
state_features = ohe_states.transform(delivery_df[['seller_state', 'customer_state']])

# Combine numeric features with OHE state features
# Feature order: [weight, freight, estimated_days] + [OHE states]
X_numeric = delivery_df[['product_weight_g', 'freight_value', 'estimated_delivery_days']].values
X_delivery = np.hstack([X_numeric, state_features])
y_delivery = delivery_df[target_delivery]

print(f"Feature shape: {X_delivery.shape} (3 numeric + {n_ohe_features} OHE)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_delivery, y_delivery, 
                                                     test_size=0.2, random_state=42)

# Train model
delivery_model = RandomForestRegressor(n_estimators=200, max_depth=10, 
                                        random_state=42, n_jobs=-1)
delivery_model.fit(X_train, y_train)

# Evaluate
y_pred = delivery_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"  MAE:  {mae:.2f} days")
print(f"  RMSE: {rmse:.2f} days")
print(f"  R²:   {r2:.4f}")
print(f"\nTarget Statistics:")
print(f"  Mean: {y_delivery.mean():.2f} days")
print(f"  Std:  {y_delivery.std():.2f} days")
print(f"  Range: {y_delivery.min():.0f} - {y_delivery.max():.0f} days")

# Save model
pickle.dump(delivery_model, open(f'{MODELS_DIR}/delivery_model.pkl', 'wb'))
print(f"✓ Delivery model saved (expects {delivery_model.n_features_in_} features)")

# ============================================================================
# MODEL 2: DELIVERY RISK PREDICTION (RandomForestClassifier)
# ============================================================================
print("\n" + "="*60)
print("MODEL 2: DELIVERY RISK PREDICTION")
print("Algorithm: RandomForestClassifier")
print("="*60)

features_risk = ['product_weight_g', 'freight_value', 'estimated_delivery_days',
                 'order_total_payment', 'seller_state', 'customer_state']
target_risk = 'is_late'

risk_df = df[features_risk + [target_risk]].dropna()
print(f"Training samples: {len(risk_df):,}")
print(f"Class distribution: {risk_df[target_risk].value_counts().to_dict()}")

# Encode states using OHE
state_features_risk = ohe_states.transform(risk_df[['seller_state', 'customer_state']])

# Combine numeric features with OHE state features
# Feature order: [weight, freight, estimated_days, order_total] + [OHE states]
X_numeric_risk = risk_df[['product_weight_g', 'freight_value', 'estimated_delivery_days', 'order_total_payment']].values
X_risk = np.hstack([X_numeric_risk, state_features_risk])
y_risk = risk_df[target_risk]

print(f"Feature shape: {X_risk.shape} (4 numeric + {n_ohe_features} OHE)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_risk, y_risk, 
                                                     test_size=0.2, random_state=42)

# Train model with class balancing
risk_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     class_weight='balanced', random_state=42, n_jobs=-1)
risk_model.fit(X_train, y_train)

# Evaluate
y_pred = risk_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['On-Time', 'Late']))

# Save model
pickle.dump(risk_model, open(f'{MODELS_DIR}/risk_model.pkl', 'wb'))
print(f"✓ Risk model saved (expects {risk_model.n_features_in_} features)")

# ============================================================================
# MODEL 3: CUSTOMER SATISFACTION PREDICTION (GradientBoostingClassifier)
# ============================================================================
print("\n" + "="*60)
print("MODEL 3: CUSTOMER SATISFACTION PREDICTION")
print("Algorithm: GradientBoostingClassifier")
print("="*60)

features_sat = ['delivery_diff', 'order_total_payment', 'product_weight_g',
                'freight_value', 'is_late']
target_sat = 'is_satisfied'

sat_df = df[features_sat + [target_sat]].dropna()
print(f"Training samples: {len(sat_df):,}")
print(f"Class distribution: {sat_df[target_sat].value_counts().to_dict()}")

X_sat = sat_df[features_sat]
y_sat = sat_df[target_sat]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sat, y_sat, 
                                                     test_size=0.2, random_state=42)

# Train model
satisfaction_model = GradientBoostingClassifier(n_estimators=100, max_depth=5,
                                                 learning_rate=0.1, random_state=42)
satisfaction_model.fit(X_train, y_train)

# Evaluate
y_pred = satisfaction_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nPerformance Metrics:")
print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Unsatisfied', 'Satisfied']))

# Feature importance
print(f"\nFeature Importance:")
for feat, imp in sorted(zip(features_sat, satisfaction_model.feature_importances_), 
                        key=lambda x: x[1], reverse=True):
    print(f"  {feat}: {imp:.4f}")

# Save model
pickle.dump(satisfaction_model, open(f'{MODELS_DIR}/satisfaction_model.pkl', 'wb'))
print(f"✓ Satisfaction model saved to {MODELS_DIR}/satisfaction_model.pkl")

# ============================================================================
# MODEL 4: CUSTOMER SEGMENTATION (KMeans Clustering)
# ============================================================================
print("\n" + "="*60)
print("MODEL 4: CUSTOMER SEGMENTATION")
print("Algorithm: KMeans Clustering (k=4)")
print("="*60)

features_cluster = ['order_total_payment', 'freight_value', 'product_weight_g', 'avg_review_score']

cluster_df = df[features_cluster].dropna()
print(f"Clustering samples: {len(cluster_df):,}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(cluster_df)

# Train KMeans with k=4
kmeans_model = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_model.fit(X_scaled)

# Analyze clusters
cluster_df['cluster'] = kmeans_model.labels_
print(f"\nCluster Distribution:")
print(cluster_df['cluster'].value_counts().sort_index())

print(f"\nCluster Profiles (Mean Values):")
cluster_profiles = cluster_df.groupby('cluster')[features_cluster].mean()
print(cluster_profiles.round(2))

# Assign business labels based on characteristics
cluster_labels = {}
for cluster_id in range(4):
    profile = cluster_profiles.loc[cluster_id]
    if profile['order_total_payment'] > cluster_profiles['order_total_payment'].median():
        if profile['avg_review_score'] > cluster_profiles['avg_review_score'].median():
            cluster_labels[cluster_id] = 'Premium Loyal'
        else:
            cluster_labels[cluster_id] = 'High-Value At-Risk'
    else:
        if profile['avg_review_score'] > cluster_profiles['avg_review_score'].median():
            cluster_labels[cluster_id] = 'Budget Satisfied'
        else:
            cluster_labels[cluster_id] = 'Budget At-Risk'

print(f"\nCluster Labels:")
for k, v in cluster_labels.items():
    print(f"  Cluster {k}: {v}")

# Save model, scaler, and labels
pickle.dump(kmeans_model, open(f'{MODELS_DIR}/kmeans_model.pkl', 'wb'))
pickle.dump(scaler, open(f'{MODELS_DIR}/scaler.pkl', 'wb'))
pickle.dump(cluster_labels, open(f'{MODELS_DIR}/cluster_labels.pkl', 'wb'))
print(f"✓ KMeans model saved to {MODELS_DIR}/kmeans_model.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*60)
print("TRAINING COMPLETE - SUMMARY")
print("="*60)
print("""
Models Saved:
  1. delivery_model.pkl     - Delivery Time Prediction (RandomForestRegressor)
  2. risk_model.pkl         - Delivery Risk Prediction (RandomForestClassifier)
  3. satisfaction_model.pkl - Customer Satisfaction (GradientBoostingClassifier)
  4. kmeans_model.pkl       - Customer Segmentation (KMeans, k=4)

Supporting Files:
  - ohe_states.pkl          - OneHotEncoder for seller/customer states
  - scaler.pkl              - Feature scaler for clustering
  - cluster_labels.pkl      - Business labels for clusters

All models are ready for use in the dashboard!
""")
print("="*60)

# -------------------------------
# Zip models if in Colab
# -------------------------------
if IN_COLAB:
    zip_path = '/content/models.zip'
    shutil.make_archive(base_name=zip_path.replace('.zip',''), format='zip', root_dir=MODELS_DIR)
    print(f"Models folder zipped at: {zip_path}")