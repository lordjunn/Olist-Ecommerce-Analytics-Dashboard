# Dashboard 
# ============================================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pickle
import os
from pathlib import Path

from functions import plot_top_categories

BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(page_title="E-commerce Analytics Dashboard", layout="wide")

# ============================================================================
# INPUT VALIDATION HELPERS
# ============================================================================
def validate_state_in_encoder(ohe_encoder, seller_state, customer_state):
    """
    Validate that seller and customer states are known to the OHE encoder.
    Returns (is_valid, warning_message).
    """
    warnings = []
    known_seller = set(ohe_encoder.categories_[0])
    known_customer = set(ohe_encoder.categories_[1])
    
    if seller_state not in known_seller:
        warnings.append(f"Seller state '{seller_state}' not in training data")
    if customer_state not in known_customer:
        warnings.append(f"Customer state '{customer_state}' not in training data")
    
    is_valid = len(warnings) == 0
    return is_valid, "; ".join(warnings) if warnings else None

def validate_model_features(model, features, model_name):
    """
    Validate that feature array matches expected model input shape.
    Returns (is_valid, error_message).
    """
    expected = getattr(model, 'n_features_in_', None)
    actual = features.shape[1] if len(features.shape) > 1 else len(features)
    if expected is not None and actual != expected:
        return False, f"{model_name} expects {expected} features but got {actual}"
    return True, None

def safe_predict(model, features, model_name):
    """
    Safely predict with validation. Returns (prediction, error_message).
    """
    is_valid, err = validate_model_features(model, np.array(features), model_name)
    if not is_valid:
        return None, err
    try:
        return model.predict(features), None
    except Exception as e:
        return None, str(e)

def safe_predict_proba(model, features, model_name):
    """
    Safely predict_proba with validation. Returns (probabilities, error_message).
    """
    is_valid, err = validate_model_features(model, np.array(features), model_name)
    if not is_valid:
        return None, err
    try:
        return model.predict_proba(features), None
    except Exception as e:
        return None, str(e)

# ============================================================================
# LOAD MODELS
# ============================================================================
def load_models():
    """Load all pre-trained models and encoders."""
    models_dir = BASE_DIR / 'models'
    models = {}
    
    try:
        models['delivery'] = pickle.load(open(models_dir / 'delivery_model.pkl', 'rb'))
        models['risk'] = pickle.load(open(models_dir / 'risk_model.pkl', 'rb'))
        models['satisfaction'] = pickle.load(open(models_dir / 'satisfaction_model.pkl', 'rb'))
        models['kmeans'] = pickle.load(open(models_dir / 'kmeans_model.pkl', 'rb'))
        models['ohe_states'] = pickle.load(open(models_dir / 'ohe_states.pkl', 'rb'))
        models['scaler'] = pickle.load(open(models_dir / 'scaler.pkl', 'rb'))
        
        # Optional: cluster labels
        try:
            models['cluster_labels'] = pickle.load(open(models_dir / 'cluster_labels.pkl', 'rb'))
        except:
            models['cluster_labels'] = {0: 'Segment A', 1: 'Segment B', 2: 'Segment C', 3: 'Segment D'}
        
        models['loaded'] = True
    except Exception as e:
        st.error(f"⚠️ Models not found. Please run 'train_models.py' first to generate the models.")
        st.error(f"Error: {str(e)}")
        models['loaded'] = False
    
    return models

models = load_models()

# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare the cleaned dataset."""
    df = pd.read_csv(BASE_DIR / 'olist_cleaner_dataset.csv')
    return df

df = load_data()
st.title("🛒 E-commerce Analytics Dashboard")
st.markdown("**Olist Brazilian E-commerce Analysis**")

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs([
    "📊 Executive Summary", 
    "🚚 Logistics & Risk Optimizer", 
    "👥 Customer Segments & Satisfaction"
])

with tab1:
    st.header("1. Executive Summary")
    st.markdown("**Key Performance Indicators (KPIs) aggregated by Customer State and Product Category**")
    
    # Grand totals (always shown)
    total_revenue = df['order_total_payment'].sum()
    total_orders = df['order_id'].nunique()
    avg_score = df['avg_review_score'].mean()
    late_rate = df['is_late'].mean() * 100
    satisfaction_rate = df['is_satisfied'].mean() * 100
    
    st.subheader("📈 Grand Totals")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"R$ {total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Avg Review Score", f"{avg_score:.2f} ⭐")
    col4.metric("Late Delivery Rate", f"{late_rate:.1f}%")
    col5.metric("Satisfaction Rate", f"{satisfaction_rate:.1f}%")
    
    st.divider()
    
    # Filters
    st.subheader("🔍 Filter Data")
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        state_filter = st.multiselect(
            "Filter by Customer State", 
            sorted(df['customer_state'].unique()), 
            default=[]
        )
    with col_f2:
        category_filter = st.multiselect(
            "Filter by Product Category", 
            sorted(df['product_category'].unique()), 
            default=[]
        )
    
    filtered_df = df.copy()
    if state_filter:
        filtered_df = filtered_df[filtered_df['customer_state'].isin(state_filter)]
    if category_filter:
        filtered_df = filtered_df[filtered_df['product_category'].isin(category_filter)]
    
    has_filters = bool(state_filter) or bool(category_filter)
    has_data = not filtered_df.empty
        
    # Filtered totals
    if has_filters and has_data:
        st.subheader("📊 Filtered Totals")

        total_revenue_f = filtered_df['order_total_payment'].sum()
        total_orders_f = filtered_df['order_id'].nunique()
        avg_score_f = filtered_df['avg_review_score'].mean()
        late_rate_f = filtered_df['is_late'].mean() * 100
        satisfaction_rate_f = filtered_df['is_satisfied'].mean() * 100

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Filtered Revenue", f"R$ {total_revenue_f:,.2f}")
        col2.metric("Filtered Orders", f"{total_orders_f:,}")
        col3.metric("Filtered Avg Score", f"{avg_score_f:.2f} ⭐")
        col4.metric("Filtered Late Delivery Rate", f"{late_rate_f:.1f}%")
        col5.metric("Filtered Satisfaction Rate", f"{satisfaction_rate_f:.1f}%")

        selected_states = ", ".join(state_filter) if state_filter else "All States"
        selected_categories = ", ".join(category_filter) if category_filter else "All Categories"

        st.info(f"Showing data for: **{selected_states}** | **{selected_categories}**")

    elif has_filters and not has_data:
        selected_states = ", ".join(state_filter) if state_filter else "All States"
        selected_categories = ", ".join(category_filter) if category_filter else "All Categories"

        st.info(f"No data exists for: **{selected_states}** | **{selected_categories}**.")


    if has_filters and has_data:
        st.download_button(
            "📥 Download Filtered Data",
            filtered_df.to_csv(index=False),
            "filtered_data.csv",
            mime="text/csv"
        )

    
    st.divider()
    if has_data:     
        # Revenue by State
        st.subheader("💰 Revenue by Customer State")
        state_revenue = filtered_df.groupby('customer_state')['order_total_payment'].sum().reset_index()
        state_revenue = state_revenue.sort_values('order_total_payment', ascending=False).head(10)
        fig_state = px.bar(
            state_revenue, 
            x='customer_state', 
            y='order_total_payment',
            title="Top 10 States by Revenue",
            labels={'customer_state': 'State', 'order_total_payment': 'Revenue (R$)'},
            color='order_total_payment',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_state, use_container_width=True)

        # Top Product Categories
        st.subheader("📦 Top Product Categories")
        col_m1, col_m2 = st.columns([1, 3])
        with col_m1:
            metric = st.selectbox("Select Metric", ["Orders", "Revenue"], key="metric")
            top_n = st.slider("Number of Categories", min_value=5, max_value=12, value=10, key="top_n")
        
        with col_m2:
            if metric == "Orders":
                fig = plot_top_categories(
                    filtered_df,
                    category_col='product_category',
                    top_n=top_n,
                    title=f"Top {top_n} Categories by Orders",
                    y_label="Order Count"
                )
            else:
                fig = plot_top_categories(
                    filtered_df,
                    category_col='product_category',
                    value_col='order_total_payment',
                    top_n=top_n,
                    title=f"Top {top_n} Categories by Revenue",
                    y_label="Total Revenue (R$)"
                )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("2. Logistics & Risk Optimizer")
    st.markdown("**Delivery Time Prediction & Late Delivery Risk Assessment**")
    
    # Delivery Risk Heatmap
    st.subheader("🗺️ Delivery Risk Heatmap")
    st.markdown("Shows the probability of late delivery for each seller-customer state pair.")
    
    risk_matrix = df.groupby(['seller_state', 'customer_state'])['is_late'].mean().unstack()
    top_states = df['customer_state'].value_counts().nlargest(10).index
    risk_matrix_filtered = risk_matrix.reindex(index=top_states, columns=top_states)
    
    fig_heatmap = px.imshow(
        risk_matrix_filtered,
        labels=dict(x="Customer State", y="Seller State", color="Late Probability"),
        title="Late Delivery Risk by State Pair (Top 10 States)",
        color_continuous_scale="RdYlGn_r",
        aspect="auto"
    )
    fig_heatmap.update_layout(height=500)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.divider()
    
    # Prediction Section
    st.subheader("🔮 Delivery Prediction Tool")
    st.markdown("Enter order details to predict delivery time and risk.")
    
    if not models.get('loaded', False):
        st.error("⚠️ Models not loaded. Please run 'train_models.py' first.")
    else:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            weight = st.number_input("Product Weight (g)", min_value=0.0, value=500.0, step=100.0)
            freight_value = st.number_input("Freight Value (R$)", min_value=0.0, value=20.0, step=5.0)
            estimated_days = st.number_input("Estimated Delivery Days", min_value=1, value=15, step=1)
            seller_state = st.selectbox("Seller State", sorted(df['seller_state'].unique()), key='seller')
            customer_state = st.selectbox("Customer State", sorted(df['customer_state'].unique()), key='customer')
            order_total = st.number_input("Order Total (R$)", min_value=0.0, value=100.0, step=10.0)
        
        with col_p2:
            # Validate and encode inputs using OHE
            try:
                # Validate states are known to encoder
                is_valid, warn_msg = validate_state_in_encoder(
                    models['ohe_states'], seller_state, customer_state
                )
                if not is_valid:
                    st.warning(f"⚠️ {warn_msg} — prediction may be less accurate")
                
                # Transform states using OHE encoder
                state_ohe = models['ohe_states'].transform(
                    pd.DataFrame([[seller_state, customer_state]],
                                columns=['seller_state', 'customer_state'])
                )

                # Delivery model features: [weight, freight, estimated_days] + [OHE states]
                X_numeric_delivery = np.array([[weight, freight_value, estimated_days]])
                features_delivery = np.hstack([X_numeric_delivery, state_ohe])

                # Risk model features: [weight, freight, estimated_days, order_total] + [OHE states]
                X_numeric_risk = np.array([[weight, freight_value, estimated_days, order_total]])
                features_risk = np.hstack([X_numeric_risk, state_ohe])

                # Predict delivery time with validation
                pred_days, err = safe_predict(models['delivery'], features_delivery, 'Delivery Model')
                if err:
                    st.error(f"Delivery prediction error: {err}")
                else:
                    pred_days = max(1, pred_days[0])  # Ensure positive
                    st.metric("📅 Predicted Delivery Time", f"{pred_days:.1f} days")
                
                # Predict risk with validation
                risk_proba, err = safe_predict_proba(models['risk'], features_risk, 'Risk Model')
                if err:
                    st.error(f"Risk prediction error: {err}")
                else:
                    late_prob = risk_proba[0][1] if len(risk_proba[0]) > 1 else 0
                    
                    # Color-coded risk
                    if late_prob < 0.2:
                        risk_color = "🟢"
                        risk_level = "Low Risk"
                    elif late_prob < 0.5:
                        risk_color = "🟡"
                        risk_level = "Medium Risk"
                    else:
                        risk_color = "🔴"
                        risk_level = "High Risk"
                    
                    st.metric(f"{risk_color} Late Delivery Risk", f"{late_prob*100:.1f}%")
                    st.info(f"**Risk Level:** {risk_level}")
                
                # Compare with estimated (only if delivery prediction succeeded)
                if pred_days is not None and not isinstance(pred_days, str):
                    delay = pred_days - estimated_days
                    if delay > 0:
                        st.warning(f"⚠️ Predicted to be **{delay:.1f} days late** vs. estimate")
                    else:
                        st.success(f"✅ Predicted to arrive **{abs(delay):.1f} days early**")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    st.divider()
    
    # Default values for historical data
    default_seller_state = df['seller_state'].mode()[0]  # most common seller state
    default_customer_state = df['customer_state'].mode()[0]  # most common customer state

    # Use selected values if they exist, otherwise defaults
    try:
        sel_seller_state = seller_state
    except NameError:
        sel_seller_state = default_seller_state

    try:
        sel_customer_state = customer_state
    except NameError:
        sel_customer_state = default_customer_state
    
    # Historical Data for Selected Route
    st.subheader(f"📜 Historical Orders for Selected Route: {sel_seller_state} to {sel_customer_state}")
    route_orders = df[
        (df['seller_state'] == sel_seller_state) & 
        (df['customer_state'] == sel_customer_state)
    ][['order_id', 'actual_delivery_days', 'estimated_delivery_days', 
       'delivery_diff', 'is_late', 'product_weight_g', 'order_total_payment']]
    
    if len(route_orders) > 0:
        col_h1, col_h2, col_h3 = st.columns(3)
        col_h1.metric("Total Orders", len(route_orders))
        col_h2.metric("Avg Delivery Days", f"{route_orders['actual_delivery_days'].mean():.1f}")
        col_h3.metric("Late Rate", f"{route_orders['is_late'].mean()*100:.1f}%")
        
        st.dataframe(
            route_orders.sort_values('actual_delivery_days', ascending=False).head(20),
            use_container_width=True
        )
    else:
        st.info("No historical data for this route.")
        
        
    st.subheader("📊 Feature Importance Analysis")

    if models.get('loaded', False):
        # Get feature importance from risk model
        if hasattr(models['risk'], 'feature_importances_'):
            # Feature names
            feature_names = ['Weight', 'Freight', 'Est. Days', 'Order Total'] + \
                        [f"Seller_{s}" for s in models['ohe_states'].categories_[0]] + \
                        [f"Customer_{s}" for s in models['ohe_states'].categories_[1]]
            
            importances = models['risk'].feature_importances_
            
            # Get top 15 features
            indices = np.argsort(importances)[-15:]
            
            fig_importance = px.bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                title="Top 15 Features for Late Delivery Prediction",
                labels={'x': 'Importance', 'y': 'Feature'}
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        

with tab3:
    st.header("3. Customer Segments & Satisfaction")
    st.markdown("**Customer Satisfaction Prediction & Segmentation Analysis**")
    
    # Satisfaction Prediction
    st.subheader("😊 Customer Satisfaction Predictor")
    st.markdown("Predict customer satisfaction based on order characteristics.")
    
    if not models.get('loaded', False):
        st.error("⚠️ Models not loaded. Please run 'train_models.py' first.")
    else:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            delivery_diff_input = st.slider(
                "Delivery Difference (days)", 
                min_value=-30, max_value=30, value=0,
                help="Negative = early delivery, Positive = late delivery"
            )
            order_payment_input = st.number_input("Order Total Payment (R$)", min_value=0.0, value=100.0, step=10.0)
            weight_input = st.number_input("Product Weight (g)", min_value=0.0, value=500.0, step=100.0, key='weight_sat')
            freight_input = st.number_input("Freight Value (R$)", min_value=0.0, value=20.0, step=5.0, key='freight_sat')
            is_late_input = st.selectbox("Was Delivery Late?", ["No", "Yes"])
        
        with col_s2:
            # Predict satisfaction with validation
            try:
                is_late_val = 1 if is_late_input == "Yes" else 0
                
                # Features for satisfaction model: delivery_diff, order_total_payment, product_weight_g, freight_value, is_late
                features_sat = np.array([[delivery_diff_input, order_payment_input, weight_input, freight_input, is_late_val]])
                
                # Validate and predict
                sat_pred, err = safe_predict(models['satisfaction'], features_sat, 'Satisfaction Model')
                if err:
                    st.error(f"Satisfaction prediction error: {err}")
                else:
                    sat_proba, _ = safe_predict_proba(models['satisfaction'], features_sat, 'Satisfaction Model')
                    satisfied_prob = sat_proba[0][1] if sat_proba is not None and len(sat_proba[0]) > 1 else sat_pred[0]
                    
                    if sat_pred[0] == 1:
                        st.success(f"✅ **Predicted: SATISFIED** ({satisfied_prob*100:.1f}% confidence)")
                    else:
                        st.error(f"❌ **Predicted: UNSATISFIED** ({(1-satisfied_prob)*100:.1f}% confidence)")
                
                # Show factors
                st.markdown("**Key Factors Affecting Satisfaction:**")
                if delivery_diff_input > 5:
                    st.warning(f"⚠️ Significant late delivery (+{delivery_diff_input} days) negatively impacts satisfaction")
                elif delivery_diff_input > 0:
                    st.warning(f"⚠️ Late delivery (+{delivery_diff_input} days) may hurt satisfaction")
                elif delivery_diff_input < -5:
                    st.success(f"✅ Early delivery ({delivery_diff_input} days) boosts satisfaction")
                else:
                    st.info(f"📦 On-time delivery ({delivery_diff_input} days difference)")
                
                if is_late_val == 1:
                    st.warning("⚠️ Late delivery flag is a strong negative predictor")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    
    st.divider()
    
    # Customer Segmentation (Clustering)
    st.subheader("👥 Customer Segmentation")
    st.markdown("Segments based on: **Order Value, Freight Cost, Product Weight, Review Score**")
    
    # Prepare clustering data (add order_id and product_category for details)
    cluster_features = ['order_total_payment', 'freight_value', 'product_weight_g', 'avg_review_score']
    cluster_df = df[cluster_features + ['is_satisfied', 'order_id', 'product_category']].dropna().copy()
    
    if models.get('loaded', False):
        # Apply clustering using pre-trained model
        X_cluster = cluster_df[cluster_features]
        X_cluster_scaled = models['scaler'].transform(X_cluster)
        cluster_df['cluster'] = models['kmeans'].predict(X_cluster_scaled)
        
        # Get cluster labels from saved model or generate them
        cluster_labels = models.get('cluster_labels', {0: 'Segment A', 1: 'Segment B', 2: 'Segment C', 3: 'Segment D'})
        cluster_df['segment'] = cluster_df['cluster'].map(cluster_labels)
        
        # Display cluster summary
        col_c1, col_c2 = st.columns([1, 2])
        
        with col_c1:
            st.markdown("**Segment Summary**")
            summary = cluster_df.groupby('segment').agg({
                'order_total_payment': 'mean',
                'freight_value': 'mean',
                'avg_review_score': 'mean',
                'is_satisfied': 'mean'
            }).round(2)
            summary.columns = ['Avg Payment', 'Avg Freight', 'Avg Score', 'Satisfaction %']
            summary['Satisfaction %'] = (summary['Satisfaction %'] * 100).round(1)
            summary['Count'] = cluster_df.groupby('segment').size()
            st.dataframe(summary, use_container_width=True)
        
        with col_c2:
            # Add category filter for visualization
            category_filter_vis = st.multiselect(
                "Filter by Product Category",
                options=sorted(cluster_df['product_category'].unique()),
                default=[],
                key="category_filter_vis"
            )
            
            # Filter data for visualization
            if category_filter_vis:
                vis_df = cluster_df[cluster_df['product_category'].isin(category_filter_vis)]
            else:
                vis_df = cluster_df
            
            # Add axis selection controls
            st.markdown("**Visualization Controls**")
            col_x, col_y = st.columns(2)
            with col_x:
                x_axis = st.selectbox(
                    "X-axis feature",
                    cluster_features,
                    index=0,  # Default to order_total_payment
                    key="x_axis"
                )
            with col_y:
                y_axis = st.selectbox(
                    "Y-axis feature", 
                    cluster_features,
                    index=3,  # Default to avg_review_score
                    key="y_axis"
                )
            
            # Create axis labels mapping
            axis_labels = {
                'order_total_payment': 'Order Total (R$)',
                'freight_value': 'Freight Value (R$)',
                'product_weight_g': 'Product Weight (g)',
                'avg_review_score': 'Review Score'
            }
            
            # Scatter plot with selected axes
            fig_cluster = px.scatter(
                vis_df.sample(min(5000, len(vis_df)), random_state=42),  # Sample for performance
                x=x_axis,
                y=y_axis,
                color='segment',
                title=f"Customer Segments: {axis_labels[x_axis]} vs {axis_labels[y_axis]}",
                labels={x_axis: axis_labels[x_axis], y_axis: axis_labels[y_axis]},
                opacity=0.6,
                hover_data=['product_category', 'order_id']  # Add hover to show product category and order ID
            )
            st.plotly_chart(fig_cluster, use_container_width=True)
    
        # Segment distribution pie chart
        st.subheader("📊 Segment Distribution")
        segment_counts = cluster_df['segment'].value_counts()
        fig_pie = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Order Distribution by Segment"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # New: Order Details and History
        st.subheader("🔍 Order Details & History")
        st.markdown("Select a segment and filter by product categories to view order IDs and their details.")
        
        # Filters
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            selected_segment = st.selectbox(
                "Select Segment",
                options=list(cluster_labels.values()),
                key="segment_filter"
            )
        with col_f2:
            category_filter = st.multiselect(
                "Filter by Product Category (e.g., furniture)",
                options=sorted(cluster_df['product_category'].unique()),
                default=[],
                key="category_filter"
            )
        
        # Filter data based on selections
        filtered_cluster_df = cluster_df[cluster_df['segment'] == selected_segment]
        if category_filter:
            filtered_cluster_df = filtered_cluster_df[filtered_cluster_df['product_category'].isin(category_filter)]
        
        if not filtered_cluster_df.empty:
            # Display order details (limit to top 50 for brevity)
            order_history_df = filtered_cluster_df[['order_id', 'product_category', 'order_total_payment', 'avg_review_score', 'segment']]
            order_history_rows = len(order_history_df)  # Count of orders before adding total row
            
            # Add total row
            total_row = pd.DataFrame({
                'order_id': ['Total'],
                'product_category': [str(category_filter) if category_filter else 'All'],
                'avg_review_score': [order_history_df['avg_review_score'].mean()],
                'order_total_payment': [order_history_df['order_total_payment'].sum()],
                'segment': [selected_segment]
            })
            order_history_df = pd.concat([order_history_df, total_row], ignore_index=True)
            
            st.dataframe(order_history_df, use_container_width=True)
            st.info(f"Showing {order_history_rows} orders and totals in '{selected_segment}' segment.")
        else:
            st.info("No data matches the selected filters.")
        
        # Insights
        st.subheader("💡 Key Insights")
        
        # Find high-risk and premium segments
        at_risk_segments = [k for k, v in cluster_labels.items() if 'At-Risk' in v]
        premium_segments = [k for k, v in cluster_labels.items() if 'Premium' in v]
        
        if at_risk_segments:
            at_risk_count = cluster_df[cluster_df['cluster'].isin(at_risk_segments)].shape[0]
            at_risk_pct = at_risk_count / len(cluster_df) * 100
            st.markdown(f"- **At-Risk Segments:** {at_risk_count:,} orders ({at_risk_pct:.1f}%) - These customers have lower satisfaction scores")
        
        if premium_segments:
            premium_count = cluster_df[cluster_df['cluster'].isin(premium_segments)].shape[0]
            premium_pct = premium_count / len(cluster_df) * 100
            st.markdown(f"- **Premium Segments:** {premium_count:,} orders ({premium_pct:.1f}%) - High-value loyal customers")
        
        st.markdown("""
        - **Recommendation:** Focus retention efforts on At-Risk segments with targeted promotions
        - **Upsell Opportunity:** Budget Satisfied customers may respond to premium product recommendations
        """)
    else:
        st.warning("⚠️ Models not loaded. Please run 'train_models.py' to enable clustering analysis.")

