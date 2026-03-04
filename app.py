# app.py
import sqlite3
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --- Load database ---
conn = sqlite3.connect("restaurant_db.db")
master_table_encoded = pd.read_sql("SELECT * FROM restaurants", conn)
conn.close()

# --- Sidebar Inputs ---
st.sidebar.header("Choose your preferences")
cuisines = ['Café & Beverages', 'Continental / Western', 'Pan-Asian', 'South Asian']
areas = ['Dha Phase 6', 'Gulberg', 'Johar Town']

user_cuisine = st.sidebar.selectbox("Cuisine", cuisines, index=2)
user_area = st.sidebar.selectbox("Area", areas, index=0)
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 3.0, 0.1)
min_discount = st.sidebar.slider("Min Discount", 0.0, 1.0, 0.35, 0.05)

# --- Run button ---
if st.sidebar.button("Show Recommendations"):

    # --- User vector ---
    user_vector = pd.Series(0, index=[
        'cuisine_Café & Beverages', 'cuisine_Continental / Western',
        'cuisine_Pan-Asian', 'cuisine_South Asian',
        'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
    ])
    user_vector[f'cuisine_{user_cuisine}'] = 1
    user_vector[f'area_{user_area}'] = 1

    discount_cols = ['hbl_discount', 'meezan_discount', 'ubl_discount']

    # --- Zero out discounts below user-selected min_discount ---
    for col in discount_cols:
        master_table_encoded[col] = master_table_encoded[col].apply(
            lambda x: x if x >= min_discount else 0
        )

    # --- Recalculate max_discount after filtering individual banks ---
    master_table_encoded['max_discount'] = master_table_encoded[discount_cols].max(axis=1)

    # --- Filter restaurants ---
    filtered_restaurants = master_table_encoded[
        (master_table_encoded[f'cuisine_{user_cuisine}'] == 1) &
        (master_table_encoded[f'area_{user_area}'] == 1) &
        (master_table_encoded['google_rating'] >= min_rating) &
        (master_table_encoded['max_discount'] >= min_discount)
    ].copy()

    if filtered_restaurants.empty:
        st.warning("No restaurants match your criteria.")
    else:
        # --- CBF similarity ---
        cbf_features = [
            'cuisine_Café & Beverages', 'cuisine_Continental / Western',
            'cuisine_Pan-Asian', 'cuisine_South Asian',
            'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
        ]
        filtered_restaurants['cbf_similarity'] = cosine_similarity(
            filtered_restaurants[cbf_features],
            user_vector.values.reshape(1, -1)
        )

        # --- MCDM score ---
        filtered_restaurants['mcdm_score'] = (
            filtered_restaurants['google_rating_norm'] * 0.4 +
            filtered_restaurants['max_discount'] * 0.3 +
            filtered_restaurants['trending_norm'] * 0.2 +
            filtered_restaurants['avg_sentiment'] * 0.1
        )

        filtered_restaurants['final_score'] = (
            filtered_restaurants['cbf_similarity'] *
            filtered_restaurants['mcdm_score']
        )

        # --- Active discounts column ---
        def format_discounts(row):
            active = []
            for col, bank in zip(discount_cols, ['HBL', 'Meezan', 'UBL']):
                if row[col] > 0:
                    active.append(f"{bank} ({int(row[col]*100)}%)")
            return ", ".join(active) if active else "No Discount"

        filtered_restaurants['Available_Discounts'] = filtered_restaurants.apply(format_discounts, axis=1)

        # --- Display top 3 recommendations ---
        display_columns = [
            'restaurant_name',
            'google_rating',
            'Available_Discounts',
            'trending_label',
            'sentiment_label'
        ]

        st.dataframe(filtered_restaurants[display_columns].head(3).round(2))
        
