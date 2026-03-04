
import streamlit as st
import pandas as pd
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("🍽️ Restaurant Recommendation System")

@st.cache_data
def load_data():
    conn = sqlite3.connect("restaurant_db.db")
    df = pd.read_sql("SELECT * FROM restaurants", conn)
    conn.close()
    return df

master_table_encoded = load_data()

cuisines = ['Café & Beverages', 'Continental / Western', 'Pan-Asian', 'South Asian']
areas = ['Dha Phase 6', 'Gulberg', 'Johar Town']

user_cuisine = st.selectbox("Select Cuisine", cuisines)
user_area = st.selectbox("Select Area", areas)

min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.1)
min_discount = st.slider("Minimum Discount", 0.0, 1.0, 0.3, 0.05)

if st.button("Show Recommendations"):

    discount_cols = ['hbl_discount', 'meezan_discount', 'ubl_discount']
    master_table_encoded['max_discount'] = master_table_encoded[discount_cols].max(axis=1)

    filtered = master_table_encoded[
        (master_table_encoded[f'cuisine_{user_cuisine}'] == 1) &
        (master_table_encoded[f'area_{user_area}'] == 1) &
        (master_table_encoded['google_rating'] >= min_rating) &
        (master_table_encoded['max_discount'] >= min_discount)
    ].copy()

    if filtered.empty:
        st.warning("No restaurants match your criteria.")
    else:
        filtered['mcdm_score'] = (
            filtered['google_rating_norm'] * 0.4 +
            filtered['max_discount'] * 0.3 +
            filtered['trending_norm'] * 0.2 +
            filtered['avg_sentiment'] * 0.1
        )

        cbf_features = [
            'cuisine_Café & Beverages', 'cuisine_Continental / Western',
            'cuisine_Pan-Asian', 'cuisine_South Asian',
            'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
        ]
        user_vector = pd.Series(0, index=cbf_features)
        user_vector[f'cuisine_{user_cuisine}'] = 1
        user_vector[f'area_{user_area}'] = 1

        filtered['cbf_similarity'] = cosine_similarity(
            filtered[cbf_features], user_vector.values.reshape(1, -1)
        )

        filtered['final_score'] = filtered['mcdm_score'] * filtered['cbf_similarity']

        def format_discounts(row):
            active = []
            if row['hbl_discount'] > 0:
                active.append(f"HBL ({int(row['hbl_discount']*100)}%)")
            if row['meezan_discount'] > 0:
                active.append(f"Meezan ({int(row['meezan_discount']*100)}%)")
            if row['ubl_discount'] > 0:
                active.append(f"UBL ({int(row['ubl_discount']*100)}%)")
            return ", ".join(active) if active else "No Discount"

        filtered['Available_Discounts'] = filtered.apply(format_discounts, axis=1)

        display_columns = [
            'restaurant_name',
            'google_rating',
            'Available_Discounts',
            'trending_label',
            'sentiment_label'
        ]
        st.subheader("Top 3 Recommendations")
        st.dataframe(filtered[display_columns].head(3).round(2))
