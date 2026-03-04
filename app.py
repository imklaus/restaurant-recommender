import sqlite3
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# --- Page Config ---
st.set_page_config(page_title="Restaurant Recommender", page_icon="🍴", layout="wide")

# --- Load DB ---
conn = sqlite3.connect("restaurant_db.db")
master_table_encoded = pd.read_sql("SELECT * FROM restaurants", conn)
conn.close()

# --- Sidebar Filters ---
st.sidebar.header("Filter Restaurants")
cuisines = ['Café & Beverages', 'Continental / Western', 'Pan-Asian', 'South Asian']
areas = ['Dha Phase 6', 'Gulberg', 'Johar Town']

user_cuisine = st.sidebar.selectbox("Cuisine", cuisines, index=2)
user_area = st.sidebar.selectbox("Area", areas, index=0)
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 3.0, 0.1)
min_discount = st.sidebar.slider("Min Discount", 0.0, 1.0, 0.35, 0.05)

discount_cols = ['hbl_discount', 'meezan_discount', 'ubl_discount']

# --- Show Recommendations ---
if st.button("Show Recommendations"):

    with st.spinner('Fetching your top restaurants... 🍽️'):
        # --- User vector ---
        user_vector = pd.Series(0, index=[
            'cuisine_Café & Beverages', 'cuisine_Continental / Western',
            'cuisine_Pan-Asian', 'cuisine_South Asian',
            'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
        ])
        user_vector[f'cuisine_{user_cuisine}'] = 1
        user_vector[f'area_{user_area}'] = 1

        # Discount filter
        for col in discount_cols:
            master_table_encoded[col] = master_table_encoded[col].apply(lambda x: x if x >= min_discount else 0)
        master_table_encoded['max_discount'] = master_table_encoded[discount_cols].max(axis=1)

        # Filter
        filtered = master_table_encoded[
            (master_table_encoded[f'cuisine_{user_cuisine}'] == 1) &
            (master_table_encoded[f'area_{user_area}'] == 1) &
            (master_table_encoded['google_rating'] >= min_rating) &
            (master_table_encoded['max_discount'] >= min_discount)
        ].copy()

        if filtered.empty:
            st.warning("No restaurants match your criteria 😢")
        else:
            # --- Scores ---
            cbf_features = [
                'cuisine_Café & Beverages', 'cuisine_Continental / Western',
                'cuisine_Pan-Asian', 'cuisine_South Asian',
                'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
            ]
            filtered['cbf_similarity'] = cosine_similarity(filtered[cbf_features], user_vector.values.reshape(1,-1))
            filtered['mcdm_score'] = (
                filtered['google_rating_norm']*0.4 + 
                filtered['max_discount']*0.3 +
                filtered['trending_norm']*0.2 +
                filtered['avg_sentiment']*0.1
            )
            filtered['final_score'] = filtered['cbf_similarity'] * filtered['mcdm_score']
            filtered = filtered.sort_values('final_score', ascending=False).head(3)

            # --- Display Million-Dollar Cards ---
            for idx, row in filtered.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div style='border-radius:20px; padding:20px; margin-bottom:15px;
                                    box-shadow:0 8px 20px rgba(0,0,0,0.15);
                                    background: linear-gradient(120deg, #ffffff, #f0f0f0);'>
                        <div style='display:flex; justify-content:space-between; align-items:center'>
                            <div>
                                <h2 style='margin:0'>{row['restaurant_name']}</h2>
                                <p style='margin:5px 0'>⭐ {row['google_rating']} | 🔥 {row['trending_label']} | 😊 {row['sentiment_label']}</p>
                                {" ".join([f"<span style='background-color:{'#4CAF50' if row[col]>0 else '#d3d3d3'}; color:white; padding:5px 10px; border-radius:8px; margin-right:5px'>{bank} {int(row[col]*100) if row[col]>0 else 0}%</span>" for col, bank in zip(discount_cols,['HBL','Meezan','UBL'])])}
                            </div>
                            <div style='width:150px;'>
                                <progress value='{min(row['final_score'],1.0)}' max='1' style='width:100%; height:25px; border-radius:10px'></progress>
                            </div>
                        </div>
                        </div>
                        """, unsafe_allow_html=True)
