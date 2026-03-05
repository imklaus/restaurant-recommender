import sqlite3
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Restaurant Recommender", page_icon="🍴", layout="wide")

# --- Main Heading at top center ---
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>AI Restaurant Recommendation Syystem</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Load DB ---
conn = sqlite3.connect("restaurant_db.db")
master_table_encoded = pd.read_sql("SELECT * FROM restaurants", conn)
conn.close()

# --- Sidebar: Filters ---
st.sidebar.header("Find Your Restaurant 🍽️")

cuisines = ['Café & Beverages', 'Continental / Western', 'Pan-Asian', 'South Asian']
areas = ['Dha Phase 6', 'Gulberg', 'Johar Town']

user_cuisine = st.sidebar.selectbox("Cuisine", cuisines, index=2)
user_area = st.sidebar.selectbox("Area", areas, index=0)
min_rating = st.sidebar.slider("Min Rating", 0.0, 5.0, 3.0, 0.1)
min_discount = st.sidebar.slider("Min Discount", 0.0, 1.0, 0.35, 0.05)

# --- Show Recommendations button BELOW filters ---
show_btn = st.sidebar.button("Show Recommendations")

# --- Main App: Show Recommendations ---
if show_btn:

    # --- User vector for CBF ---
    user_vector = pd.Series(0, index=[
        'cuisine_Café & Beverages', 'cuisine_Continental / Western',
        'cuisine_Pan-Asian', 'cuisine_South Asian',
        'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
    ])
    user_vector[f'cuisine_{user_cuisine}'] = 1
    user_vector[f'area_{user_area}'] = 1

    # --- Discount filtering ---
    discount_cols = ['hbl_discount', 'meezan_discount', 'ubl_discount']
    for col in discount_cols:
        master_table_encoded[col] = master_table_encoded[col].apply(lambda x: x if x >= min_discount else 0)
    master_table_encoded['max_discount'] = master_table_encoded[discount_cols].max(axis=1)

    # --- Filter by cuisine, area, rating, discount ---
    filtered = master_table_encoded[
        (master_table_encoded[f'cuisine_{user_cuisine}'] == 1) &
        (master_table_encoded[f'area_{user_area}'] == 1) &
        (master_table_encoded['google_rating'] >= min_rating) &
        (master_table_encoded['max_discount'] >= min_discount)
    ].copy()

    if filtered.empty:
        st.warning("No restaurants match your criteria.")
    else:
        # --- Content-Based Filtering & MCDM ---
        cbf_features = [
            'cuisine_Café & Beverages', 'cuisine_Continental / Western',
            'cuisine_Pan-Asian', 'cuisine_South Asian',
            'area_Dha Phase 6', 'area_Gulberg', 'area_Johar Town'
        ]
        filtered['cbf_similarity'] = cosine_similarity(
            filtered[cbf_features], user_vector.values.reshape(1, -1)
        )
        filtered['mcdm_score'] = (
            filtered['google_rating_norm'] * 0.4 +
            filtered['max_discount'] * 0.3 +
            filtered['trending_norm'] * 0.2 +
            filtered['avg_sentiment'] * 0.1
        )
        filtered['final_score'] = filtered['cbf_similarity'] * filtered['mcdm_score']
        filtered = filtered.sort_values('final_score', ascending=False).head(3)

        # --- Display each restaurant as a card ---
        for idx, row in filtered.iterrows():
            with st.container():
                cols = st.columns([1, 3])
                img_url = row['image_url'] if 'image_url' in row else "https://via.placeholder.com/100"
                with cols[0]:
                    st.image(img_url, width=6500)
                with cols[1]:
                    st.markdown(f"### {row['restaurant_name']}")
                    st.markdown(f"⭐ Rating: {row['google_rating']}")
                    # Bank discount badges
                    discount_badges = []
                    for col, bank in zip(discount_cols, ['HBL', 'Meezan', 'UBL']):
                        if row[col] > 0:
                            discount_badges.append(
                                f"<span style='background-color:#4CAF50;color:white;padding:3px 6px;border-radius:5px'>{bank} {int(row[col]*100)}%</span>"
                            )
                    st.markdown(" ".join(discount_badges), unsafe_allow_html=True)
                    st.markdown(f"🔥 Trending: {row['trending_label']} | 😊 Sentiment: {row['sentiment_label']}")
                st.markdown("---")
