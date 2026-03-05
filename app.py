import streamlit as st
import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Restaurant Recommendation Engine",
    page_icon="🍽️",
    layout="wide"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>

.main {
background-color:#f5f7fb;
}

.header {
background: linear-gradient(90deg,#1f2937,#111827);
padding:35px;
border-radius:14px;
color:white;
margin-bottom:30px;
}

.title {
font-size:40px;
font-weight:700;
}

.card {
background:white;
padding:24px;
border-radius:16px;
box-shadow:0 6px 24px rgba(0,0,0,0.08);
margin-bottom:18px;
}

.top-card {
background:linear-gradient(120deg,#fff9e6,#ffffff);
border:2px solid #ffd54f;
}

.restaurant {
font-size:22px;
font-weight:600;
}

.meta {
color:#666;
font-size:14px;
margin-top:4px;
}

.discount {
background:#eef4ff;
padding:5px 12px;
border-radius:8px;
font-size:13px;
margin-right:6px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="header">
<div class="title">Restaurant Recommendation Engine</div>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATABASE ----------------

conn = sqlite3.connect("restaurant_db.db")
master_table_encoded = pd.read_sql("SELECT * FROM restaurants", conn)
conn.close()

# ---------------- FILTER OPTIONS ----------------

cuisines = ['Café & Beverages', 'Continental / Western', 'Pan-Asian', 'South Asian']
areas = ['Dha Phase 6', 'Gulberg', 'Johar Town']

# ---------------- FILTER UI ----------------

col1,col2,col3,col4 = st.columns(4)

with col1:
    user_cuisine = st.selectbox("Cuisine", cuisines)

with col2:
    user_area = st.selectbox("Area", areas)

with col3:
    min_rating = st.slider("Minimum Rating",0.0,5.0,3.0,0.1)

with col4:
    min_discount = st.slider("Minimum Discount",0.0,1.0,0.30,0.05)

run = st.button("Find Restaurants")

# ---------------- RECOMMENDATION ENGINE ----------------

if run:

    discount_cols = ['hbl_discount','meezan_discount','ubl_discount']

    user_vector = pd.Series(0,index=[
        'cuisine_Café & Beverages','cuisine_Continental / Western',
        'cuisine_Pan-Asian','cuisine_South Asian',
        'area_Dha Phase 6','area_Gulberg','area_Johar Town'
    ])

    user_vector[f'cuisine_{user_cuisine}'] = 1
    user_vector[f'area_{user_area}'] = 1

    master_table_encoded['max_discount'] = master_table_encoded[discount_cols].max(axis=1)

    filtered = master_table_encoded[
        (master_table_encoded[f'cuisine_{user_cuisine}']==1) &
        (master_table_encoded[f'area_{user_area}']==1) &
        (master_table_encoded['google_rating']>=min_rating) &
        (master_table_encoded['max_discount']>=min_discount)
    ].copy()

    if filtered.empty:

        st.warning("No restaurants match your criteria")

    else:

        for col in discount_cols:
            filtered[col] = filtered[col].apply(lambda x: x if x>=min_discount else 0)

        cbf_features = [
            'cuisine_Café & Beverages','cuisine_Continental / Western',
            'cuisine_Pan-Asian','cuisine_South Asian',
            'area_Dha Phase 6','area_Gulberg','area_Johar Town'
        ]

        filtered['cbf_similarity'] = cosine_similarity(
            filtered[cbf_features],
            user_vector.values.reshape(1,-1)
        )

        filtered['mcdm_score'] = (
            filtered['google_rating_norm']*0.4 +
            filtered['max_discount']*0.3 +
            filtered['trending_norm']*0.2 +
            filtered['avg_sentiment']*0.1
        )

        filtered['final_score'] = filtered['cbf_similarity'] * filtered['mcdm_score']

        final_recommendations = filtered.sort_values(
            by='final_score',
            ascending=False
        ).head(3)

        # ---------------- DISPLAY RESULTS ----------------

        rank = 1

        for _,row in final_recommendations.iterrows():

            card_class = "card"

            if rank == 1:
                card_class = "card top-card"

            st.markdown(f"""
            <div class="{card_class}">

            <div style="display:flex;justify-content:space-between">

            <div>

            <div class="restaurant">
            #{rank} {row['restaurant_name']}
            </div>

            <div class="meta">
            Rating: {row['google_rating']} |
            Trending: {row['trending_label']} |
            Sentiment: {row['sentiment_label']}
            </div>

            <div style="margin-top:12px">

            <span class="discount">HBL {int(row['hbl_discount']*100)}%</span>
            <span class="discount">Meezan {int(row['meezan_discount']*100)}%</span>
            <span class="discount">UBL {int(row['ubl_discount']*100)}%</span>

            </div>

            </div>

            </div>

            </div>
            """, unsafe_allow_html=True)

            rank += 1
