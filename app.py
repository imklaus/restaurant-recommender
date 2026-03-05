import streamlit as st
import sqlite3
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

# ---------------- STYLE ----------------

st.markdown("""
<style>

.main {
background:#f4f6fb;
}

.header {
background: linear-gradient(90deg,#111827,#1f2937);
padding:30px;
border-radius:14px;
color:white;
margin-bottom:25px;
}

.title {
font-size:38px;
font-weight:700;
}

.card {
background:rgba(255,255,255,0.95);
padding:22px;
border-radius:16px;
box-shadow:0 8px 25px rgba(0,0,0,0.08);
margin-bottom:18px;
}

.top-card {
border:2px solid #ffd54f;
background:linear-gradient(120deg,#fff9e6,#ffffff);
}

.restaurant {
font-size:22px;
font-weight:700;
color:#111;
}

.meta {
color:#444;
font-size:14px;
margin-top:5px;
}

.bank {
display:inline-block;
padding:6px 12px;
border-radius:8px;
font-size:13px;
margin-right:6px;
font-weight:600;
}

.hbl {background:#e7f6ed;color:#008f3b;}
.meezan {background:#e8f3ff;color:#0057b8;}
.ubl {background:#fff0f0;color:#c00000;}

img {
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown("""
<div class="header">
<div class="title">Restaurant Recommendation Engine</div>
</div>
""", unsafe_allow_html=True)

# ---------------- DATABASE ----------------

conn = sqlite3.connect("restaurant_db.db")
master_table_encoded = pd.read_sql("SELECT * FROM restaurants", conn)
conn.close()

# ---------------- FILTERS (SIDEBAR) ----------------

st.sidebar.header("Filters")

cuisines = ['Café & Beverages','Continental / Western','Pan-Asian','South Asian']
areas = ['Dha Phase 6','Gulberg','Johar Town']

user_cuisine = st.sidebar.selectbox("Cuisine", cuisines)
user_area = st.sidebar.selectbox("Area", areas)

min_rating = st.sidebar.slider("Minimum Rating",0.0,5.0,3.0,0.1)
min_discount = st.sidebar.slider("Minimum Discount",0.0,1.0,0.30,0.05)

run = st.sidebar.button("Find Restaurants")

# ---------------- IMAGE FUNCTION ----------------

def get_restaurant_image(name):

    query = name.replace(" ","+")

    return f"https://source.unsplash.com/600x400/?restaurant,{query}"

# ---------------- ENGINE ----------------

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

        # ---------------- DISPLAY ----------------

        rank = 1

        for _,row in final_recommendations.iterrows():

            card_class = "card"

            if rank == 1:
                card_class = "card top-card"

            image_url = get_restaurant_image(row['restaurant_name'])

            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)

            col1,col2 = st.columns([1,2])

            with col1:
                st.image(image_url)

            with col2:

                st.markdown(f"""
                <div class="restaurant">🍽️ #{rank} {row['restaurant_name']}</div>
                <div class="meta">
                Rating: {row['google_rating']} |
                Trending: {row['trending_label']} |
                Sentiment: {row['sentiment_label']}
                </div>
                """, unsafe_allow_html=True)

                banks = ""

                if row['hbl_discount']>0:
                    banks += f'<span class="bank hbl">HBL {int(row["hbl_discount"]*100)}%</span>'

                if row['meezan_discount']>0:
                    banks += f'<span class="bank meezan">Meezan {int(row["meezan_discount"]*100)}%</span>'

                if row['ubl_discount']>0:
                    banks += f'<span class="bank ubl">UBL {int(row["ubl_discount"]*100)}%</span>'

                st.markdown(banks, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            rank += 1
