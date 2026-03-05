import streamlit as st
import pandas as pd
import numpy as np
import re

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Restaurant Recommender",
    page_icon="🍽️",
    layout="wide"
)

st.title("🍽️ Restaurant Recommendation System")

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("restaurant_final_dataset.csv")
    return df

df = load_data()

# -----------------------------
# Image function (stable)
# -----------------------------
def get_restaurant_image(name):
    return "https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=800&q=60"

# -----------------------------
# Bank Selection
# -----------------------------
bank = st.selectbox(
    "Select your Bank",
    ["All Banks", "HBL", "Meezan", "UBL"]
)

# -----------------------------
# Minimum Rating Filter
# -----------------------------
min_rating = st.slider(
    "Minimum Google Rating",
    3.0,
    5.0,
    4.0,
    0.1
)

# -----------------------------
# Apply Filters
# -----------------------------
filtered = df[df["google_rating"] >= min_rating]

if bank == "HBL":
    filtered = filtered[filtered["hbl_discount"] > 0]

elif bank == "Meezan":
    filtered = filtered[filtered["meezan_discount"] > 0]

elif bank == "UBL":
    filtered = filtered[filtered["ubl_discount"] > 0]

# -----------------------------
# Ranking Score
# -----------------------------
filtered["score"] = (
    filtered["google_rating_norm"] * 0.5
    + filtered["engagement_norm"] * 0.3
    + filtered["avg_sentiment"] * 0.2
)

final_recommendations = filtered.sort_values(
    by="score",
    ascending=False
).head(10)

# -----------------------------
# Display Results
# -----------------------------
st.subheader("Top Recommended Restaurants")

rank = 1

for _, row in final_recommendations.iterrows():

    image_url = get_restaurant_image(row["restaurant_name"])

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image_url, use_container_width=True)

    with col2:

        st.markdown(
            f"<h3 style='color:white'>🍽️ #{rank} {row['restaurant_name']}</h3>",
            unsafe_allow_html=True
        )

        st.write(
            f"⭐ Rating: {row['google_rating']}  |  🔥 Trending: {row['trending_label']}  |  💬 Sentiment: {row['sentiment_label']}"
        )

        if row["hbl_discount"] > 0:
            st.success(f"HBL {int(row['hbl_discount']*100)}% Discount")

        if row["meezan_discount"] > 0:
            st.info(f"Meezan {int(row['meezan_discount']*100)}% Discount")

        if row["ubl_discount"] > 0:
            st.error(f"UBL {int(row['ubl_discount']*100)}% Discount")

    st.divider()

    rank += 1

# -----------------------------
# Empty Case
# -----------------------------
if final_recommendations.empty:
    st.warning("No restaurants found with current filters.")
