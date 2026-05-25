import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title="Zomato Restaurant Clustering",
    page_icon="🍽️",
    layout="centered"
)

@st.cache_resource
def load_models():
    kmeans = joblib.load("kmeans_model.pkl")
    scaler = joblib.load("scaler.pkl")
    pca    = joblib.load("pca.pkl")
    return kmeans, scaler, pca

kmeans, scaler, pca = load_models()

CLUSTER_INFO = {
    0: {
        "label": "⭐ Premium Restaurants",
        "desc": "High rating (4.2+), high cost, very high votes. Expensive, popular, high-quality restaurants.",
    },
    1: {
        "label": "🍛 Budget Restaurants",
        "desc": "Medium/low rating, low cost, low votes. Cheap restaurants, less popular.",
    },
    2: {
        "label": "😐 Low Rated / Less Popular",
        "desc": "Low rating (~3.2), low votes, low-medium cost. New, unpopular, or low-quality restaurants.",
    },
    3: {
        "label": "🥂 Popular Mid-Range",
        "desc": "Good rating (~4.0), medium-high cost (~₹1499), moderate votes. Affordable-to-premium popular places.",
    },
}

st.title("🍽️ Zomato Restaurant Cluster Predictor")
st.markdown("Enter a restaurant's **rating**, **number of votes**, and **approximate cost for two people** to find which cluster it belongs to.")
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    rate = st.number_input("⭐ Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.1, format="%.1f")

with col2:
    votes = st.number_input("🗳️ Votes", min_value=0, max_value=100000, value=500, step=10)

with col3:
    cost = st.number_input("💰 Approx Cost (₹ for 2)", min_value=50, max_value=10000, value=800, step=50)

st.divider()

if st.button("🔍 Predict Cluster", use_container_width=True, type="primary"):
    input_data   = np.array([[rate, votes, cost]])
    input_scaled = scaler.transform(input_data)
    cluster_id   = int(kmeans.predict(input_scaled)[0])
    info         = CLUSTER_INFO[cluster_id]

    st.success(f"### Predicted Cluster: {info['label']}")
    st.info(f"📝 {info['desc']}")

    st.markdown("#### 📊 Cluster Reference (Average Values)")
    st.markdown("""
| Cluster | Label | Avg Rating | Avg Votes | Avg Cost (₹) |
|:-------:|-------|:----------:|:---------:|:------------:|
| 0 | ⭐ Premium | 4.29 | 3105 | 1112 |
| 1 | 🍛 Budget | 3.86 | 190 | 451 |
| 2 | 😐 Low Rated | 3.20 | 57 | 413 |
| 3 | 🥂 Popular Mid-Range | 4.04 | 523 | 1499 |
""")

st.divider()
st.markdown("### 🗺️ Cluster Overview")

cols = st.columns(2)
for i, (cid, info) in enumerate(CLUSTER_INFO.items()):
    with cols[i % 2]:
        st.markdown(f"**{info['label']}**")
        st.caption(info["desc"])

st.divider()

with st.expander("ℹ️ How does this work?"):
    st.markdown("""
**ML Pipeline:**
1. Load Zomato restaurant data (7663 rows → 7055 after cleaning)
2. Select 3 features: `rate`, `votes`, `approx_cost(for two people)`
3. Scale with **StandardScaler**
4. Use **Elbow Method** to find optimal clusters → **k = 4**
5. Train **KMeans (k=4)** on scaled data
6. Your input is scaled with the same scaler, then predicted by KMeans
""")