import streamlit as st
import pandas as pd
import joblib
import os


st.set_page_config(
    page_title="Retail Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    kmeans = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    cluster_features = joblib.load(os.path.join(MODEL_DIR, "cluster_features.pkl"))
    return kmeans, scaler, cluster_features

try:
    kmeans, scaler, cluster_features = load_models()
except FileNotFoundError:
    st.error(
        "Could not find kmeans_model.pkl, scaler.pkl, or cluster_features.pkl. "
        "Make sure these files are in the same folder as app.py."
    )
    st.stop()


SEGMENT_PROFILES = {
    0: {
        "label": "Established Steady Spenders",
        "description": (
            "Older, moderate income, solid and consistent spending. Longest-tenured "
            "customers on the platform, but rarely respond to marketing campaigns."
        ),
        "stats": {
            "Age": 59.8, "Income": 57930, "Total_Children": 1.16, "Recency": 47.75,
            "Total_Spending": 791.70, "Total_Purchases": 21.53, "NumWebVisitsMonth": 5.98,
            "Customer_Tenure_Days": 455.28, "Total_Accepted_Cmp": 0.20
        },
        "share": "27% of customers"
    },
    1: {
        "label": "Low-Value Browsers",
        "description": (
            "Lowest income and lowest spending of all segments, despite visiting the "
            "site most frequently. Largest segment by far, and the least responsive to campaigns."
        ),
        "stats": {
            "Age": 51.9, "Income": 34238, "Total_Children": 1.23, "Recency": 49.25,
            "Total_Spending": 100.60, "Total_Purchases": 8.00, "NumWebVisitsMonth": 6.50,
            "Customer_Tenure_Days": 317.14, "Total_Accepted_Cmp": 0.09
        },
        "share": "45% of customers (largest segment)"
    },
    2: {
        "label": "High-Value Campaign Responders",
        "description": (
            "Highest income and highest spending, with a campaign acceptance rate far "
            "above every other segment. The smallest but most valuable group to target."
        ),
        "stats": {
            "Age": 53.5, "Income": 80722, "Total_Children": 0.28, "Recency": 48.91,
            "Total_Spending": 1590.98, "Total_Purchases": 21.18, "NumWebVisitsMonth": 3.54,
            "Customer_Tenure_Days": 351.39, "Total_Accepted_Cmp": 2.51
        },
        "share": "6% of customers (smallest segment)"
    },
    3: {
        "label": "Affluent but Disengaged",
        "description": (
            "High income and high spending similar to the top segment, but the lowest "
            "web engagement and weak campaign response. They spend well but tune out marketing."
        ),
        "stats": {
            "Age": 56.4, "Income": 75731, "Total_Children": 0.21, "Recency": 50.24,
            "Total_Spending": 1222.91, "Total_Purchases": 19.72, "NumWebVisitsMonth": 2.28,
            "Customer_Tenure_Days": 300.03, "Total_Accepted_Cmp": 0.28
        },
        "share": "20% of customers"
    },
}

FEATURE_LABELS = {
    "Age": "Age (years)",
    "Income": "Annual Income ($)",
    "Total_Children": "Children at Home",
    "Recency": "Days Since Last Purchase",
    "Total_Spending": "Total Spending ($)",
    "Total_Purchases": "Total Purchases",
    "NumWebVisitsMonth": "Web Visits / Month",
    "Customer_Tenure_Days": "Customer Tenure (days)",
    "Total_Accepted_Cmp": "Campaigns Accepted",
}


@st.dialog("Predicted Segment")
def show_result(profile, input_values):
    st.markdown(f"### {profile['label']}")
    st.caption(profile["share"])
    st.info(profile["description"])

    with st.expander("See how this customer compares to the segment average"):
        compare_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in cluster_features],
            "This Customer": [input_values[f] for f in cluster_features],
            f"{profile['label']} Average": [profile["stats"][f] for f in cluster_features],
        })
        st.dataframe(compare_df, hide_index=True, width="stretch")


st.title("🛍️ Retail Customer Segmentation")
st.markdown(
    "Predict which customer segment a shopper belongs to based on their "
    "purchasing behaviour and demographics, using a KMeans clustering model."
)

tab_predict, tab_explore = st.tabs(["🔮 Predict a Segment", "📊 Segment Profiles"])


with tab_predict:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=45)
        income = st.number_input("Annual Income ($)", min_value=0, max_value=300000, value=50000, step=1000)
        total_children = st.number_input(
            "Children at Home", min_value=0, max_value=10, value=1,
            help="The model was trained on households with 0-4 children. "
                 "Values above that are accepted but the prediction becomes a rougher estimate."
        )

    with col2:
        recency = st.number_input("Days Since Last Purchase", min_value=0, max_value=365, value=30)
        total_spending = st.number_input("Total Spending ($)", min_value=0, max_value=5000, value=500, step=50)
        total_purchases = st.number_input("Total Purchases", min_value=0, max_value=100, value=15)

    with col3:
        web_visits = st.number_input("Web Visits / Month", min_value=0, max_value=20, value=5)
        tenure_days = st.number_input("Customer Tenure (days)", min_value=0, max_value=3000, value=365)
        accepted_cmp = st.number_input(
            "Campaigns Accepted", min_value=0, max_value=5, value=0,
            help="Out of the 5 past marketing campaigns run by the business, "
                 "how many did this customer respond to/accept?"
        )

    st.write("")

    if st.button("Predict Segment", type="primary", width="content"):
        input_values = {
            "Age": age,
            "Income": income,
            "Total_Children": total_children,
            "Recency": recency,
            "Total_Spending": total_spending,
            "Total_Purchases": total_purchases,
            "NumWebVisitsMonth": web_visits,
            "Customer_Tenure_Days": tenure_days,
            "Total_Accepted_Cmp": accepted_cmp,
        }

        # Build input in the exact feature order the scaler/model expect
        input_df = pd.DataFrame([[input_values[f] for f in cluster_features]], columns=cluster_features)
        input_scaled = scaler.transform(input_df)
        input_scaled_df = pd.DataFrame(input_scaled, columns=cluster_features)
        predicted_cluster = int(kmeans.predict(input_scaled_df)[0])

        profile = SEGMENT_PROFILES.get(predicted_cluster)
        show_result(profile, input_values)


with tab_explore:
    st.subheader("Segment Overview")

    cols = st.columns(4)
    for i, (cid, profile) in enumerate(SEGMENT_PROFILES.items()):
        with cols[i]:
            st.markdown(f"**{profile['label']}**")
            st.caption(profile["share"])
            st.write(profile["description"])

    st.markdown("---")
    st.subheader("Average Feature Values by Segment")

    profile_table = pd.DataFrame({
        profile["label"]: profile["stats"] for profile in SEGMENT_PROFILES.values()
    }).T
    profile_table.index.name = "Segment"
    st.dataframe(profile_table, width="stretch")

st.markdown("---")
st.caption("Built with scikit-learn KMeans clustering · Doks Studio / 3MTT Data Science Capstone")