import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import auc

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(page_title="Model Analysis Dashboard", layout="wide")

API_BASE_URL = "https://dun3co-logregmodel.hf.space"  # 

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def fetch_coefficients():
    url = f"{API_BASE_URL}/coefficients"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json()["coefficients"])

@st.cache_data(show_spinner=False)
def fetch_shap(limit=100):
    url = f"{API_BASE_URL}/explain?limit={limit}"
    response = requests.post(url)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame(data["shap_summary"])

@st.cache_data(show_spinner=False)
def fetch_metrics(limit=100):
    url = f"{API_BASE_URL}/metrics?limit={limit}"
    response = requests.post(url)
    response.raise_for_status()
    return response.json()

# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 Logistic Regression Model Analysis Dashboard")
st.markdown("This dashboard visualizes model insights served by your FastAPI API on Hugging Face Spaces.")

st.divider()

# ============================================================
# FEATURE IMPORTANCE (COEFFICIENTS)
# ============================================================

st.header("1️⃣ Logistic Regression Coefficients")

if st.button("Fetch Feature Importance", type="primary"):
    with st.spinner("Retrieving coefficients..."):
        try:
            importance = fetch_coefficients()
            importance["odds_ratio"] = np.exp(importance["coefficient"])
            importance = importance.sort_values(by="coefficient", key=abs, ascending=False)

            top_n = st.slider("Select number of top features", 5, 30, 15, key="coeff_slider")

            fig, ax = plt.subplots(figsize=(8, 6))
            importance.head(top_n).set_index("feature")["coefficient"].plot(kind="barh", ax=ax, color="#4C72B0")
            ax.set_title("Logistic Regression Feature Importance (Coefficients)")
            ax.set_xlabel("Coefficient Value")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            st.dataframe(importance.head(top_n).style.format({"coefficient": "{:.3f}", "odds_ratio": "{:.3f}"}))

        except Exception as e:
            st.error(f"❌ Error fetching coefficients: {e}")

st.divider()

# ============================================================
# SHAP FEATURE IMPORTANCE
# ============================================================

st.header("2️⃣ SHAP Feature Importance (Global Explanation)")

limit = st.slider("Number of test samples for SHAP computation", 20, 500, 100, step=20, key="shap_limit")

if st.button("Run SHAP Analysis"):
    with st.spinner("Calculating SHAP feature importances..."):
        try:
            shap_df = fetch_shap(limit)
            shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#55A868")
            ax.set_title("SHAP Feature Importance (Mean Absolute SHAP Value)")
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_ylabel("Feature")
            st.pyplot(fig)

            st.dataframe(shap_df.sort_values("mean_abs_shap", ascending=False).head(20))

        except Exception as e:
            st.error(f"❌ Error running SHAP analysis: {e}")

st.divider()

# ============================================================
# MODEL PERFORMANCE METRICS
# ============================================================

st.header("3️⃣ Model Performance Metrics (ROC / PR Curves)")

limit_metrics = st.slider("Number of samples for metrics", 20, 500, 100, step=20, key="metric_limit")

if st.button("Compute Model Metrics"):
    with st.spinner("Fetching metrics from API..."):
        
        metrics = fetch_metrics(limit_metrics)

        if "error" in metrics:
            st.error(f"API error: {metrics['error']}")
        else:
            roc_auc = metrics.get("roc_auc")
            pr_auc = metrics.get("pr_auc")
            thresholds = metrics.get("thresholds", [])
            precision = metrics.get("precision", [])
            recall = metrics.get("recall", [])

            if roc_auc is not None and pr_auc is not None:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ROC AUC", f"{roc_auc:.3f}")
                with col2:
                    st.metric("PR AUC", f"{pr_auc:.3f}")

                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.plot([0, 1], [0, 1], "k--", label="Random")
                if thresholds and len(thresholds) == len(recall):
                    ax1.plot(recall, precision, label="Model")
                ax1.set_xlabel("False Positive Rate (approx.)")
                ax1.set_ylabel("True Positive Rate (approx.)")
                ax1.set_title("ROC-like Visualization")
                ax1.legend()
                st.pyplot(fig1)

                fig2, ax2 = plt.subplots(figsize=(5, 5))
                ax2.plot(recall, precision, color="#C44E52")
                ax2.set_xlabel("Recall")
                ax2.set_ylabel("Precision")
                ax2.set_title("Precision-Recall Curve")
                st.pyplot(fig2)
            else:
                st.warning("Metrics not available. Check API logs or ensure your dataset has a 'y' column.")


st.divider()

# ============================================================
# HEALTH CHECK
# ============================================================

st.caption("🔍 Checking API health...")

try:
    res = requests.get(f"{API_BASE_URL}/health", timeout=5)
    if res.status_code == 200:
        st.success("✅ API is online and healthy")
    else:
        st.warning(f"⚠️ API responded with status: {res.status_code}")
except Exception as e:
    st.error(f"❌ Could not connect to API: {e}")
