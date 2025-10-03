import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import requests
import zipfile
import io


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
r = requests.get(url)

# Open the zip file in memory
z = zipfile.ZipFile(io.BytesIO(r.content))

# Extract and read "bank.csv"
df = pd.read_csv(z.open("bank-full.csv"), sep=";")

# Load dataset

loaded_pipeline = joblib.load("model_1mvp.pkl")
print("Model loaded successfully.")

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Tips Analysis App", layout="wide")
st.title("🍽️ Restaurant Tips Analysis")
st.write("This app helps you explore how customers tip based on time, day, and bill amount.")

# Sidebar filter (only time)
with st.sidebar:
    st.subheader("Filters")
    all_time_options = sorted(tips_noduplicates["time"].dropna().unique().tolist())
    selected_options = st.multiselect(
        "Meal time",
        options=all_time_options,
        default=all_time_options,
    )

# Main content
if not selected_options:
    st.info("Select at least one time to display the plot.")
else:
    tips_filtered = tips_noduplicates[tips_noduplicates["time"].isin(selected_options)]

    # Group KPI table
    table_kpi = (
        tips_filtered
        .groupby(["day", "smoker"])
        .agg(
            total_bill_sum=("total_bill", "sum"),
            total_people=("size", "sum")
        )
        .reset_index()
    )
    table_kpi["bill_per_person"] = table_kpi["total_bill_sum"] / table_kpi["total_people"]

    st.subheader("Summary Table")
    st.dataframe(table_kpi)

st.caption("Developed for best team.")
