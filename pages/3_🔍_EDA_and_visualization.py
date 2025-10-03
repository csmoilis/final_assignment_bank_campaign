import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import requests
import zipfile
import io

# Use session state to load data only once per session
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    df = pd.read_csv(z.open("bank-full.csv"), sep=";")
    return df

if "bank_df" not in st.session_state:
    st.session_state["bank_df"] = load_data()

df = st.session_state["bank_df"]

distribution_variables = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
imbalance_variables = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

with st.sidebar:
    st.header("Distribution visualizations")
    with st.form("Visualization options"):
        select_variable = st.multiselect("Select variable(s) to visualize", distribution_variables, default=['age','balance'])
        plot_type = st.radio("Plot type", options=["Histogram", "KDE"], index=0)
        submit_button = st.form_submit_button(label="Update")
    st.header("Imbalance plots")
    with st.form("Imbalance options"):
        select_imbalance = st.multiselect("Select variable(s) to visualize", imbalance_variables, default=['job','education'])
        submit_button2 = st.form_submit_button(label="Update")

st.set_page_config(page_title="EDA and Visualization", page_icon=":mag:", layout="wide")

# Distribution plots (histogram or kde by 'y')
if submit_button and select_variable:
    st.subheader("Distribution by Target (y)")
    # Set plot width to 1/3 of the screen (Streamlit columns)
    n_cols = 3
    cols = st.columns(n_cols)
    for idx, var in enumerate(select_variable):
        col = cols[idx % n_cols]
        if plot_type == "Histogram":
            chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
                x=alt.X(var, bin=alt.Bin(maxbins=30), title=var),
                y=alt.Y('count()', title='Count'),
                color=alt.Color('y', title='Subscribed'),
                tooltip=[var, 'y']
            ).properties(
                width=350,
                height=250,
                title=f"{var} histogram by subscription"
            )
            col.altair_chart(chart, use_container_width=True)
        else:  # KDE
            fig, ax = plt.subplots(figsize=(4, 3))
            for label, color in zip(['yes', 'no'], ['green', 'red']):
                subset = df[df['y'] == label][var]
                sns.kdeplot(subset, label=f"y = {label}", color=color, fill=True, alpha=0.3, ax=ax)
            ax.set_title(f"{var} KDE by subscription")
            ax.set_xlabel(var)
            ax.set_ylabel("Density")
            ax.legend()
            col.pyplot(fig)
            plt.close(fig)

# Helper to cache imbalance proportions per variable in session state
def get_prop_df(var):
    cache_key = f"prop_df_{var}"
    if cache_key not in st.session_state:
        prop_df = (
            df.groupby(var)['y']
            .value_counts(normalize=True)
            .rename('proportion')
            .reset_index()
        )
        st.session_state[cache_key] = prop_df
    return st.session_state[cache_key]

# Imbalance plots (stacked bar by 'y')
if submit_button2 and select_imbalance:
    st.subheader("Imbalance by Target (y)")
    for var in select_imbalance:
        prop_df = get_prop_df(var)
        df_tooltip = pd.merge(df, prop_df, on=[var, 'y'], how='left')

        chart = alt.Chart(df_tooltip).mark_bar().encode(
            x=alt.X(var, title=var),
            y=alt.Y('count()', stack='normalize', title='Proportion'),
            color=alt.Color('y', title='Subscribed', sort=['no', 'yes']),  # 'yes' on top
            order=alt.Order('y', sort='ascending'),  # ensures 'yes' is on top
            tooltip=[
                var,
                'y',
                alt.Tooltip('proportion:Q', title='Proportion', format='.2%'),
                alt.Tooltip('count():Q', title='Count')
            ]
        ).properties(
            width=350,
            height=650,
            title=f"{var} imbalance by subscription"
        )
        st.altair_chart(chart, use_container_width=True)
