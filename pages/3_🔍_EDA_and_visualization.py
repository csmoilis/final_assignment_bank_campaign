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

st.title("Exploratory Data Analysis and Visualization")
st.markdown("""
This page allows you to explore the distribution of numerical variables and the imbalance of categorical variables in relation to the target variable `y` (whether the client subscribed to a long term deposit). Use the sidebar to select variables and plot types.
- **Distribution Plots**: Choose numerical variables to see their distribution (histogram or KDE) segmented by the target variable `y`.
- **Imbalance Plots**: Choose categorical variables to visualize their class distribution segmented by the target variable `y`.

Depeding on your selections, the plots will update accordingly, and our thoughts will be attached after each plot.
""")
# Distribution plots (histogram or kde by 'y')
if submit_button and select_variable:
    st.subheader("Distribution by Target (y)")
    n_cols = 2
    n_vars = len(select_variable)
    n_rows = (n_vars + n_cols - 1) // n_cols  # Ceiling division

    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col_idx in range(n_cols):
            idx = row * n_cols + col_idx
            if idx >= n_vars:
                # If there are fewer plots than grid cells, leave empty
                continue
            var = select_variable[idx]
            col = cols[col_idx]
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
                # After plotting each distribution plot:
                if var == "age":
                    col.info("Age is right-skewed, most clients are between 30 and 50.")
                elif var == "balance":
                    col.info("Balance has a long tail, with most clients having low or negative balances. We will clip it for our model.")
                elif var == "day":
                    col.info("Day of month is fairly uniform, with slight peaks around the start and end of the month. There might be weekends affecting this.")
                elif var == "duration":
                    col.info("Duration is right-skewed, with many short calls and a few very long ones. We will clip it for our model.")
                elif var == "campaign":
                    col.info("Campaign calls are right-skewed, with most clients receiving few calls. We won't clip this for our model, as it might be informative or affect the target variable too much.")
                elif var == "pdays":
                    col.info("-1 indicates no previous contact, which is very common in the data. Other values are right-skewed, with many clients not contacted for a long time. We've decided to bin them in the model.")
                elif var == "previous":
                    col.info("Previous contacts are right-skewed, with most clients having few previous contacts. We will bin this for our model.")
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
                # After plotting each distribution plot:
                if var == "age":
                    col.info("The density shows a peak around 35-40 years, with a slight difference between subscribers and non-subscribers. Notably, subscribers tend to be slightly older.")
                elif var == "balance":
                    col.info("Subscribers tend to have higher balances, as seen by the green curve shift.")
                elif var == "day":
                    col.info("The density is fairly uniform, with slight peaks, maybe around weekend? Better at the start of month?. Subscribers show a slightly different pattern.")
                elif var == "duration":
                    col.info("Subscribers tend to have longer call durations, as seen by the green curve shift. However, there might be model leakage, as we can't use call duration to predict subscription, because we dont know for how long the call is going to go.")
                elif var == "campaign":
                    col.info("The density is right-skewed, with most clients having few campaign calls.")
                elif var == "pdays":
                    col.info("The density shows a peak at -1 (no previous contact). Subscribers tend to have been contacted more recently. We've binned this for our model.")
                elif var == "previous":
                    col.info("The density is right-skewed, with most clients having few previous contacts. We've binned this for our model.")
    
            

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

        # Custom month order if plotting 'month'
        if var == "month":
            month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            x_axis = alt.X(var, title=var, sort=month_order)
        else:
            x_axis = alt.X(var, title=var)

        chart = alt.Chart(df_tooltip).mark_bar().encode(
            x=x_axis,
            y=alt.Y('count()', stack='normalize', title='Proportion'),
            color=alt.Color('y', title='Subscribed', sort=['no', 'yes']),
            order=alt.Order('y', sort='ascending'),
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
        # After plotting each imbalance plot:
        if var == "job":
            st.info("Certain jobs (e.g., management, retired and maybe suprisingly student) have higher subscription rates, but not by much.")
        elif var == "education":
            st.info("Higher education levels seem correlated with higher subscription rates.")
        elif var == "marital":
            st.info("Pretty uniform subscription rates across marital statuses, with slight variations.")
        elif var == "default":
            st.info("Clients with credit default have a little lower subscription rate.")
        elif var == "housing":
            st.info("Clients with housing loans have a little lower subscription rate.")
        elif var == "loan":
            st.info("Clients with personal loans have a little lower subscription rate.")
        elif var == "contact":
            st.info("Contact method affects subscription rates, with cellular contacts having a higher rate.")
        elif var == "month":
            st.info("Subscription rates vary some by month, with peaks in certain months (e.g., mar. sep. oct. and dec.).")
        elif var == "poutcome":
            st.info("Previous campaign outcomes strongly influence subscription rates, with 'success' leading to much higher rates.")


# Thoughts based on the visualizations


