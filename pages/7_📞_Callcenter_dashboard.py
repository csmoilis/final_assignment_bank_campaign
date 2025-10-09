import streamlit as st
from streamlit_extras.let_it_rain import rain
import requests
import random
import pandas as pd
import datetime


st.title("📞 Callcenter Dashboard")

with st.expander("ℹ️ - About this dashboard", expanded=False):
    st.markdown(
        """
        This dashboard simulates a call center environment where agents can manage a queue of customers to upsell a long term deposit bank product.
        In the original paper that came with the dataset, they mention that there was inbound calls too, but it's not present in the dataset.
        The dashboard fetches customer data from an API(NocoDB with test and synthetic data), displays customer information, and uses a machine learning model to predict the likelihood of a successful upsell.
        
        **How to use the dashboard:**
        1. Set the queue size and upsell bonus in the sidebar. The bonus is simply a multiplier for the potential earnings from successful upsells.
        2. View the current queue of customers and their details.
        3. For each customer, see the model's predicted probability of subscription.
        4. After each call, indicate whether the upsell was successful and submit the result.
        5. Track your total bonus based on successful upsells.

        **TIP** see what happens when the queue is empty 😉
        """
    )

# --- Sidebar: Set queue size and bonus, and show model probability ---
with st.sidebar:
    st.header("Queue Settings")
    queue_size = st.number_input("Queue size", min_value=1, max_value=50, value=10, step=1)
    bonus = st.number_input("Upsell Bonus (currency/unit)", min_value=1.0, value=10.0, step=1.0)
    if st.button("Reset Queue"):
        st.session_state.queue = None  # Force re-fetch
        st.session_state.total_bonus = 0.0
    # Placeholder for model probability
    model_prob_placeholder = st.empty()



# --- Cached data fetch ---
@st.cache_data(show_spinner=False)
def fetch_customers(limit):
    API_DATA_URL = "https://dun3co-sdc-nocodb.hf.space/api/v2/tables/m39a8axnn3980w9/records"
    API_DATA_TOKEN = st.secrets["NOCODB_TOKEN"]
    HEADERS = {"xc-token": API_DATA_TOKEN}
    params = {"offset": 0, "limit": limit, "viewId": "vwjuv5jnaet9npuu"}
    res = requests.get(API_DATA_URL, headers=HEADERS, params=params)
    res.raise_for_status()
    return res.json()["list"]

# --- Initialize or reset queue and bonus ---
if "queue" not in st.session_state or st.session_state.queue is None:
    records = fetch_customers(queue_size)
    st.session_state.queue = random.sample(records, len(records))
if "total_bonus" not in st.session_state:
    st.session_state.total_bonus = 0.0

# --- Calculate maximum potential bonus for the remaining queue ---
def get_max_potential_bonus(queue, bonus):
    if not queue:
        return 0.0, []
    API_MODEL_URL = "https://dun3co-marketing-lr-prediction.hf.space/predict"
    inputs = []
    for row in queue:
        inputs.append({
            "age": int(row["age"]),
            "balance": float(row["balance"]),
            "day": int(row["day"]),
            "campaign": int(row["campaign"]),
            "job": str(row["job"]),
            "education": str(row["education"]),
            "default": str(row["default"]),
            "housing": str(row["housing"]),
            "loan": str(row["loan"]),
            "months_since_previous_contact": str(row["months_since_previous_contact"]),
            "n_previous_contacts": str(row["n_previous_contacts"]),
            "poutcome": str(row["poutcome"]),
            "had_contact": bool(row["had_contact"]),
            "is_single": bool(row["is_single"]),
            "uknown_contact": bool(row["uknown_contact"]),
        })
    try:
        response = requests.post(API_MODEL_URL, json={"data": inputs})
        response.raise_for_status()
        probabilities = response.json()["probabilities"]
        max_bonus = sum((1 - p) * bonus for p in probabilities)
        return max_bonus, probabilities
    except Exception:
        return None, None

# --- 3. Show queue visually and bonus info ---
#st.subheader("Queue")

# Layout: queue info (left), bonus info (center), (right column left empty for centering)
queue_col, bonus_col, empty_col = st.columns([2, 1.2, 0.8])

with queue_col:
    st.subheader("Queue")
    for i, row in enumerate(st.session_state.queue):
        st.write(f"Position {i+1}: {row['job']} ({row['age']} yrs, {row['education']})")

# Calculate max potential bonus and get probabilities for queue
max_potential_bonus, queue_probabilities = get_max_potential_bonus(st.session_state.queue, bonus)

# --- 4. Simulate next call ---
if st.session_state.queue:
    st.subheader("Active Call")
    active_row = st.session_state.queue[0]

    # Use current day of month if possible, fallback to API day
    today_day = datetime.datetime.now().day
    try:
        day_value = int(today_day)
    except Exception:
        day_value = int(active_row["day"])

    # Prepare model input for active call
    input_row = {
        "age": int(active_row["age"]),
        "balance": float(active_row["balance"]),
        "day": day_value,
        "campaign": int(active_row["campaign"]),
        "job": str(active_row["job"]),
        "education": str(active_row["education"]),
        "default": str(active_row["default"]),
        "housing": str(active_row["housing"]),
        "loan": str(active_row["loan"]),
        "months_since_previous_contact": str(active_row["months_since_previous_contact"]),
        "n_previous_contacts": str(active_row["n_previous_contacts"]),
        "poutcome": str(active_row["poutcome"]),
        "had_contact": bool(active_row["had_contact"]),
        "is_single": bool(active_row["is_single"]),
        "uknown_contact": bool(active_row["uknown_contact"]),
    }
    payload = {"data": [input_row]}

    # --- 5. Get model prediction for active call ---
    API_MODEL_URL = "https://dun3co-marketing-lr-prediction.hf.space/predict"
    try:
        response = requests.post(API_MODEL_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        probability = result["probabilities"][0]
        # Show in sidebar
        model_prob_placeholder.metric("Model Probability (Subscribe)", f"{probability:.2%}")
    except Exception as e:
        st.error(f"Model API call failed: {e}")
        probability = None
        model_prob_placeholder.metric("Model Probability (Subscribe)", "N/A")

    # --- Customer info as tiles ---
    st.write("### Customer Information")
    keys = [k for k in active_row.keys() if k != "y"] #Dropping the target variable "y"
    values = [active_row[k] for k in keys] #Dropping the target variable "y"
    n_cols = 4
    cols = st.columns(n_cols)
    for i, key in enumerate(keys):
        col = cols[i % n_cols]
        with col:
            # Show the current day_value for the "day" field
            display_value = day_value if key == "day" else values[i]
            st.markdown(
                f"""
                <div style="
                    border: 2px solid #e6e6e6;
                    border-radius: 16px;
                    padding: 18px 10px 14px 10px;
                    margin-bottom: 1em;
                    background: linear-gradient(135deg, #f9f9f9 80%, #eaf6ff 100%);
                    box-shadow: 0 2px 8px 0 rgba(0,0,0,0.04);
                    min-height: 80px;
                    text-align: center;
                ">
                    <div style="font-size: 1.05em; font-weight: 600; color: #2c3e50; margin-bottom: 0.3em;">
                        {key.replace('_', ' ').capitalize()}
                    </div>
                    <div style="font-size: 1.15em; color: #0074d9;">
                        {display_value}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --- Bonus info and worker action column ---
    with bonus_col:
        st.markdown(
            """
            <div style="border:2px solid #e6e6e6; border-radius:14px; padding:18px 14px; background:#f8fbff; margin-bottom:1em;">
                <div style="font-size:1.2em; font-weight:700; margin-bottom:1em;">Bonus KPI's</div>
                <div style="font-size:1.1em; margin-bottom:0.7em;">
                    <b>Current Bonus:</b> <span style="color:#0074d9;">{current_bonus}</span>
                </div>
                <div style="font-size:1.1em; margin-bottom:0.7em;">
                    <b>Current Call Bonus:</b> <span style="color:#28a745;">{current_call_bonus}</span>
                </div>
                <div style="font-size:1.1em;">
                    <b>Max Potential Bonus:</b> <span style="color:#ff851b;">{max_potential_bonus}</span>
                </div>
            </div>
            """.format(
                current_bonus=f"{st.session_state.total_bonus:.2f}",
                current_call_bonus=f"{(1 - probability) * bonus:.2f}" if probability is not None else "N/A",
                max_potential_bonus=f"{max_potential_bonus:.2f}" if max_potential_bonus is not None else "N/A"
            ),
            unsafe_allow_html=True,
        )

        # Plain Streamlit widgets for worker action (no custom styling)
        st.subheader("Callcenter Worker Action")
        upsell = st.radio("Did you upsell?", options=["Yes", "No"], key="upsell_radio", horizontal=True)
        submit = st.button("Submit", disabled=not st.session_state.queue, key="upsell_submit")

        if submit:
            if upsell == "Yes" and probability is not None:
                st.session_state.total_bonus += (1 - probability) * bonus
            st.session_state.queue.pop(0)
            st.rerun()

else:
    rain(emoji="💸", font_size=54, falling_speed=5, animation_length="infinite")
    st.success("Queue is empty! All calls handled.")
    st.markdown(
        f"""
        <div style="border:2px solid #e6e6e6; border-radius:14px; padding:18px 14px; background:#f8fbff; margin-bottom:1em;">
            <div style="font-size:1.2em; font-weight:700; margin-bottom:1em;">Total Bonus Earned</div>
            <div style="font-size:2em; color:#0074d9; text-align:center;">
                {st.session_state.total_bonus:.2f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
