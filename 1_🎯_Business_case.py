import streamlit as st
import pandas as pd
import numpy as np


st.set_page_config(page_title="Banking marketing campaign", page_icon=":phone:", layout="wide")

st.markdown("""
# Business problem:
### We're a bank trying to reach potential customers to subscribe to a term deposit.

Currently, the bank spends over **$220,000** contacting customers who ultimately **do not purchase** the term deposit — indicating inefficiency in campaign targeting.  

Our project focuses on improving the **return on investment (ROI)** of these marketing efforts by:  
- Identifying which customer segments are most likely to subscribe.  
- Reducing wasted expenditure on customers unlikely to convert.  
- Simulating different cost and profit scenarios to evaluate potential improvements.  

The model uses a dataset that includes:  
- **Demographics:** age, marital status, education level.  
- **Financial details:** account balance, housing loans, personal loans.  
- **Campaign history:** number of contacts, previous outcomes, and time since last contact.  
- **Call information:** month and duration of the last contact.  

By predicting the **probability of a new customer subscribing**, the model supports better targeting decisions.  
It also informs the **call centre incentive structure** — offering higher bonuses for converting low-probability customers, encouraging efficiency and motivation among staff.  

It's a direct to consumer marketing case, and our plan is to most effectively use our marketing budget.

# Stakeholders
### - Management: Want to maximise ROI by understanding where marketing spend generates the most value and how to allocate resources effectively.  
### - Marketing team leaders: Seek insights into which customer segments and campaign strategies yield higher conversion rates, enabling smarter decision-making and more efficient campaigns.  
### - Call centre staff: Use model predictions to prioritise calls, improve success rates, and align bonuses with the difficulty of conversion — ensuring fair rewards for high-effort sales.
            
            """)

