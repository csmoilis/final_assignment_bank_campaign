import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Data handling and feature", page_icon="📊")
col1, col2 =st.columns([0.4, 0.6])

with col1:
    st.markdown("""
    # Explanation of dataset
    The dataset can be broken down into three main components.
                
    There's the client attributes. These are the standard information the bank would have, presuming that we're calling our existing customer base, and not cold-calling for new customers.
    It's mainly categorical features, with age and balance being numerical.
    There are some categorical values that are almost bolean with nan/unknow values.
    
    Then we have our contact attributes. These are features regarding how a customer was contacted, on what day and which month, and for how long the contact was.
    We checked the dataset, and found that the collection had found place from may 2008 to november 2010. Quickly we realized that this suggest the data has some temporal features to it. As we can see we have the month and the day of month as variables. We know from the dataset that it started may 2008, and is in descending order,
    so the year is implicitly in the data, with the first grouping of may belonging to 2008, the next to 2009 etc. for every month. From that we can build a datetime variable.
    **NOTE HERE ABOUT DURATION AND LEAKAGE**            

                
    Lastly there's campaign attributes, 
                """)

with col2:
    st.markdown("""

            """)

    st.markdown("""

## Original data
### Client Atributes

| Column      | Description                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------- |
| `age`       | Age of the client (numeric).                                                                   |
| `job`       | Type of job (categorical). Examples: `admin.`, `technician`, `blue-collar`, `management`, etc. |
| `marital`   | Marital status (categorical). Values: `married`, `single`, `divorced`.                         |
| `education` | Education level (categorical). Values: `primary`, `secondary`, `tertiary`, `unknown`.          |
| `default`   | Has credit in default? (categorical). Values: `yes`, `no`, `unknown`.                          |
| `balance`   | Average yearly balance in euros (numeric).                                                     |
| `housing`   | Has a housing loan? (categorical). Values: `yes`, `no`, `unknown`.                             |
| `loan`      | Has a personal loan? (categorical). Values: `yes`, `no`, `unknown`.                            |

### Contact attributes

| Column     | Description                                                                                                                |
| ---------- | -------------------------------------------------------------------------------------------------------------------------- |
| `contact`  | Communication type (categorical). Values: `cellular`, `telephone`.                                                         |
| `day`      | Last contact day of the month (numeric).                                                                                   |
| `month`    | Last contact month of the year (categorical). Values: `jan`, `feb`, `mar`, etc.                                            |
| `duration` | Last contact duration, in seconds (numeric).                                                                               |

### Campaing Attributes

| Column     | Description                                                                                                                           |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `campaign` | Number of contacts performed during this campaign for this client (numeric, includes last contact).                                   |
| `pdays`    | Number of days that passed after the client was last contacted in a previous campaign (-1 means client was not previously contacted). |
| `previous` | Number of contacts performed before this campaign for this client (numeric).                                                          |
| `poutcome` | Outcome of the previous marketing campaign (categorical). Values: `success`, `failure`, `other`, `unknown`.                           |
                

## New features
                
| Column      | Description                                                                                    |
| ----------- | ---------------------------------------------------------------------------------------------- |
| `age`       | Age of the client (numeric).                                                                   |
| `job`       | Type of job (categorical). Examples: `admin.`, `technician`, `blue-collar`, `management`, etc. |
| `marital`   | Marital status (categorical). Values: `married`, `single`, `divorced`.                         |
| `education` | Education level (categorical). Values: `primary`, `secondary`, `tertiary`, `unknown`.          |
| `default`   | Has credit in default? (categorical). Values: `yes`, `no`, `unknown`.                          |
| `balance`   | Average yearly balance in euros (numeric).                                                     |
| `housing`   | Has a housing loan? (categorical). Values: `yes`, `no`, `unknown`.                             |
| `loan`      | Has a personal loan? (categorical). Values: `yes`, `no`, `unknown`.                            |
""")
