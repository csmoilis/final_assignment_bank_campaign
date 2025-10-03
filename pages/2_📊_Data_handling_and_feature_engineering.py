import streamlit as st
import time
import numpy as np

st.set_page_config(page_title="Data handling and feature", page_icon="📊")

container1 = st.container(border=True, vertical_alignment="center")
container2 = st.container(border=True, vertical_alignment="center")

with container1:
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

                    
        Lastly there's campaign attributes. There's been a previous campaign (later we will see, some sample imbalance because of this) and the campaign feature captures the number of contacts during the campaign. Pdays is the number of days since the customer was last contacted, previous the number of times, and poutcome the outcome of the previous campaign.
        
        Importantly, the pdays is -1 if the customer has not been contacted before, meaning that a -1 in this feature is different from any other value, and we will later engineer a feature to capture this.
                    
        ---
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
                    


    """)
with container2:
    col1, col2 =st.columns([0.4, 0.6])
    with col1:
        st.markdown("""
        # Feature engineering
                    
        We did some feature engineering to make the data more suitable for modeling.
        as mentioned earlier, we created a datetime variable from the day, month and year (inferred from the order of the data).
        
        """)

    with col2:
        st.markdown("""
        ### Feature Engineering

        | Feature Name                  | Description                                                                                       |
        |-------------------------------|---------------------------------------------------------------------------------------------------|
        | `months_since_previous_contact` | Binned version of `pdays` into intervals (e.g., "No contact", "0 - 5 months", etc.)              |
        | `n_previous_contacts`           | Binned version of `previous` into categories ("No contact", "1", ..., "More than 6")             |
        | `had_contact`                   | Boolean: True if client had previous contact (`months_since_previous_contact` ≠ "No contact")    |
        | `is_single`                     | Boolean: True if marital status is "single"                                                      |
        | `uknown_contact`                | Boolean: True if contact type is "unknown"                                                       |
        | `month_num`                     | Numeric month extracted from categorical `month`                                                  |
        | `year`                          | Year inferred from campaign sequence                                                             |
        | `date`                          | Combined datetime column from day, month, and year                                               |
        | `year_moth`                     | Year and month as datetime for time-based splitting                                              |
        | `balance` (capped)              | Capped at 99th percentile to reduce outlier impact                                               |
        | `campaign` (capped)             | Capped at 90th percentile for distribution analysis                                              |

        These features were created to improve model interpretability, handle outliers, and enable time-based splits.
                """)
