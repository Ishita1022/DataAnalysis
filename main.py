import streamlit as st
import pandas as pd
from data_analysis import data_analysis
from train_models_impl import train_models
from predict_fraud import predict_fraud
from preprocess_data import preprocess_data

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('data/FastagFraudDetection.csv', parse_dates=['Timestamp'])
    return df

def main():
    
    # Set page configuration
    # st.set_page_config(page_title="Exploratory Data Analysis", page_icon="", layout="centered")

    # Custom CSS for styling
    st.markdown(
        """
        <style>
        body {
            background-color: #FDEFC1;
            background-size: cover;
        }
       [data-testid=stSidebar] {
        background-color: rgb(65,163,175);
         }
        .stApp {
            background-color: rgb(225,202,105);
            color: black;
        }
        /* Adjusting headers and markdown containers */
        .markdown-text-container, .stMarkdown {
            background-color: rgb(65,163,175); /* Blue background */
            color: white; /* White text color */
            border-radius: 10px;
            padding: 10px;
            box-shadow: 2px 2px 5px grey;
            margin-bottom: 20px; /* Add space below each markdown section */
        }
        h1, h2, h3, h4, h5, h6 {
            margin-top: 20px; /* Adds space above each header */
            margin-bottom: 20px; /* Adds space below each header */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    
    # Load dataset
    df = load_data()
    
    # Initialize session state variables if they are not already set
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'best_model_metrics' not in st.session_state:
        st.session_state.best_model_metrics = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = []

    # Navigation
    page = st.sidebar.selectbox("Select Page", ["Home", "Data Analysis", "Model Training", "Fraud Prediction"])

    if page == "Home":
        # Title and Sub-Title
        st.markdown("<div border-radius:50px;'><h1 style='text-align:center; color:white;'>Exploratory Data Analysis</h1></div>", unsafe_allow_html=True)

        st.image("fastag.jpg", use_column_width=True)
        st.markdown("""
            ## Problem Statement
            This app is designed to address the problem of fraud in Fastag transactions, 
            a critical issue in the efficient operation of toll collections in automated systems. 
            By leveraging data analysis, model training, and fraud prediction capabilities, 
            this app helps in identifying potential fraudulent activities and provides insights to mitigate these risks.

            ## Project Description
            The project uses Exploratory Data Analysis (EDA) to uncover patterns and anomalies in the data, followed by predictive modeling to identify fraudulent transactions. 
            Users can interactively explore data, train various models, and use them to predict fraud cases.

            ## Regression Models Used
            Several machine learning models are employed to predict fraud:
            - **Logistic Regression:** Useful for binary classification of fraud vs. non-fraud cases.
            - **Decision Tree:** Provides clear decision paths and is easy to interpret.
            - **Random Forest:** An ensemble of decision trees, enhancing the prediction accuracy and overfitting resistance.
            - **K-Nearest Neighbors (KNN):** Uses feature similarity to predict fraud instances.
            - **Neural Network:** A more complex model that can capture non-linear relationships in the data.
        """, unsafe_allow_html=True)

    elif page == "Data Analysis":
        data_analysis(df)

    elif page == "Model Training":
        available_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'KNN', 'Neural Network']
        selected_models = st.multiselect("Select Models to Train", available_models, default=st.session_state.selected_models)

        if selected_models:
            st.session_state.selected_models = selected_models
            trained_models = train_models(df, selected_models)
        else:
            st.warning("Please select at least one model from the list.")

    elif page == "Fraud Prediction":
        predict_fraud()

    # Optional: Button to clear session state
    if st.sidebar.button('Reset Application'):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.experimental_rerun()

if __name__ == "__main__":
    main()
