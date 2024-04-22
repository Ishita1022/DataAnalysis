import streamlit as st
import pandas as pd
import numpy as np

def predict_fraud():
    st.subheader("Predict Fraud")
    st.write("Use the best performing model to predict potential fraud cases.")

    # Verify if the model has been trained and is stored in the session state
    if (st.session_state.get('best_model_name') is None or 
        st.session_state.get('best_model') is None or 
        st.session_state.get('best_model_metrics') is None):
        st.warning("No trained model available. Please train the model first.")
        return

    # Displaying the best model and its metrics
    st.write(f"Best Performing Model: {st.session_state['best_model_name']}")
    st.write("Evaluation Metrics:")
    for metric, value in st.session_state['best_model_metrics'].items():
        st.write(f"{metric}: {value:.2f}")

    # Input fields for the user to provide data for prediction
    transaction_amount = st.number_input("Transaction Amount", min_value=0, max_value=10000, value=1000, key="transaction_amount")
    amount_paid = st.number_input("Amount Paid", min_value=0, max_value=10000, value=500, key="amount_paid")

    # Button to initiate prediction
    if st.button('Predict Fraud'):
        # Assembling input data in the correct format
        input_data = pd.DataFrame({
            'Transaction_Amount': [transaction_amount],
            'Amount_paid': [amount_paid]
        })

        # Making the prediction using the stored model
        if st.session_state.best_model_name == 'Neural Network':
            if 'scaler' not in st.session_state:
                st.error("Scaler not found. Please train the model again.")
                return
            scaler = st.session_state.scaler
            input_data_scaled = scaler.transform(input_data)
            prediction_prob = st.session_state.best_model.predict(input_data_scaled)
            prediction = np.round(prediction_prob)
        else:
            prediction = st.session_state.best_model.predict(input_data)

        # Print the prediction value for troubleshooting
        st.write(f"Prediction Value: {prediction[0]}")

        # Checking if the model provides probability estimates and obtaining them
        if hasattr(st.session_state.best_model, 'predict_proba'):
            probability = st.session_state.best_model.predict_proba(input_data)[0, 1]
            proba_text = f"Fraud Probability: {probability:.2%}"
        else:
            probability = 'N/A'
            proba_text = "Fraud Probability: Not Available"

        # Displaying the prediction and probability
        pred_text = " Fraudulent" if prediction[0] == 0 else "Not Fraudulent"
        st.write(f"Prediction: **{pred_text}**")
        st.write(proba_text)

        # Optional: Display feature importances or model insights for interpretability
        # This could involve using SHAP values or other methods to explain the prediction
