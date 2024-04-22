import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def train_models(df, selected_models):
    st.subheader("Model Training")
    # Select features and target variable
    selected_features = ['Transaction_Amount', 'Amount_paid']

    # Check if 'Fraud_indicator' is present in the dataset
    if 'Fraud_indicator' not in df.columns:
        st.error("Error: 'Fraud_indicator' column not found in the dataset.")
        return

    # Check if there are samples from both classes (0 and 1) in the target variable
    if len(df['Fraud_indicator'].unique()) < 2:
        st.error("Error: Dataset contains only one class in 'Fraud_indicator'. Ensure there are samples from both classes.")
        return

    X = df[selected_features]
    y = df['Fraud_indicator']

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Store scaler in session state
    st.session_state.scaler = scaler

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Neural Network': None
    }

    model_results = {}  # Dictionary to store evaluation metrics for each model

    for model_name in selected_models:
        st.write(f"### {model_name} Evaluation Metrics:")
        if model_name == 'Neural Network':
            # Neural network model
            model = Sequential()
            model.add(Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

            y_pred_prob = model.predict(X_test_scaled)
            y_pred = np.round(y_pred_prob)

        else:
            model = models[model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Store evaluation metrics in dictionary
        model_results[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

        # Display model evaluation metrics
        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1 Score: {f1:.2f}")

        # Plot accuracy metrics
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        values = [accuracy, precision, recall, f1]
        plt.figure()
        plt.bar(metrics, values, color=['blue', 'green', 'orange', 'red'])
        plt.ylabel('Score')
        plt.title(f'{model_name} Model Metrics')
        plt.ylim(0, 1)
        st.pyplot(plt)
        st.write("---")

    # Find and save the best performing model among the selected models
    best_model_name = max(selected_models, key=lambda x: model_results[x]['Accuracy'])
    best_model_metrics = model_results[best_model_name]
    st.session_state.best_model_name = best_model_name
    st.session_state.best_model_metrics = best_model_metrics

    if best_model_name == 'Neural Network':
        st.session_state.best_model = model
    else:
        st.session_state.best_model = models[best_model_name]

    st.write("Best model saved to session state.")
