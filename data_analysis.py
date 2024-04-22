

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt



def data_analysis(df):

    

# Interactive "About the Data" section
    st.subheader("About the Data")
    about_data_option = st.selectbox(
        "Select anu column",
        ["Select", "Data Shape", "Columns", "Description"],
        index=0  # Default to 'Select'
    )
    
    if about_data_option == "Data Shape":
        st.write("Shape of the Dataset:", df.shape)
    elif about_data_option == "Columns":
        st.write("Columns in the Dataset:", df.columns.tolist())
    elif about_data_option == "Description":
        st.write("Description of the Dataset:")
        st.dataframe(df.describe())


     # Interactive Null Values and DataType section
    st.subheader("Null Values and DataType of Columns")
    column_selected = st.selectbox(
        "Select a Column to Inspect", 
        ['Select a column'] + list(df.columns)
    )
    
    if column_selected != 'Select a column':
        dtype = df[column_selected].dtype
        num_nulls = df[column_selected].isnull().sum()
        st.write(f"Data Type of **{column_selected}** Column: {dtype}")
        st.write(f"Null Values in **{column_selected}** Column: {num_nulls}")

    # Interactive Examination of Categorical Columns
    st.subheader("Examine Categorical Columns")
    categorical_columns = df.select_dtypes(include=['object']).columns  # assuming 'object' dtype for categorical data
    cat_column_selected = st.selectbox(
    "Select a Categorical Column to Examine", 
    ['Select a column'] + list(categorical_columns)
    )

    if cat_column_selected != 'Select a column':
        st.write(f"Count of every unique value in **{cat_column_selected}** column:")
        value_counts = df[cat_column_selected].value_counts()
        st.write(value_counts)

    # Checkbox to display basic data information
    if st.checkbox('Show Data Overview'):
        st.subheader("Dataset Overview:")
        st.dataframe(df.head())

    # Missing values analysis
    if st.checkbox('Analyze Missing Values'):
        st.subheader("Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values)

        if st.button('Impute Missing Values'):
            for column in df.columns:
                if df[column].dtype in ['int64', 'float64']:
                    df[column] = df[column].fillna(df[column].median())
                else:
                    df[column] = df[column].fillna(df[column].mode()[0])
            st.success("Missing values imputed successfully.")
            st.write(df.isnull().sum())

        # Dataset statistics
        if st.checkbox('Show Dataset Statistics'):
            st.subheader("Dataset Statistics:")
            st.write(df.describe())

    # Feature importance analysis
    if st.checkbox('Analyze Feature Importance'):
        st.subheader("Feature Importance Analysis")
        numeric_df = df.select_dtypes(include=['int64', 'float64'])  # Assuming numeric data is relevant
        target_variable = 'Fraud_indicator'
        independent_variables = [col for col in numeric_df.columns if col != target_variable]

        st.write("Dependent Variable (Target):", target_variable)
        st.write("Independent Variables (Predictors):", independent_variables)

        X = numeric_df[independent_variables].dropna()
        y = df[target_variable].dropna()
        clf = RandomForestClassifier()
        clf.fit(X, y)

        feature_importance = pd.DataFrame({'Feature': independent_variables, 'Importance': clf.feature_importances_}).sort_values(by='Importance', ascending=False)
        st.bar_chart(feature_importance.set_index('Feature'))

    # Distribution of the target variable
    if st.checkbox('Show Distribution of Fraud Indicator'):
        st.subheader("Target Variable Distribution:")
        fraud_distribution = df[target_variable].value_counts()
        st.write(fraud_distribution)
        st.bar_chart(fraud_distribution)


    #DataVisualization
    st.subheader("Data Visualization")
    chart_type = st.selectbox("Choose a Visualization Type", ["Transaction Amount Distribution", "Fraud Cases by Vehicle Type", "Average Speed by Lane Type", "Transactions by TollBooth"])

    if chart_type == "Transaction Amount Distribution":
        if st.button("Show Distribution of Transaction Amounts"):
            fig, ax = plt.subplots()
            sns.histplot(df['Transaction_Amount'], kde=True, ax=ax)
            plt.title('Distribution of Transaction Amounts')
            st.pyplot(fig)

    elif chart_type == "Fraud Cases by Vehicle Type":
        if st.button("Show Fraud Cases by Vehicle Type"):
            fig, ax = plt.subplots()
            sns.countplot(x='Vehicle_Type', hue='Fraud_indicator', data=df)
            plt.title('Fraud Cases by Vehicle Type')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            plt.legend(title='Fraud Indicator')
            st.pyplot(fig)

    elif chart_type == "Average Speed by Lane Type":
        if st.button("Show Average Vehicle Speed by Lane Type"):
            fig, ax = plt.subplots()
            sns.barplot(x='Lane_Type', y='Vehicle_Speed', data=df, estimator=lambda x: sum(x) / len(x))
            plt.title('Average Vehicle Speed by Lane Type')
            plt.xlabel('Lane Type')
            plt.ylabel('Average Speed')
            st.pyplot(fig)

    elif chart_type == "Transactions by TollBooth":
        if st.button("Show Transactions by TollBooth"):
            fig, ax = plt.subplots()
            tollbooth_counts = df['TollBoothID'].value_counts()
            sns.barplot(x=tollbooth_counts.index, y=tollbooth_counts.values, ax=ax)
            plt.title('Number of Transactions by TollBooth')
            plt.xlabel('TollBooth ID')
            plt.ylabel('Transactions Count')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)

     # Data types and correlation analysis
    if st.checkbox('Show Data Types and Correlation Matrix'):
        st.subheader("Dataset Information:")
        st.write(df.dtypes)

        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        st.subheader("Dataset Correlation:")
        corr_matrix = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)