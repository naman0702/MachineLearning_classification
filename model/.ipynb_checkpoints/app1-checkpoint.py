import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="ML Model Evaluation App", layout="wide")
st.title("📊 Machine Learning Model Evaluation Dashboard")

st.markdown(
    """
    Upload **test data only (CSV)** and evaluate different classification models.
    """
)

# (a) Dataset upload (CSV)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # Assume last column is target
    # --------------------------------------------------
    # Step 3: Drop unnecessary columns
    df.drop(columns=["id", "Unnamed: 32"], inplace=True)
    
    # Step 4: Encode target variable
    # M = 1 (Malignant), B = 0 (Benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}) #we are changing M and B to o,1 as in ML world string won't help.
    
    # Step 5: Separate features and target
    X = df.drop("diagnosis", axis=1) #we don't want this as an input as we have to predict this.
    y = df["diagnosis"]
    
    #print(y)
    # Step 6: Train-test split 20% test 80% training and random sample is 50
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50, stratify=y
    )

    # --------------------------------------------------
    # (b) Model selection dropdown
    # --------------------------------------------------
    model_name = st.selectbox(
        "🤖 Select Classification Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors"
        ]
    )