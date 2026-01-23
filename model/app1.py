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
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # --------------------------------------------------
    # Train-test split (only for demo)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
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