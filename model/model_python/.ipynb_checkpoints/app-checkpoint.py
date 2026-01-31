import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#importing python file functions.
from logistic import logistic_model
from decision_tree import decision_classifier
from knn import knn_classifier
from naiveBayes import nb_classifier
from random_forest import rf_classifier
from XGB import xgb_classifier
from classification import preprocess_data

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Model Evaluation Dashboard",
    layout="wide"
)

# --------------------------------------------------
# Header
# --------------------------------------------------
st.title("📊 Machine Learning Model Evaluation Dashboard")
st.caption("Upload test dataset (CSV) and evaluate different classification models.")

st.divider()

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("⚙ Settings")

    uploaded_file = st.file_uploader(
        "📂 Upload Test Dataset (CSV)",
        type=["csv"]
    )

    model_name = st.selectbox(
        "🤖 Select Classification Model",
        [
            "Select the Model",
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors",
            "Navie Bayes Classification",
            "Random Forest",
            "XGB Classifier"
        ]
    )

# --------------------------------------------------
# Main Area
# --------------------------------------------------
result = None
y_test = None
y_pred = None

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    # Dataset Preview
    with st.expander("🔍 Preview Dataset"):
        st.dataframe(df, use_container_width=True)

    # Preprocessing
    X, y, target_column = preprocess_data(df, target_column=None)

    # --------------------------------------------------
    # Model Training
    # --------------------------------------------------
    if model_name != "Select the Model":

        st.subheader(f"🔹 {model_name}")

        with st.spinner("Training model..."):

            if model_name == "Logistic Regression":
                result, y_test, y_pred = logistic_model(X, y)

            elif model_name == "Decision Tree":
                result, y_test, y_pred = decision_classifier(X, y)

            elif model_name == "K-Nearest Neighbors":
                result, y_test, y_pred = knn_classifier(X, y)

            elif model_name == "Navie Bayes Classification":
                result, y_test, y_pred = nb_classifier(X, y)

            elif model_name == "Random Forest":
                result, y_test, y_pred = rf_classifier(X, y)

            elif model_name == "XGB Classifier":
                result, y_test, y_pred = xgb_classifier(X, y)

        st.success("🎉 Model trained successfully!")

        # --------------------------------------------------
        # Metrics Cards
        # --------------------------------------------------
        st.subheader("📈 Evaluation Metrics")

        c1, c2, c3, c4, c5,c6 = st.columns(6)

        c1.metric("Accuracy", f"{result['Accuracy']:.3f}")
        c2.metric("Precision", f"{result['Precision']:.3f}")
        c3.metric("Recall", f"{result['Recall']:.3f}")
        c4.metric("F1 Score", f"{result['F1']:.3f}")
        c5.metric("AUC", f"{result['AUC']:.3f}")
        c6.metric("MCC", f"{result['MCC']:.3f}")

        st.divider()

        # --------------------------------------------------
        # Confusion Matrix + Report
        # --------------------------------------------------
        st.subheader("📊 Confusion Matrix & Classification Report")

        col1, col2 = st.columns(2)

        # Confusion Matrix
        with col1:
            fig, ax = plt.subplots(figsize=(5,4))
            cm = confusion_matrix(y_test, y_pred)

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax
            )

            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            ax.set_title("Confusion Matrix")

            st.pyplot(fig, use_container_width=True)

        # Classification Report
        with col2:
            st.markdown("### 📄 Classification Report")
            st.code(classification_report(y_test, y_pred))

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
