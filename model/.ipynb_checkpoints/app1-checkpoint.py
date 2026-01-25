import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from logistic import logistic_model
from decision_tree import decision_classifier

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
result = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    # --------------------------------------------------
    # Assume last column is target
    # --------------------------------------------------
    # Step 1: Drop unnecessary columns
    df.drop(columns=["id", "Unnamed: 32"], inplace=True)
    
    # Step 2: Encode target variable
    # M = 1 (Malignant), B = 0 (Benign)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}) #we are changing M and B to o,1 as in ML world string won't help.
    
    # Step 3: Separate features and target
    X = df.drop("diagnosis", axis=1) #we don't want this as an input as we have to predict this.
    y = df["diagnosis"]
    
    #print(y)
    # Step 6: Train-test split 20% test 80% training and random sample is 50
    #X_train, X_test, y_train, y_test = train_test_split(
    #    X, y, test_size=0.2, random_state=50, stratify=y
    #)

    # --------------------------------------------------
    # (b) Model selection dropdown
    # --------------------------------------------------
    model_name = st.selectbox(
        "🤖 Select Classification Model",
        [
            "Select the Model",
            "Logistic Regression",
            "Decision Tree",
            "K-Nearest Neighbors"
            "Navie Bayes Classification"
            "Random Forest"
            "XGB Classifier"
        ]
    )
    y_test = None
    y_pred = None
    if model_name == "Logistic Regression":
        output = logistic_model(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    elif model_name == "Decision Tree":
        output = decision_classifier(X,y)
        result = output[0]
        y_test = output[1]
        y_pred = output[2]
    
    if model_name != "Select the Model":#printing model 
        st.subheader("📈 Evaluation Matrix")
        #st.write(logistic_model(X,y))
        if result is not None:
            dataf = pd.DataFrame(
            result.items(),
            columns=["Metric", "Value"]
            )
            st.dataframe(dataf)        
        #printing confusion matrix.
        st.subheader("📊 Confusion Matrix")
        col1,spacer, col2 = st.columns([1,0.3,2])
        cm = confusion_matrix(y_test, y_pred)
        with col1:
            fig, ax = plt.subplots()
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
            st.pyplot(fig)

        with col2:
            #Classification Report
            st.subheader("📄 Classification Matrix")
            st.text(classification_report(y_test, y_pred))
    
   
