import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def knn_classifier(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=40
    )
    scaler = StandardScaler() # Scaling (VERY IMPORTANT for KNN)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Create model
    model = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="euclidean"
    )
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Step 10: Evaluation Metrics
    baseline_metrics ={
    "accuracy": accuracy_score(y_test, y_pred),
    "auc": float(roc_auc_score(y_test, y_prob)),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "mcc": float(matthews_corrcoef(y_test, y_pred))
    }
    return baseline_metrics, y_test, y_pred