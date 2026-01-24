
from sklearn.linear_model import LogisticRegression

def logistic_model(X,y):
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50, stratify=y
    )
    # Step 7: Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) #fit and transform --> Fit -> Mean & std Tansform --> feature scaling training data.
    X_test = scaler.transform(X_test) #Apply the same feature scaling on test data.
    # Step 8: Train Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Step 9: Predictions
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
    return baseline_metrics