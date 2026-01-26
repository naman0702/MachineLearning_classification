import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df, target_column=None):
    df = df.copy()

    # ---------------------------
    # 1. Drop ID-like columns
    # ---------------------------
    drop_cols = [c for c in df.columns 
                 if "id" in c.lower() or "unnamed" in c.lower()]
    df.drop(columns=drop_cols, inplace=True)

    # ---------------------------
    # 2. Auto-detect target column
    # ---------------------------
    if target_column is None:

        # Common target names
        common_targets = [
            "target", "label", "class", "outcome", "y",
            "diagnosis", "default", "churn", "species"
        ]

        for col in df.columns:
            if col.lower() in common_targets:
                target_column = col
                break

        # If still not found → assume last column
        if target_column is None:
            target_column = df.columns[-1]

    # ---------------------------
    # 3. Separate X and y
    # ---------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ---------------------------
    # 4. Encode target if categorical
    # ---------------------------
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # ---------------------------
    # 5. One-hot encode categorical features
    # ---------------------------
    X = pd.get_dummies(X, drop_first=True)

    return X, y, target_column