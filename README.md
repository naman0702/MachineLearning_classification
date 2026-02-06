# Comparative Analysis of Machine Learning Classification Models Using Performance Evaluation Metrics
**a) Problem Statement**
This Assignment aims to implement and evaluate six widely used machine learning classification models on a single dataset to ensure fair and consistent comparison.
The models include Logistic Regression, Decision Tree, K-Nearest Neighbors, Naive Bayes, Random Forest, and XGBoost.
Each model’s performance will be assessed using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).
The objective is to identify the most effective classification approach based on comprehensive metric-based evaluation.

**b) Data set Overview**
The dataset originates from the Breast Cancer Wisconsin (Diagnostic) Data Set, a widely used benchmark in machine learning for medical diagnostics.

**Data set Description**
The dataset contains 569 samples with 32 columns: one ID, one diagnosis label, and 30 numerical features extracted from digitized FNA images of breast masses.
The target variable Diagnosis indicates whether a tumor is Malignant (M) or Benign (B).
The 30 feature columns describe cell nuclei characteristics such as radius, texture, perimeter, and area.

**c) Table with the evaluation metrics calculated for all the 6 models as below**

| ML Model Name            | Accuracy            | AUC                | Precision           | Recall              | F1                 | MCC                |
|-------------------------|--------------------:|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|
| Logistic Regression     | 0.982456140350877  | 0.996031746031746 | 1.000000000000000 | 0.952380952380952 | 0.975609756097561 | 0.962621902223779 |
| Decision Tree           | 0.964912280701754  | 0.991111111111111 | 0.926829268292683 | 0.974358974358974 | 0.950000000000000 | 0.923683823893713 |
| kNN                     | 0.973684210526315  | 0.984273504273504 | 0.973684210526315 | 0.948717948717948 | 0.961038961038961 | 0.941357448663283 |
| Naive Bayes             | 0.973684210526315  | 0.989743589743589 | 0.950000000000000 | 0.974358974358974 | 0.962025316455696 | 0.942072735780544 |
| Random Forest (Ensemble)| 0.956140350877193  | 0.988034188034188 | 0.925000000000000 | 0.948717948717948 | 0.936708860759493 | 0.903329484741950 |
| XGBoost (Ensemble)      | 0.964912280701754  | 0.991111111111111 | 0.926829268292683 | 0.974358974358974 | 0.950000000000000 | 0.923683823893710 |

**d) Observations on the performance of each model**

| ML Model Name             | Observation about model performance |
|--------------------------|-------------------------------------|
| Logistic Regression      | Achieves the best overall performance with very high accuracy, perfect precision, excellent AUC, and strong F1 and MCC, indicating excellent class separation and balanced predictions. |
| Decision Tree            | Shows strong recall and high overall accuracy, capturing most malignant cases, but slightly lower precision leads to some false positives. |
| kNN                      | Provides well-balanced performance with high accuracy, precision, recall, and F1-score, indicating reliable and stable classification. |
| Naive Bayes              | Performs well in detecting malignant cases with high recall and good F1-score, though precision is slightly lower compared to kNN. |
| Random Forest (Ensemble) | Delivers reasonably strong performance but is weaker than other models in terms of overall balance and correlation. |
| XGBoost (Ensemble)       | Exhibits performance similar to Decision Tree with high recall, but does not outperform simpler models such as Logistic Regression or kNN. |
