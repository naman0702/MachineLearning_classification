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

ML Model Name	Accuracy	AUC	Precision	Recall	F1	MCC
Logistic Regression	0.982456140350877	0.996031746031746	1.000000000000000	0.952380952380952	0.975609756097561	0.962621902223779
Decision Tree	0.964912280701754	0.991111111111111	0.926829268292683	0.974358974358974	0.950000000000000	0.923683823893713
kNN	0.973684210526315	0.984273504273504	0.973684210526315	0.948717948717948	0.961038961038961	0.941357448663283
Naive Bayes	0.973684210526315	0.989743589743589	0.950000000000000	0.974358974358974	0.962025316455696	0.942072735780544
Random Forest (Ensemble)	0.956140350877193	0.988034188034188	0.925000000000000	0.948717948717948	0.936708860759493	0.903329484741950
XGBoost (Ensemble)	0.964912280701754	0.991111111111111	0.926829268292683	0.974358974358974	0.950000000000000	0.923683823893713

<img width="3053" height="267" alt="image" src="https://github.com/user-attachments/assets/9ae75b29-b36b-4f30-a56f-21907a3fb0ae" />

**Observations on the performance of each model**
Logistic Regression
Logistic Regression achieves the best overall performance with an Accuracy of 98.25%, AUC of 0.996, Precision of 1.00, Recall of 0.952, F1-score of 0.976, and MCC of 0.963. These values indicate excellent class separation and highly balanced predictions.

Decision Tree
The Decision Tree model records an Accuracy of 96.49%, AUC of 0.991, Precision of 0.927, Recall of 0.974, F1-score of 0.950, and MCC of 0.924. It shows strong recall but slightly lower precision, meaning it captures most malignant cases but with some false positives.

k-Nearest Neighbors (kNN)
kNN achieves an Accuracy of 97.37%, AUC of 0.984, Precision of 0.974, Recall of 0.949, F1-score of 0.961, and MCC of 0.941. These results indicate a well-balanced and reliable classification performance.

Naive Bayes
Naive Bayes reports an Accuracy of 97.37%, AUC of 0.990, Precision of 0.950, Recall of 0.974, F1-score of 0.962, and MCC of 0.943. It performs well in identifying malignant cases, though precision is slightly lower than kNN.

Random Forest (Ensemble)
Random Forest obtains an Accuracy of 95.61%, AUC of 0.988, Precision of 0.925, Recall of 0.949, F1-score of 0.937, and MCC of 0.903. While reasonably strong, it performs weaker than most other models in overall correlation and balance.

XGBoost (Ensemble)
XGBoost shows an Accuracy of 96.49%, AUC of 0.991, Precision of 0.927, Recall of 0.974, F1-score of 0.950, and MCC of 0.924. Its performance is similar to the Decision Tree and does not exceed simpler models such as Logistic Regression or kNN.

<img width="3794" height="267" alt="image" src="https://github.com/user-attachments/assets/1dc33a2a-2fda-4a04-bcaa-3804f9f1fc5b" />
