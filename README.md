# K-Nearest-Neighbor-Classification-Mini-Project-Overview
Project Overview In this project, you will analyze a classification dataset using Python and Scikit-learn to build a K-Nearest Neighbor classifier. You will learn how to scale features, split data for training/testing, tune the k parameter, and evaluate predictions using standard classification metrics.

# K-Nearest Neighbor Classification (Pattern Recognition Mini-Project)

## 📌 Project Overview
This project implements an instance-based machine learning workflow using the **K-Nearest Neighbor (KNN)** algorithm to classify tumors as either malignant or benign using the classic **Breast Cancer Wisconsin Dataset**. 

The main objective is to understand how feature scaling, distance metrics, and hyperparameter tuning ($k$-value) directly influence model variance and bias.

---

## 📈 Key Findings & Performance Summary

Through 5-fold cross-validation on the scaled training dataset, **$k = 7$** (or $k=9$ depending on execution variance) was identified as the optimal number of neighbors, yielding a balanced trade-off between underfitting and overfitting.

### Final Classification Metrics (Holdout Test Set)
| Metric | Score | Analytical Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | ~96.5% | High overall rate of correct predictions across both classes. |
| **Precision** | ~95.9% | Out of all samples predicted as benign, 95.9% were actually benign. |
| **Recall (Sensitivity)** | ~98.6% | Crucial for medical diagnostics; minimizes missed positive cases (False Negatives). |
| **F1-Score** | ~97.2% | Harmonic mean proving robust balance between Precision and Recall. |

---

## 🧠 Core Machine Learning Concepts Applied

### 1. The Critical Role of Feature Scaling
KNN makes classifications by measuring the geometric distance (Euclidean distance) between data points:

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

If features have vastly different ranges (e.g., area vs. smoothness), the feature with the larger magnitude will entirely dominate the distance math. This project applies `StandardScaler` to normalize features to a mean of 0 and variance of 1, ensuring every physical property contributes equally to distance boundaries.

### 2. Tuning $k$ via Cross-Validation
* **Low $k$ (e.g., $k=1$):** High variance, low bias. The model forms highly complex decision boundaries around individual noise artifacts, causing overfitting.
* **High $k$ (e.g., $k=11$):** Low variance, high bias. The model smooths out local patterns and assumes the majority class, causing underfitting.
* By utilizing **Stratified 5-Fold Cross-Validation**, we ensure our hyperparameter choice generalizes well across five distinct variations of the training data.

---

## 💻 Repository Structure

```text
├── knn_classification.ipynb   # Executed Jupyter Notebook with code & visualizations
├── requirements.txt           # File listing required Python libraries
└── README.md                  # Project documentation (this file)
