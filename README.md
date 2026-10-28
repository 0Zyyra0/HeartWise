# 💓 HeartWise — Heart Disease Prediction with Machine Learning

> Final project for the **Machine Learning with Python** course (Maktabkhooneh)

## Summary

HeartWise is an end-to-end machine learning project in Python built on the classic heart disease dataset (303 patients, 13 clinical features). It covers every chapter of the course: preprocessing, regression, classification, clustering and a recommender system. The best model (Random Forest) reached about **82% test accuracy and ~0.91 ROC AUC**.

## Description

The main goal is to predict whether a patient has heart disease (`output = 1`) from features such as age, sex, chest pain type, blood pressure, cholesterol and ECG results. The project also shows how each course chapter applies to a real dataset:

| Course chapter | What was done in the project |
|---|---|
| 1. Introduction & preprocessing | Checked missing values, removed one duplicate row, standardized features, stratified train/test split |
| 2. Regression | Linear regression to predict maximum heart rate (`thalachh`) |
| 3. Classification | Logistic Regression, KNN, Decision Tree and Random Forest with GridSearchCV and 5-fold cross-validation |
| 4. Clustering | K-Means (Elbow and Silhouette methods) and Hierarchical Clustering, visualized with PCA |
| 5. Recommender system | "Similar patients" lookup using nearest neighbours (content-based) |

## Dataset

`data/heart.csv` has 303 rows and 14 columns (one duplicate row was removed, leaving 302; there are no missing values).

| Column | Meaning |
|---|---|
| age | Age in years |
| sex | Sex (1 = male, 0 = female) |
| cp | Chest pain type (0–3) |
| trtbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG result (0–2) |
| thalachh | Maximum heart rate achieved |
| exng | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise relative to rest |
| slp | Slope of the peak exercise ST segment |
| caa | Number of major vessels colored by fluoroscopy (0–4) |
| thall | Thallium stress test result |
| **output** | **Target: 1 = heart disease, 0 = healthy** |

## Results

### Classification (test set, 61 samples)

| Model | CV accuracy | Test accuracy | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.826 | 0.787 | 0.879 | 0.817 | 0.865 |
| KNN (k=17) | 0.830 | 0.803 | 0.939 | 0.838 | 0.900 |
| Decision Tree | 0.789 | 0.787 | 0.939 | 0.827 | 0.880 |
| **Random Forest** | **0.851** | **0.820** | 0.939 | **0.849** | **0.913** |

Top Random Forest features: `cp` (0.21), `thall` (0.15), `caa` (0.13), `oldpeak` (0.13) and `thalachh` (0.10).

### Regression
Predicting `thalachh` with linear regression gives R² ≈ 0.43 and a mean absolute error of about 15 bpm. The linear model explains part of the variation but is not precise.

### Clustering
With K=2, both methods split the data into a high-risk and a low-risk group without seeing the labels:
- K-Means: 77.6% disease rate in one cluster vs 11.3% in the other
- Hierarchical: 70.6% vs 16.5%

The silhouette score is low (about 0.17 for K=2), so the clusters are not cleanly separated.

### Recommender (similar patients)
For each patient, the 5 most similar patients are retrieved. In 81% of cases the majority label among the neighbours matched the patient's own label.

## Notes and limitations

- The dataset is small (302 samples) and the test set has only 61 samples, so differences of a few percent between models may be due to chance. Cross-validation accuracy is more reliable than test accuracy.
- This is an educational project and **not a medical diagnostic tool**.
- Hyperparameters were tuned on the training set only to avoid data leakage.

## Project structure

```
HeartWise/
├── data/heart.csv                    # dataset
├── main.py                           # the whole project (all chapters)
├── README.md
└── results/
    ├── results.json                  # all metrics
    ├── classification_results.csv
    ├── similar_patients_example.csv
    └── *.png                         # plots
```

## How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python main.py
```

Metrics and plots are saved to the `results/` folder.

## Tools

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn
