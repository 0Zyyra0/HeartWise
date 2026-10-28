"""
Heart Disease ML Project - covers the 5 chapters of the course:
1) preprocessing  2) regression  3) classification  4) clustering  5) recommender
Run:  python main.py
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score,
                             mean_absolute_error, precision_score, r2_score,
                             recall_score, roc_auc_score, roc_curve,
                             silhouette_score)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RS = 42
ROOT = Path(__file__).parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
results = {}

# ---------------------------------------------------------------- Chapter 1
df = pd.read_csv(ROOT / "data" / "heart.csv")
results["rows_raw"] = len(df)
results["missing_values"] = int(df.isna().sum().sum())
results["duplicates"] = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)
results["rows_clean"] = len(df)
results["class_balance"] = df["output"].value_counts().to_dict()

plt.figure(figsize=(9, 7))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
plt.title("Correlation matrix")
plt.tight_layout(); plt.savefig(OUT / "01_correlation.png", dpi=130); plt.close()

X = df.drop(columns="output")
y = df["output"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

# ---------------------------------------------------------------- Chapter 2
# Regression: predict max heart rate (thalachh) from the other features
Xr = df.drop(columns=["thalachh", "output"])
yr = df["thalachh"]
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=RS)
lr = make_pipeline(StandardScaler(), LinearRegression()).fit(Xr_tr, yr_tr)
pred = lr.predict(Xr_te)
results["regression"] = {
    "target": "thalachh (max heart rate)",
    "R2": round(r2_score(yr_te, pred), 3),
    "MAE_bpm": round(mean_absolute_error(yr_te, pred), 2),
}
plt.figure(figsize=(5, 5))
plt.scatter(yr_te, pred, alpha=.7)
lim = [yr.min(), yr.max()]
plt.plot(lim, lim, "r--")
plt.xlabel("Actual thalachh"); plt.ylabel("Predicted"); plt.title("Linear regression")
plt.tight_layout(); plt.savefig(OUT / "02_regression.png", dpi=130); plt.close()

# ---------------------------------------------------------------- Chapter 3
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier()),
    "Decision Tree": DecisionTreeClassifier(random_state=RS),
    "Random Forest": RandomForestClassifier(random_state=RS),
}
grids = {
    "KNN": {"kneighborsclassifier__n_neighbors": list(range(3, 26, 2))},
    "Decision Tree": {"max_depth": [2, 3, 4, 5, 6], "min_samples_leaf": [1, 3, 5, 10]},
    "Random Forest": {"n_estimators": [100, 300], "max_depth": [3, 5, None]},
}
rows, fitted = [], {}
for name, m in models.items():
    if name in grids:
        gs = GridSearchCV(m, grids[name], cv=5, scoring="accuracy").fit(X_tr, y_tr)
        m = gs.best_estimator_
        best = gs.best_params_
    else:
        m.fit(X_tr, y_tr); best = {}
    p = m.predict(X_te)
    proba = m.predict_proba(X_te)[:, 1]
    rows.append({
        "model": name,
        "cv_acc": round(cross_val_score(m, X_tr, y_tr, cv=5).mean(), 3),
        "test_acc": round(accuracy_score(y_te, p), 3),
        "precision": round(precision_score(y_te, p), 3),
        "recall": round(recall_score(y_te, p), 3),
        "f1": round(f1_score(y_te, p), 3),
        "roc_auc": round(roc_auc_score(y_te, proba), 3),
        "best_params": {k.split("__")[-1]: v for k, v in best.items()},
    })
    fitted[name] = m
cls = pd.DataFrame(rows)
cls.to_csv(OUT / "classification_results.csv", index=False)
results["classification"] = rows
best_name = cls.sort_values("roc_auc", ascending=False).iloc[0]["model"]
results["best_classifier"] = best_name

plt.figure(figsize=(6, 5))
for name, m in fitted.items():
    fpr, tpr, _ = roc_curve(y_te, m.predict_proba(X_te)[:, 1])
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("ROC curves"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "03_roc.png", dpi=130); plt.close()

ConfusionMatrixDisplay.from_estimator(fitted[best_name], X_te, y_te, cmap="Blues")
plt.title(f"Confusion matrix - {best_name}")
plt.tight_layout(); plt.savefig(OUT / "03_confusion_matrix.png", dpi=130); plt.close()

rf = fitted["Random Forest"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.plot.barh(figsize=(6, 5), title="Random Forest feature importance")
plt.tight_layout(); plt.savefig(OUT / "03_feature_importance.png", dpi=130); plt.close()
results["top_features"] = imp.sort_values(ascending=False).head(5).round(3).to_dict()

# ---------------------------------------------------------------- Chapter 4
Xs = StandardScaler().fit_transform(X)
ks = range(2, 9)
sil = {k: silhouette_score(Xs, KMeans(k, n_init=10, random_state=RS).fit_predict(Xs)) for k in ks}
inertia = [KMeans(k, n_init=10, random_state=RS).fit(Xs).inertia_ for k in ks]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
ax[1].plot(list(ks), list(sil.values()), "o-"); ax[1].set_title("Silhouette"); ax[1].set_xlabel("k")
plt.tight_layout(); plt.savefig(OUT / "04_kmeans_k_selection.png", dpi=130); plt.close()

K = 2  # matches the two clinical outcomes; silhouette values are reported for transparency
km = KMeans(K, n_init=10, random_state=RS).fit(Xs)
hc = AgglomerativeClustering(K, linkage="ward").fit(Xs)
df["kmeans"], df["hier"] = km.labels_, hc.labels_
results["clustering"] = {
    "silhouette_by_k": {int(k): round(v, 3) for k, v in sil.items()},
    "kmeans_k": K,
    "kmeans_disease_rate_per_cluster": df.groupby("kmeans")["output"].mean().round(3).to_dict(),
    "hier_disease_rate_per_cluster": df.groupby("hier")["output"].mean().round(3).to_dict(),
}
from sklearn.decomposition import PCA
p2 = PCA(2, random_state=RS).fit_transform(Xs)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].scatter(p2[:, 0], p2[:, 1], c=df["kmeans"], cmap="viridis", s=18); ax[0].set_title("K-Means (PCA view)")
ax[1].scatter(p2[:, 0], p2[:, 1], c=df["output"], cmap="coolwarm", s=18); ax[1].set_title("True label (PCA view)")
plt.tight_layout(); plt.savefig(OUT / "04_clusters_pca.png", dpi=130); plt.close()

# ---------------------------------------------------------------- Chapter 5
# Content-based "similar patients" recommender (nearest neighbours on standardised features)
nn = NearestNeighbors(n_neighbors=6).fit(Xs)
_, idx = nn.kneighbors(Xs[[0]])
similar = df.iloc[idx[0][1:]]
similar.to_csv(OUT / "similar_patients_example.csv", index=False)
results["recommender"] = {
    "example_patient_index": 0,
    "similar_patient_indices": [int(i) for i in idx[0][1:]],
    "share_of_similar_with_disease": float(similar["output"].mean()),
}

# leave-one-out style check: does the neighbours' majority label match the patient's own label?
_, all_idx = NearestNeighbors(n_neighbors=6).fit(Xs).kneighbors(Xs)
maj = (df["output"].values[all_idx[:, 1:]].mean(axis=1) >= 0.5).astype(int)
results["recommender"]["neighbour_majority_agreement"] = round(float((maj == df["output"].values).mean()), 3)

with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
