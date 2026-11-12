import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

df = pd.read_csv("data/network_traffic.csv")
y = df["label"]; X = df.drop(columns=["label"]).select_dtypes("number")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e12, 1e12)
if len(X) > 100_000:
    X, _, y, _ = train_test_split(X, y, train_size=100_000, stratify=y, random_state=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

models = {
 "baseline (300 trees, depth None)": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=7),
 "tuned (Optuna)": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_leaf=2,
                                          max_features="log2", class_weight="balanced", n_jobs=-1, random_state=7)}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte)
    print(f"{name}: acc={accuracy_score(yte, p):.4f}  macroF1={f1_score(yte, p, average='macro'):.4f}")
print(classification_report(yte, p))   # per-class report for the tuned model