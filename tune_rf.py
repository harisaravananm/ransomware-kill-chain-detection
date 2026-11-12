import numpy as np, pandas as pd, optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("data/network_traffic.csv")
y = df["label"]
X = df.drop(columns=["label"]).select_dtypes("number")
X = X.replace([np.inf, -np.inf], np.nan).fillna(0).clip(-1e12, 1e12)

if len(X) > 100_000:
    X, _, y, _ = train_test_split(X, y, train_size=100_000, stratify=y, random_state=7)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

def objective(t):
    m = RandomForestClassifier(
        n_estimators=t.suggest_int("n_estimators", 100, 400, step=100),
        max_depth=t.suggest_int("max_depth", 10, 60),
        min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 5),
        max_features=t.suggest_categorical("max_features", ["sqrt", "log2", 0.3]),
        class_weight=t.suggest_categorical("class_weight", [None, "balanced"]),
        n_jobs=-1, random_state=7)
    return cross_val_score(m, Xtr, ytr, cv=3, scoring="f1_macro").mean()   # train split only

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=7))
study.optimize(objective, n_trials=20)
print("best CV macro-F1 (train only):", round(study.best_value, 4))
print("best params:", study.best_params)

models = {
    "baseline (300 trees, depth None)": RandomForestClassifier(n_estimators=300, max_depth=None, n_jobs=-1, random_state=7),
    "tuned (Optuna)": RandomForestClassifier(**study.best_params, n_jobs=-1, random_state=7),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    p = m.predict(Xte)
    print(f"{name}: acc={accuracy_score(yte, p):.4f}  macroF1={f1_score(yte, p, average='macro'):.4f}")
print(classification_report(yte, p))   # per-class report for the tuned model