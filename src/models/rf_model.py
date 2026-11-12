import joblib, os, sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'../..'))
from config import RF_N_ESTIMATORS, RANDOM_STATE, MODEL_RF_PATH, KILL_CHAIN_STAGES

def train_rf(X_train,y_train):
    print("[*] Training Random Forest...")
    rf=RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train,y_train)
    os.makedirs("output",exist_ok=True)
    joblib.dump(rf,MODEL_RF_PATH)
    print(f"[+] RF saved → {MODEL_RF_PATH}"); return rf

def evaluate_rf(rf,X_test,y_test):
    y_pred=rf.predict(X_test)
    acc=accuracy_score(y_test,y_pred)
    print(f"\n[RF] Accuracy: {acc:.4f}")
    print(classification_report(y_test,y_pred,
          target_names=list(KILL_CHAIN_STAGES.values())))
    return y_pred,acc

def load_rf():
    return joblib.load(MODEL_RF_PATH)
