import numpy as np, sys, os
from sklearn.metrics import accuracy_score, f1_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import KILL_CHAIN_STAGES

def tune_weights(rf, lstm_model, Xva_seq, yva):
    rf_p = rf.predict_proba(Xva_seq[:, -1, :])
    ls_p = lstm_model.predict(Xva_seq, verbose=0)
    best_w, best_f = 0.5, -1
    for w in np.arange(0, 1.01, 0.05):
        f = f1_score(yva, np.argmax(w * rf_p + (1 - w) * ls_p, axis=1), average='macro')
        if f > best_f: best_w, best_f = w, f
    print(f"[Ensemble] val grid search: RF weight {best_w:.2f}, LSTM weight {1-best_w:.2f}, macro-F1 {best_f:.4f}")
    return best_w

def ensemble_predict(rf, lstm_model, X_flat, X_seq, rf_w=0.5, lstm_w=0.5):
    rf_p = rf.predict_proba(X_flat)
    ls_p = lstm_model.predict(X_seq, verbose=0)
    combined = rf_w * rf_p + lstm_w * ls_p
    return np.argmax(combined, axis=1), np.max(combined, axis=1), combined

def evaluate_ensemble(y_pred, y_true):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"\n[Ensemble] Accuracy:{acc:.4f} | macro-F1:{f1:.4f}")
    pc = f1_score(y_true, y_pred, average=None)
    for i, n in KILL_CHAIN_STAGES.items():
        print(f"  {n:20s} F1:{pc[i]:.4f}")
    return acc, f1