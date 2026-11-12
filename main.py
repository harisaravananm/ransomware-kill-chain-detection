# main.py — full pipeline (real CIC-IDS2018 windows)
import sys, os
sys.modules['oqs'] = None
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from config import KILL_CHAIN_STAGES
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
SEP = "=" * 60

def fpr_macro(y, p):
    cm = confusion_matrix(y, p); fp = cm.sum(0) - np.diag(cm)
    tn = cm.sum() - cm.sum(0) - cm.sum(1) + np.diag(cm)
    return float(np.mean(fp / (fp + tn)))

print(f"\n{SEP}\nSTEP 2: Loading sequence dataset\n{SEP}")
from src.build_dataset import load_split
(X_tr2, y_tr2), (X_val, y_val), (X_test_seq, y_test_seq) = load_split()
X_train, y_train = X_tr2[:, -1, :], y_tr2
X_test,  y_test  = X_test_seq[:, -1, :], y_test_seq
print("train/val/test windows:", X_tr2.shape, X_val.shape, X_test_seq.shape)

print(f"\n{SEP}\nSTEP 3: Training Random Forest\n{SEP}")
from src.models.rf_model import train_rf, evaluate_rf
rf = train_rf(X_train, y_train)
rf_pred, rf_acc = evaluate_rf(rf, X_test, y_test)

print(f"\n{SEP}\nSTEP 4: Training LSTM\n{SEP}")
from src.models.lstm_model import train_lstm, evaluate_lstm
lstm, history = train_lstm(X_tr2, y_tr2, X_val, y_val)
lstm_pred, lstm_prob, lstm_acc = evaluate_lstm(lstm, X_test_seq, y_test_seq)

print(f"\n{SEP}\nSTEP 5: Hybrid Ensemble\n{SEP}")
from src.models.ensemble import ensemble_predict, evaluate_ensemble, tune_weights
rf_w = tune_weights(rf, lstm, X_val, y_val)
ens_pred, conf, ens_prob = ensemble_predict(rf, lstm, X_test, X_test_seq,
                                            rf_w=rf_w, lstm_w=1 - rf_w)
ens_acc, ens_f1 = evaluate_ensemble(ens_pred, y_test_seq)
print(classification_report(y_test_seq, ens_pred, digits=3))
print(confusion_matrix(y_test_seq, ens_pred))
print(f"\n--- TABLE METRICS (acc | macro-F1 | macro-FPR) ---")
for n, p in [("RF", rf_pred), ("LSTM", lstm_pred), ("Ensemble", ens_pred)]:
    print(f"{n:9s} {accuracy_score(y_test_seq,p):.4f} | {f1_score(y_test_seq,p,average='macro'):.4f} | {fpr_macro(y_test_seq,p):.4f}")
from sklearn.metrics import roc_auc_score
print("Per-class AUC:", [round(roc_auc_score((y_test_seq == c).astype(int), ens_prob[:, c]), 3) for c in range(5)])

print(f"\n{SEP}\nSTEP 6: SHAP\n{SEP}")
from src.explainability import compute_shap, explain_single_alert
shap_values, explainer = compute_shap(rf, X_test)

print(f"\n{SEP}\nSTEP 6b: LIME\n{SEP}")
from src.explainability import compute_lime
lime_results = compute_lime(rf, X_train, X_test[:5])
for stage, exp in lime_results:
    print(f"  LIME [{stage}]: {exp[:2]}")

print(f"\n{SEP}\nSTEP 7: Generating Figures\n{SEP}")
from src.visualizations import (plot_architecture, plot_accuracy_comparison,
    plot_confusion_matrix, plot_roc_curve, plot_loss_curve, plot_xai_comparison)
plot_architecture()
plot_accuracy_comparison(rf_acc, lstm_acc, ens_acc)
plot_confusion_matrix(y_test_seq, ens_pred)
plot_roc_curve(y_test_seq, ens_prob)
plot_loss_curve(history)
plot_xai_comparison(shap_values, X_test)

print(f"\n{SEP}\nSTEP 8: Post-Quantum Signed Alerts\n{SEP}")
try:
    from src.alert_system import AlertManager
    from src.explainability import explain_single_lime
    mgr = AlertManager(); count = 0
    for i, (stage, c) in enumerate(zip(ens_pred[:300], conf[:300])):
        if stage > 0:
            shap_r = explain_single_alert(explainer, X_test[i])
            lime_r = explain_single_lime(rf, X_train, X_test[i])
            mgr.generate_alert(i, stage, c, shap_r, lime_r)
            count += 1
            if count >= 3: break
    mgr.save_log()
except Exception as e:
    print("[!] STEP 8 FAILED:", repr(e))

print(f"\n{SEP}\nFINAL RESULTS\n{SEP}")
print(f"  RF {rf_acc*100:.2f}% | LSTM {lstm_acc*100:.2f}% | Ensemble {ens_acc*100:.2f}% | macro-F1 {ens_f1:.4f}")
print("[DONE]")