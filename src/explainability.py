import shap, numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from config import FEATURES, KILL_CHAIN_STAGES
STAGE_NAMES=list(KILL_CHAIN_STAGES.values())

try:
    import lime, lime.lime_tabular
    LIME_OK=True
except ImportError:
    LIME_OK=False
    print("[!] LIME not installed: py -3.11 -m pip install lime")

# ── SHAP ──────────────────────────────────────────────────────
def compute_shap(rf, X_test, save_dir="output/figures"):
    os.makedirs(save_dir,exist_ok=True)
    print("[*] Computing SHAP values...")
    explainer=shap.TreeExplainer(rf)
    sv=explainer.shap_values(X_test[:500])
    plt.figure(figsize=(10,8))
    shap.summary_plot(sv,X_test[:500],feature_names=FEATURES,
                      class_names=STAGE_NAMES,show=False)
    p=os.path.join(save_dir,"shap_summary.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] SHAP summary → {p}")
    return sv, explainer

def explain_single_alert(explainer, sample, feature_names=FEATURES):
    sv = explainer.shap_values(sample.reshape(1, -1))
    if isinstance(sv, list):
        arr = np.array([np.abs(np.array(s)).flatten()
                        for s in sv])
        abs_sv = arr.mean(axis=0).flatten()
    else:
        abs_sv = np.abs(np.array(sv)).flatten()

    # Clamp to safe length
    n        = min(3, len(feature_names), len(abs_sv))
    scores   = [float(abs_sv[i]) for i in range(len(abs_sv))]
    top_idx  = sorted(range(len(scores)),
                      key=lambda i: scores[i],
                      reverse=True)[:n]
    # Guard against out-of-range
    return {feature_names[i]: scores[i]
            for i in top_idx if i < len(feature_names)}

# ── LIME ──────────────────────────────────────────────────────
def compute_lime(rf, X_train, X_test, save_dir="output/figures"):
    if not LIME_OK:
        print("[!] LIME skipped"); return []
    os.makedirs(save_dir,exist_ok=True)
    print("[*] Computing LIME explanations...")
    exp_obj=lime.lime_tabular.LimeTabularExplainer(
        X_train,feature_names=FEATURES,class_names=STAGE_NAMES,
        mode='classification',discretize_continuous=True)
    n=min(3,len(X_test))
    fig,axes=plt.subplots(1,n,figsize=(15,5))
    if n==1: axes=[axes]
    results=[]
    for i in range(n):
        e=exp_obj.explain_instance(X_test[i],rf.predict_proba,
                                   num_features=6,top_labels=1)
        lbl=int(rf.predict(X_test[i].reshape(1,-1))[0])
        el=e.as_list(label=lbl)
        results.append((KILL_CHAIN_STAGES[lbl],el))
        ax=axes[i]
        feats=[x[0][:22] for x in el[:6]]
        vals=[x[1] for x in el[:6]]
        cols=['#2ca02c' if v>0 else '#d62728' for v in vals]
        ax.barh(feats,vals,color=cols)
        ax.set_title(f"Sample {i+1}: {KILL_CHAIN_STAGES[lbl]}",fontsize=9)
        ax.axvline(x=0,color='black',linewidth=0.8)
        ax.tick_params(labelsize=7)
    plt.suptitle("LIME Explanations — Kill Chain Detection",fontsize=11)
    plt.tight_layout()
    p=os.path.join(save_dir,"lime_explanations.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] LIME → {p}"); return results

def explain_single_lime(rf, X_train, sample, feature_names=FEATURES):
    if not LIME_OK: return {}
    exp_obj=lime.lime_tabular.LimeTabularExplainer(
        X_train,feature_names=feature_names,class_names=STAGE_NAMES,
        mode='classification',discretize_continuous=True)
    lbl=int(rf.predict(sample.reshape(1,-1))[0])
    e=exp_obj.explain_instance(sample,rf.predict_proba,
                                num_features=3,top_labels=1)
    return {f[:30]:round(w,4) for f,w in e.as_list(label=lbl)[:3]}
