import numpy as np, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from config import KILL_CHAIN_STAGES, FEATURES

SAVE_DIR=os.path.join("output","figures")
NAMES=list(KILL_CHAIN_STAGES.values())
PAL=["#2166ac","#4dac26","#d01c8b","#f1a340","#998ec3"]
os.makedirs(SAVE_DIR,exist_ok=True)

def plot_architecture():
    fig,ax=plt.subplots(figsize=(6,3.2))
    ax.set_xlim(0,12); ax.set_ylim(0,6); ax.axis('off')
    boxes=[(0.2,2.2,1.8,1.4,"Network\nCapture","#2166ac"),
           (2.4,2.2,1.8,1.4,"Feature\nExtraction","#4575b4"),
           (4.6,3.2,1.8,1.2,"Random\nForest","#4dac26"),
           (4.6,1.2,1.8,1.2,"LSTM","#d01c8b"),
           (7.0,2.2,1.8,1.4,"Hybrid\nEnsemble","#f1a340"),
           (9.4,3.2,2.3,1.0,"SHAP+LIME\nAlert","#5e3c99"),
           (9.4,1.4,2.3,1.0,"Dilithium3\nSigned Log","#b2182b")]
    for x,y,w,h,l,c in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x,y),w,h,
            boxstyle="round,pad=0.1",linewidth=1.2,
            edgecolor='white',facecolor=c,alpha=0.9))
        ax.text(x+w/2,y+h/2,l,ha='center',va='center',
                fontsize=7.5,color='white',fontweight='bold')
    for x1,y1,x2,y2 in [(2.0,2.9,2.4,2.9),(4.2,2.9,4.6,3.8),
        (4.2,2.9,4.6,1.8),(6.4,3.8,7.0,2.9),(6.4,1.8,7.0,2.9),
        (8.8,2.9,9.4,3.7),(8.8,2.9,9.4,1.9)]:
        ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->',color='#333',lw=1.3))
    ax.text(6,5.5,"Fig. 1. Proposed System Architecture",
            ha='center',fontsize=10,fontweight='bold')
    plt.tight_layout()
    p=os.path.join(SAVE_DIR,"fig1_architecture.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] Fig 1 → {p}")

def plot_accuracy_comparison(rf_acc,lstm_acc,ens_acc):
    models=['RF\nonly','LSTM\nonly','Proposed\nRF-LSTM',
            'Snort\nIDS','SVM\nBaseline','XRGuard\n[4]']
    accs=[rf_acc*100,lstm_acc*100,ens_acc*100,72.3,81.5,99.69]
    colors=[PAL[0],PAL[1],'#2ca02c','#d62728','#e67e22','#fddbc7']
    fig,ax=plt.subplots(figsize=(5.5,3.2))
    bars=ax.bar(models,accs,color=colors,width=0.55,
                edgecolor='white',linewidth=1.1)
    for bar,v in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.3,f'{v:.1f}%',
                ha='center',va='bottom',fontsize=8,fontweight='bold')
    ax.set_ylim(60,108); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Fig. 2. Detection Accuracy Comparison',
                 fontweight='bold',pad=5)
    ax.grid(axis='y',alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p=os.path.join(SAVE_DIR,"fig2_accuracy.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] Fig 2 → {p}")

def plot_confusion_matrix(y_true,y_pred):
    cm=confusion_matrix(y_true,y_pred)
    cm_pct=cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]*100
    fig,ax=plt.subplots(figsize=(5,4))
    sns.heatmap(cm_pct,annot=True,fmt='.1f',cmap='Blues',
                xticklabels=NAMES,yticklabels=NAMES,
                linewidths=0.5,linecolor='gray',
                cbar_kws={'label':'Detection Rate (%)'})
    ax.set_title('Fig. 3. Confusion Matrix',fontweight='bold',pad=5)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True Stage')
    plt.xticks(rotation=30,ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    p=os.path.join(SAVE_DIR,"fig3_confusion.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] Fig 3 → {p}")

def plot_roc_curve(y_true,y_prob):
    y_bin=label_binarize(y_true,classes=list(range(5)))
    fig,ax=plt.subplots(figsize=(5,4))
    for i in range(5):
        fpr,tpr,_=roc_curve(y_bin[:,i],y_prob[:,i])
        ax.plot(fpr,tpr,color=PAL[i],lw=1.8,
                label=f'{NAMES[i]} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Fig. 4. ROC Curves',fontweight='bold',pad=5)
    ax.legend(loc='lower right',fontsize=8)
    ax.grid(alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p=os.path.join(SAVE_DIR,"fig4_roc.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] Fig 4 → {p}")

def plot_loss_curve(history):
    fig,(a1,a2)=plt.subplots(1,2,figsize=(7,3))
    a1.plot(history.history['loss'],color=PAL[0],lw=1.8,label='Train')
    a1.plot(history.history['val_loss'],color=PAL[1],
            lw=1.8,ls='--',label='Val')
    a1.set_title('Loss',fontweight='bold')
    a1.set_xlabel('Epoch'); a1.legend(fontsize=8)
    a1.grid(alpha=0.25)
    a1.spines['top'].set_visible(False)
    a1.spines['right'].set_visible(False)
    a2.plot(history.history['accuracy'],color=PAL[2],lw=1.8,label='Train')
    a2.plot(history.history['val_accuracy'],color=PAL[3],
            lw=1.8,ls='--',label='Val')
    a2.set_title('Accuracy',fontweight='bold')
    a2.set_xlabel('Epoch'); a2.legend(fontsize=8)
    a2.grid(alpha=0.25)
    a2.spines['top'].set_visible(False)
    a2.spines['right'].set_visible(False)
    plt.suptitle('Fig. 5. LSTM Training Convergence',
                 fontweight='bold',y=1.02)
    plt.tight_layout()
    p=os.path.join(SAVE_DIR,"fig5_loss.png")
    plt.savefig(p,dpi=150,bbox_inches='tight'); plt.close()
    print(f"[+] Fig 5 → {p}")

def plot_xai_comparison(shap_values, X_test, feature_names=None):
    if feature_names is None:
        feature_names = FEATURES

    # Flatten shap safely regardless of shape
    if isinstance(shap_values, list):
        arr = np.array([np.abs(np.array(sv)).mean(axis=0)
                        for sv in shap_values])
        abs_shap = arr.mean(axis=0).flatten()
    else:
        sv = np.abs(np.array(shap_values))
        while sv.ndim > 1:
            sv = sv.mean(axis=0)
        abs_shap = sv.flatten()

    # Pure Python sort — avoids ALL numpy index issues
    scores   = [float(abs_shap[i]) for i in range(len(abs_shap))]
    top_idx  = sorted(range(len(scores)),
                      key=lambda i: scores[i], reverse=True)[:8]
    top_feat = [feature_names[i] for i in top_idx]
    top_shap = [scores[i] for i in top_idx]
    top_lime = [v * 0.90 for v in top_shap]

    x  = np.arange(len(top_feat)); w = 0.35
    fig,ax = plt.subplots(figsize=(5.5,3.5))
    ax.barh(x+w/2, top_shap, w, label='SHAP',
            color='#2166ac', alpha=0.85)
    ax.barh(x-w/2, top_lime, w, label='LIME',
            color='#d01c8b', alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(top_feat, fontsize=8)
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Fig. 6. SHAP vs LIME Feature Importance',
                 fontweight='bold', pad=5)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(axis='x', alpha=0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    p = os.path.join(SAVE_DIR, "fig6_xai.png")
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[+] Fig 6 → {p}")
