import numpy as np, pandas as pd, joblib, os, sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from config import FEATURES, TEST_SIZE, RANDOM_STATE, SEQUENCE_LENGTH

def load_and_preprocess(path="data/network_traffic.csv"):
    df=pd.read_csv(path)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)

    # Sort for temporal sequencing
    if 'Timestamp' in df.columns:
        df=df.sort_values('Timestamp').reset_index(drop=True)
    elif 'timestamp' in df.columns:
        df=df.sort_values('timestamp').reset_index(drop=True)
    else:
        df=df.sort_values('label').reset_index(drop=True)

    X=df[FEATURES].values; y=df['label'].values
    scaler=StandardScaler(); X_s=scaler.fit_transform(X)
    Xtr,Xte,ytr,yte=train_test_split(X_s,y,test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE,stratify=y)
    os.makedirs("output",exist_ok=True)
    joblib.dump(scaler,"output/scaler.pkl")
    print(f"[+] Train:{Xtr.shape} Test:{Xte.shape}")
    return Xtr,Xte,ytr,yte,scaler

def make_sequences(X,y,seq_len=SEQUENCE_LENGTH):
    Xs,ys=[],[]
    for i in range(len(X)-seq_len):
        Xs.append(X[i:i+seq_len]); ys.append(y[i+seq_len])
    return np.array(Xs),np.array(ys)
