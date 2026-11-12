import pandas as pd, numpy as np, glob, os, joblib
from sklearn.preprocessing import StandardScaler

SEQ, BLOCK, CAP_FILE = 10, 200, 20000
RNG = np.random.RandomState(42)
FEATURES = ['duration','protocol','dst_port','flow_bytes_per_s','flow_packets_per_s',
 'fwd_pkt_len_mean','bwd_pkt_len_mean','pkt_len_var','iat_mean','iat_std','iat_max',
 'fwd_iat_mean','bwd_iat_mean','syn_flag_count','rst_flag_count','psh_flag_count','ack_flag_count']
COLS = {'flow duration':'duration','protocol':'protocol','dst port':'dst_port','destination port':'dst_port',
 'flow byts/s':'flow_bytes_per_s','flow bytes/s':'flow_bytes_per_s','flow pkts/s':'flow_packets_per_s',
 'flow packets/s':'flow_packets_per_s','fwd pkt len mean':'fwd_pkt_len_mean','fwd packet length mean':'fwd_pkt_len_mean',
 'bwd pkt len mean':'bwd_pkt_len_mean','bwd packet length mean':'bwd_pkt_len_mean','pkt len var':'pkt_len_var',
 'packet length variance':'pkt_len_var','flow iat mean':'iat_mean','flow iat std':'iat_std','flow iat max':'iat_max',
 'fwd iat mean':'fwd_iat_mean','bwd iat mean':'bwd_iat_mean','syn flag cnt':'syn_flag_count','syn flag count':'syn_flag_count',
 'rst flag cnt':'rst_flag_count','rst flag count':'rst_flag_count','psh flag cnt':'psh_flag_count','psh flag count':'psh_flag_count',
 'ack flag cnt':'ack_flag_count','ack flag count':'ack_flag_count','timestamp':'timestamp','label':'label_raw'}
LABEL_MAP = {'benign':0,'bot':2,'ftp-bruteforce':1,'ssh-bruteforce':1,
 'infilteration':3,'infiltration':3,
 'dos attacks-hulk':4,'dos attacks-slowhttptest':4,'dos attacks-slowloris':4,'dos attacks-goldeneye':4,
 'ddos attacks-loic-http':4,'ddos attack-loic-http':4,'ddos attack-hoic':4,'ddos attack-loic-udp':4}

def slog(a): return np.sign(a) * np.log1p(np.abs(a))

def load_file(p):
    df = pd.read_csv(p, low_memory=False, usecols=lambda c: c.strip().lower() in COLS)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns=COLS)
    need = FEATURES + ['timestamp', 'label_raw']
    miss = [c for c in need if c not in df.columns]
    if miss:
        print('[!] skip', p, 'missing', miss); return None
    df['label'] = df['label_raw'].astype(str).str.strip().str.lower().map(LABEL_MAP)
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', dayfirst=True)
    for c in FEATURES: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ['timestamp', 'label'])
    df['label'] = df['label'].astype(int)
    return df.sort_values('timestamp', kind='stable').reset_index(drop=True)

def build(raw_glob="data/raw/*.csv", out="data/seq_dataset.npz"):
    files = [f for f in sorted(glob.glob("data/*TrafficForML*.csv")) if "Thuesday-20" not in f and "Thursday-20-02" not in f]
    assert files, "no CIC day files found in data/"
    print("files:", [os.path.basename(f) for f in files])
    Xs, ys, bs, stats = [], [], [], []
    for fi, p in enumerate(files):
        df = load_file(p)
        if df is None: continue
        stats.append(f"{os.path.basename(p)}: {len(df)} flows, labels {df['label'].value_counts().sort_index().to_dict()}")
        print(stats[-1])
        n = (len(df) // SEQ) * SEQ
        X = df[FEATURES].values[:n].astype(np.float32).reshape(-1, SEQ, len(FEATURES))
        y = df['label'].values[:n].reshape(-1, SEQ)[:, -1]
        b = fi * 10**7 + np.arange(len(y)) // BLOCK
        keep = []
        for c in range(5):
            idx = np.where(y == c)[0]
            if len(idx) > CAP_FILE: idx = RNG.choice(idx, CAP_FILE, replace=False)
            keep.append(idx)
        keep = np.sort(np.concatenate(keep))
        Xs.append(X[keep]); ys.append(y[keep]); bs.append(b[keep])
    X = np.concatenate(Xs); y = np.concatenate(ys); b = np.concatenate(bs)
    ub = np.unique(b); RNG.shuffle(ub); n = len(ub)
    split = np.full(len(b), 2)
    split[np.isin(b, ub[:int(.7 * n)])] = 0
    split[np.isin(b, ub[int(.7 * n):int(.8 * n)])] = 1
    caps = {0: 8000, 1: 1500, 2: 2000}
    sel = {}
    for s in range(3):
        ids = []
        for c in range(5):
            idx = np.where((split == s) & (y == c))[0]
            ids.append(RNG.choice(idx, min(caps[s], len(idx)), replace=False))
        sel[s] = np.sort(np.concatenate(ids))
    X = slog(X)
    sc = StandardScaler().fit(X[sel[0]].reshape(-1, len(FEATURES)))
    f = lambda a: sc.transform(a.reshape(-1, len(FEATURES))).reshape(a.shape).astype(np.float32)
    os.makedirs("output", exist_ok=True); joblib.dump(sc, "output/scaler.pkl")
    np.savez_compressed(out, Xtr=f(X[sel[0]]), ytr=y[sel[0]], Xva=f(X[sel[1]]), yva=y[sel[1]], Xte=f(X[sel[2]]), yte=y[sel[2]])
    with open("output/dataset_stats.txt", "w") as fh:
        fh.write("\n".join(stats) + "\n")
        for s, nm in enumerate(["train", "val", "test"]):
            fh.write(f"{nm}: {len(sel[s])} windows, per class {np.bincount(y[sel[s]], minlength=5).tolist()}\n")
    print(open("output/dataset_stats.txt").read())

def load_split(path="data/seq_dataset.npz"):
    d = np.load(path)
    return (d['Xtr'], d['ytr']), (d['Xva'], d['yva']), (d['Xte'], d['yte'])

if __name__ == "__main__":
    build()