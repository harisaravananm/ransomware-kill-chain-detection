import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# All 6 CIC files
CIC_FILES = [
    "C:/Users/acer/Downloads/archive/02-14-2018.csv",
    "C:/Users/acer/Downloads/archive/02-16-2018.csv",
    "C:/Users/acer/Downloads/archive/02-21-2018.csv",
    "C:/Users/acer/Downloads/archive/02-28-2018.csv",
    "C:/Users/acer/Downloads/archive/03-01-2018.csv",
    "C:/Users/acer/Downloads/archive/03-02-2018.csv",
]
OUT_FILE  = "data/network_traffic.csv"
N_SAMPLES = 50000   # was 10000
per_class  = N_SAMPLES // 5   # 10000 per class

LABEL_MAP = {
    'Benign':                    0,
    'FTP-BruteForce':            1,
    'SSH-BruteForce':            1,
    'Infilteration':             1,
    'Bot':                       2,
    'DoS attacks-Hulk':          4,
    'DoS attacks-SlowHTTPTest':  4,
    'DoS attacks-Slowloris':     4,
    'DoS attacks-GoldenEye':     4,
    'DDoS attacks-LOIC-HTTP':    4,
    'DDOS attack-HOIC':          4,
    'DDOS attack-LOIC-UDP':      4,
    'Brute Force -Web':          3,
    'Brute Force -XSS':          3,
    'SQL Injection':             3,
}

COL_MAP = {
    'Flow Duration':     'duration',
    'Protocol':          'protocol',
    'Src Port':          'src_port',
    'Dst Port':          'dst_port',
    'Flow Byts/s':       'flow_bytes_per_s',
    'Flow Pkts/s':       'flow_packets_per_s',
    'Fwd Pkt Len Mean':  'fwd_packet_length_mean',
    'Bwd Pkt Len Mean':  'bwd_packet_length_mean',
    'Pkt Len Var':       'packet_length_variance',
    'Flow IAT Mean':     'iat_mean',
    'Flow IAT Std':      'iat_std',
    'Flow IAT Max':      'iat_max',
    'Fwd IAT Mean':      'fwd_iat_mean',
    'Bwd IAT Mean':      'bwd_iat_mean',
    'SYN Flag Cnt':      'syn_flag_count',
    'RST Flag Cnt':      'rst_flag_count',
    'PSH Flag Cnt':      'psh_flag_count',
    'ACK Flag Cnt':      'ack_flag_count',
    'Pkt Size Avg':      'connection_frequency',
    'Label':             'label_raw',
}

FEATURES = [
    'duration','protocol','src_port','dst_port',
    'flow_bytes_per_s','flow_packets_per_s',
    'fwd_packet_length_mean','bwd_packet_length_mean',
    'packet_length_variance','packet_length_entropy',
    'iat_mean','iat_std','iat_max',
    'fwd_iat_mean','bwd_iat_mean',
    'syn_flag_count','rst_flag_count','psh_flag_count',
    'ack_flag_count','connection_frequency',
    'unique_dst_ips','unique_dst_ports','tls_cert_anomaly'
]

# Load all files
all_frames = []
for fpath in CIC_FILES:
    if not os.path.exists(fpath):
        print(f"[!] Not found: {fpath} — skipping")
        continue
    print(f"[*] Loading {os.path.basename(fpath)}...")
    try:
        df = pd.read_csv(fpath, low_memory=False)
        print(f"    Shape: {df.shape}")
        print(f"    Labels: {df['Label'].value_counts().to_dict()}")
        all_frames.append(df)
    except Exception as e:
        print(f"[!] Error: {e}")

print(f"\n[+] Loaded {len(all_frames)} files")
df = pd.concat(all_frames, ignore_index=True)
print(f"[+] Combined shape: {df.shape}")
print(f"[+] All labels: {df['Label'].value_counts().to_dict()}")

# Rename and map
df.rename(columns=COL_MAP, inplace=True)
df['label'] = df['label_raw'].map(LABEL_MAP)
df.dropna(subset=['label'], inplace=True)
df['label'] = df['label'].astype(int)

# Add missing features
if 'packet_length_entropy' not in df.columns:
    pv = df.get('packet_length_variance',
                pd.Series(np.zeros(len(df))))
    pv_numeric = pd.to_numeric(pv, errors='coerce').fillna(0)
df['packet_length_entropy'] = np.log1p(pv_numeric).clip(0, 8)

if 'unique_dst_ips'   not in df.columns: df['unique_dst_ips']   = 1
if 'unique_dst_ports' not in df.columns: df['unique_dst_ports'] = 1
if 'tls_cert_anomaly' not in df.columns: df['tls_cert_anomaly'] = 0

for c in FEATURES:
    if c not in df.columns:
        print(f"[!] Missing '{c}' — filling 0")
        df[c] = 0

df = df[FEATURES + ['label']]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

print(f"\n[+] After clean: {df.shape}")
print(f"[+] Label dist: {df['label'].value_counts().sort_index().to_dict()}")

# Balance — sample per class
per_class = N_SAMPLES // 5
frames = []
for stage in range(5):
    subset = df[df['label'] == stage]
    n = min(per_class, len(subset))
    if n == 0:
        print(f"[!] Stage {stage} STILL empty — using synthetic")
        from src.data_generator import _base,_recon,_c2,_lateral,_preenc
        gen_fn = [_base,_recon,_c2,_lateral,_preenc][stage]
        frames.append(gen_fn(per_class))
    else:
        frames.append(subset.sample(n, random_state=42))
        print(f"[+] Stage {stage}: {n} real samples")

df_final = pd.concat(frames, ignore_index=True)
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

os.makedirs("data", exist_ok=True)
df_final.to_csv(OUT_FILE, index=False)

print(f"\n[+] Saved → {OUT_FILE}")
print(f"[+] Final shape: {df_final.shape}")
print(f"[+] Final labels: {df_final['label'].value_counts().sort_index().to_dict()}")
print("\n[DONE] Now run: py -3.11 main.py")