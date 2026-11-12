# ============================================================
# cic_loader.py — Maps CIC-IDS2018 columns to project features
# ============================================================
import pandas as pd
import numpy as np
import os, glob, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from config import FEATURES

# CIC-IDS2018 column → our feature name
COL_MAP = {
    ' Flow Duration':            'duration',
    ' Protocol':                 'protocol',
    ' Destination Port':         'dst_port',
    ' Flow Bytes/s':             'flow_bytes_per_s',
    ' Flow Packets/s':           'flow_packets_per_s',
    ' Fwd Packet Length Mean':   'fwd_packet_length_mean',
    ' Bwd Packet Length Mean':   'bwd_packet_length_mean',
    ' Packet Length Variance':   'packet_length_variance',
    ' Flow IAT Mean':            'iat_mean',
    ' Flow IAT Std':             'iat_std',
    ' Flow IAT Max':             'iat_max',
    ' Fwd IAT Mean':             'fwd_iat_mean',
    ' Bwd IAT Mean':             'bwd_iat_mean',
    ' SYN Flag Count':           'syn_flag_count',
    ' RST Flag Count':           'rst_flag_count',
    ' PSH Flag Count':           'psh_flag_count',
    ' ACK Flag Count':           'ack_flag_count',
    ' Label':                    'label_raw',
}

# CIC-IDS2018 label → kill chain stage
LABEL_MAP = {
    'Benign':                    0,
    'Bot':                       2,
    'Infiltration':              1,
    'FTP-BruteForce':            1,
    'SSH-Bruteforce':            1,
    'Brute Force -Web':          3,
    'Brute Force -XSS':          3,
    'SQL Injection':             3,
    'DoS attacks-Hulk':          4,
    'DoS attacks-SlowHTTPTest':  4,
    'DoS attacks-Slowloris':     4,
    'DoS attacks-GoldenEye':     4,
    'DDoS attacks-LOIC-HTTP':    4,
    'DDOS attack-HOIC':          4,
    'DDOS attack-LOIC-UDP':      4,
}

def load_cic_dataset(input_path, output_path="data/network_traffic.csv",
                     n_samples=50000):
    """
    Load CIC-IDS2018 CSV, map columns + labels,
    save as network_traffic.csv for main.py
    """
    print(f"[*] Loading CIC-IDS2018: {input_path}")

    # Read with low_memory=False to avoid dtype warnings
    df = pd.read_csv(input_path, low_memory=False)
    print(f"[+] Raw shape: {df.shape}")
    print(f"[+] Raw labels: {df[' Label'].value_counts().to_dict()}")

    # Rename known columns
    df.rename(columns=COL_MAP, inplace=True)

    # Map labels to kill chain stages
    df['label'] = df['label_raw'].map(LABEL_MAP)
    df.dropna(subset=['label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # Add missing features with sensible defaults
    if 'src_port'           not in df.columns:
        df['src_port']           = np.random.randint(1024,65535,len(df))
    if 'packet_length_entropy' not in df.columns:
        df['packet_length_entropy'] = np.log1p(
            df.get('packet_length_variance', pd.Series(np.zeros(len(df))))
        ).clip(0,8)
    if 'connection_frequency'  not in df.columns:
        df['connection_frequency'] = df.get('flow_packets_per_s',
                                            pd.Series(np.ones(len(df))))
    if 'unique_dst_ips'        not in df.columns:
        df['unique_dst_ips']       = 1
    if 'unique_dst_ports'      not in df.columns:
        df['unique_dst_ports']     = 1
    if 'tls_cert_anomaly'      not in df.columns:
        df['tls_cert_anomaly']     = 0

    # Keep only needed columns
    keep = FEATURES + ['label']
    missing = [c for c in keep if c not in df.columns]
    if missing:
        print(f"[!] Still missing: {missing}")
        for c in missing:
            df[c] = 0

    df = df[keep]

    # Clean infinite/nan
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # Sample to manageable size — balanced per class
    per_class = n_samples // 5
    frames = []
    for stage in range(5):
        subset = df[df['label'] == stage]
        if len(subset) == 0:
            print(f"[!] Stage {stage} has no samples")
            continue
        n = min(per_class, len(subset))
        frames.append(subset.sample(n, random_state=42))

    df_final = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=42)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)

    print(f"\n[+] Saved → {output_path}")
    print(f"[+] Shape: {df_final.shape}")
    print(f"[+] Label distribution:")
    print(df_final['label'].value_counts().sort_index())
    return df_final


if __name__ == "__main__":
    # Change this to your actual file path
    CIC_FILE = "data/network_traffic.csv"
    load_cic_dataset(CIC_FILE, output_path="data/network_traffic_clean.csv")