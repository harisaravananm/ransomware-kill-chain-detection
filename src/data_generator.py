import numpy as np, pandas as pd, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),'..'))
from config import RANDOM_STATE
np.random.seed(RANDOM_STATE)

def _base(n):
    return pd.DataFrame({
        'duration':np.random.exponential(2,n),
        'protocol':np.random.choice([6,17],n),
        'src_port':np.random.randint(1024,65535,n),
        'dst_port':np.random.choice([80,443,53,22],n),
        'flow_bytes_per_s':np.random.normal(5000,1000,n).clip(0),
        'flow_packets_per_s':np.random.normal(50,10,n).clip(0),
        'fwd_packet_length_mean':np.random.normal(500,100,n).clip(0),
        'bwd_packet_length_mean':np.random.normal(400,80,n).clip(0),
        'packet_length_variance':np.random.normal(200,50,n).clip(0),
        'packet_length_entropy':np.random.normal(3.5,0.3,n).clip(0,8),
        'iat_mean':np.random.normal(0.02,0.005,n).clip(0),
        'iat_std':np.random.normal(0.01,0.002,n).clip(0),
        'iat_max':np.random.normal(0.1,0.02,n).clip(0),
        'fwd_iat_mean':np.random.normal(0.02,0.005,n).clip(0),
        'bwd_iat_mean':np.random.normal(0.02,0.005,n).clip(0),
        'syn_flag_count':np.random.poisson(1,n),
        'rst_flag_count':np.random.poisson(0.1,n),
        'psh_flag_count':np.random.poisson(2,n),
        'ack_flag_count':np.random.poisson(5,n),
        'connection_frequency':np.random.normal(10,2,n).clip(0),
        'unique_dst_ips':np.random.poisson(2,n),
        'unique_dst_ports':np.random.poisson(3,n),
        'tls_cert_anomaly':np.zeros(n),
        'label':np.zeros(n,dtype=int)
    })

def _recon(n):
    d=_base(n); d['syn_flag_count']=np.random.poisson(50,n)
    d['unique_dst_ports']=np.random.randint(100,1000,n)
    d['unique_dst_ips']=np.random.randint(50,500,n)
    d['flow_packets_per_s']=np.random.normal(500,50,n).clip(0)
    d['rst_flag_count']=np.random.poisson(30,n); d['label']=1; return d

def _c2(n):
    d=_base(n); d['iat_std']=np.random.normal(0.001,0.0005,n).clip(0)
    d['iat_mean']=np.random.normal(30,1,n).clip(0)
    d['dst_port']=np.random.choice([443,8443,4444],n)
    d['tls_cert_anomaly']=np.random.choice([0,1],n,p=[0.2,0.8])
    d['packet_length_entropy']=np.random.normal(7.5,0.2,n).clip(0,8)
    d['label']=2; return d

def _lateral(n):
    d=_base(n); d['dst_port']=np.random.choice([445,3389,135,139],n)
    d['unique_dst_ips']=np.random.randint(10,100,n)
    d['connection_frequency']=np.random.normal(200,30,n).clip(0)
    d['syn_flag_count']=np.random.poisson(20,n)
    d['flow_bytes_per_s']=np.random.normal(50000,5000,n).clip(0)
    d['label']=3; return d

def _preenc(n):
    d=_base(n)
    d['flow_bytes_per_s']=np.random.normal(100000,10000,n).clip(0)
    d['flow_packets_per_s']=np.random.normal(1000,100,n).clip(0)
    d['packet_length_entropy']=np.random.normal(7.8,0.1,n).clip(0,8)
    d['packet_length_variance']=np.random.normal(5000,500,n).clip(0)
    d['psh_flag_count']=np.random.poisson(50,n); d['label']=4; return d

def generate_dataset(n_samples=10000,save_path="data/network_traffic.csv"):
    pc=n_samples//5
    df=pd.concat([_base(pc),_recon(pc),_c2(pc),_lateral(pc),_preenc(pc)],
                 ignore_index=True).sample(frac=1,random_state=RANDOM_STATE)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    df.to_csv(save_path,index=False)
    print(f"[+] Dataset → {save_path} | {df.shape}"); return df
