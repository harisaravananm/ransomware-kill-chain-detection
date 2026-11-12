# config.py
FEATURES = ['duration','protocol','dst_port','flow_bytes_per_s','flow_packets_per_s',
 'fwd_pkt_len_mean','bwd_pkt_len_mean','pkt_len_var','iat_mean','iat_std','iat_max',
 'fwd_iat_mean','bwd_iat_mean','syn_flag_count','rst_flag_count','psh_flag_count','ack_flag_count']
KILL_CHAIN_STAGES = {
    0:'Normal',1:'Reconnaissance',
    2:'C2_Beaconing',3:'Lateral_Movement',4:'Pre_Encryption'
}
SEVERITY = {0:'None',1:'Low',2:'Medium',3:'High',4:'Critical'}
SEQUENCE_LENGTH = 10
RF_N_ESTIMATORS = 100
LSTM_EPOCHS     = 30
BATCH_SIZE      = 32
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
DATA_PATH       = "data/network_traffic.csv"
MODEL_RF_PATH   = "output/rf_model.pkl"
MODEL_LSTM_PATH = "output/lstm_model.h5"
FIGURES_PATH    = "output/figures/"
ALERTS_LOG      = "output/alerts.json"
