import json, time, os, sys
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(__file__),'..'))
from config import KILL_CHAIN_STAGES, ALERTS_LOG, SEVERITY
from src.pqc.pqc_security import PQCAlertSystem

class AlertManager:
    def __init__(self):
        self.pqc=PQCAlertSystem(); self.alerts=[]
        os.makedirs("output",exist_ok=True)

    def generate_alert(self,idx,stage_id,confidence,
                       shap_r,lime_r=None,src_ip=None):
        if stage_id==0: return None
        alert={
            "alert_id":   f"ALERT-{int(time.time())}-{idx}",
            "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "src_ip":     src_ip or f"192.168.{np.random.randint(1,10)}.{np.random.randint(2,254)}",
            "stage":      KILL_CHAIN_STAGES[stage_id],
            "stage_id":   int(stage_id),
            "severity":   SEVERITY[stage_id],
            "confidence": round(float(confidence)*100,2),
            "shap":       shap_r,
            "lime":       lime_r or {},
            "action":     "BLOCK" if stage_id>=3 else "MONITOR"
        }
        secured=self.pqc.secure_alert(alert)
        secured["verified"]=self.pqc.verify_alert(secured)
        self.alerts.append(secured)
        self._print(alert,secured["verified"])
        return secured

    def _print(self,a,v):
        s="="*55
        print(f"\n{s}")
        print(f"  ALERT [{a['severity']}] {a['stage']}")
        print(f"  IP:{a['src_ip']}  Conf:{a['confidence']}%")
        print(f"  Action:{a['action']}  PQC:{'OK' if v else 'FAIL'}")
        print(f"  SHAP:{list(a['shap'].keys())}")
        print(f"  LIME:{list(a['lime'].keys())}")
        print(s)

    def save_log(self):
        with open(ALERTS_LOG,'w') as f:
            json.dump(self.alerts,f,indent=2)
        print(f"[+] {len(self.alerts)} alerts → {ALERTS_LOG}")
