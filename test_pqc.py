import sys
sys.modules['oqs'] = None
from src.pqc.pqc_security import PQCAlertSystem

p = PQCAlertSystem()
alert = {"id": 1, "stage": "Lateral_Movement", "confidence": 0.93}
signed = p.secure_alert(alert)
print("signed OK | sign_ms:", signed["sign_ms"], "| verify:", p.verify_alert(signed))
signed["alert"]["stage"] = "Normal"
print("after tampering, verify:", p.verify_alert(signed))

try:
    from src.alert_system import AlertManager
    m = AlertManager()
    m.generate_alert(0, 3, 0.93, {"dst_port": 0.04, "duration": 0.02}, {"rst_flag_count > -0.52": 0.03})
    m.save_log()
    print("AlertManager OK")
except Exception as e:
    import traceback; traceback.print_exc()

    