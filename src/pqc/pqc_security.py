import json, time
from dilithium_py.ml_dsa import ML_DSA_65
from kyber_py.ml_kem import ML_KEM_768

class PQCAlertSystem:
    def __init__(self):
        self.sig_pk, self.sig_sk = ML_DSA_65.keygen()
        self.ek, self.dk = ML_KEM_768.keygen()

    def sign_alert(self, alert: dict) -> dict:
        msg = json.dumps(alert, sort_keys=True, default=str).encode()
        t = time.perf_counter()
        sig = ML_DSA_65.sign(self.sig_sk, msg)
        ms = (time.perf_counter() - t) * 1000
        _key, ct = ML_KEM_768.encaps(self.ek)
        return {"alert": alert, "signature": sig.hex(), "kem_ciphertext": ct.hex(),
                "sign_ms": round(ms, 1), "algorithms": "ML-DSA-65 + ML-KEM-768"}

    def verify_alert(self, signed: dict) -> bool:
        msg = json.dumps(signed["alert"], sort_keys=True, default=str).encode()
        return ML_DSA_65.verify(self.sig_pk, msg, bytes.fromhex(signed["signature"]))

    def secure_alert(self, alert):
        return self.sign_alert(alert)

    sign = sign_alert
    verify = verify_alert