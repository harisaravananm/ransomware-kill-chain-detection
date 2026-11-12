import time, json, statistics as st
from dilithium_py.ml_dsa import ML_DSA_65
from kyber_py.ml_kem import ML_KEM_768

N = 20
def bench(f):
    ts, r = [], None
    for _ in range(N):
        a = time.perf_counter(); r = f(); ts.append((time.perf_counter() - a) * 1000)
    return st.mean(ts), st.stdev(ts), r

alert = json.dumps({"id": 1, "stage": "Lateral_Movement", "conf": 0.93}).encode()
m, s, (pk, sk) = bench(ML_DSA_65.keygen);                      print(f"ML-DSA-65 keygen : {m:.2f} +/- {s:.2f} ms")
m, s, sig = bench(lambda: ML_DSA_65.sign(sk, alert));          print(f"ML-DSA-65 sign   : {m:.2f} +/- {s:.2f} ms")
m, s, ok = bench(lambda: ML_DSA_65.verify(pk, alert, sig));    print(f"ML-DSA-65 verify : {m:.2f} +/- {s:.2f} ms  valid={ok}")
m, s, (ek, dk) = bench(ML_KEM_768.keygen);                     print(f"ML-KEM-768 keygen: {m:.2f} +/- {s:.2f} ms")
m, s, (key, ct) = bench(lambda: ML_KEM_768.encaps(ek));        print(f"ML-KEM-768 encaps: {m:.2f} +/- {s:.2f} ms")
m, s, key2 = bench(lambda: ML_KEM_768.decaps(dk, ct));         print(f"ML-KEM-768 decaps: {m:.2f} +/- {s:.2f} ms  keys_match={key == key2}")
print("sizes (bytes): pk", len(pk), "sig", len(sig), "ek", len(ek), "ct", len(ct))