import numpy as np

def make_sequences(X, y, seq_len=10):
    """Windows of seq_len rows from the SAME class, original row order kept.
    The label is the class of the window's last row.
    Returns Xs (N, seq_len, F), ys (N,), last_idx (N,) = row index of each
    window's last row, so RF predictions can be aligned with the windows."""
    X = np.asarray(X); y = np.asarray(y)
    Xs, ys, last = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]            # rows of this class, in original order
        for k in range(len(idx) - seq_len + 1):
            w = idx[k:k + seq_len]
            Xs.append(X[w]); ys.append(c); last.append(w[-1])
    return np.array(Xs, dtype=np.float32), np.array(ys), np.array(last)