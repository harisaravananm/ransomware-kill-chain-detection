import numpy as np, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from config import LSTM_EPOCHS, BATCH_SIZE, MODEL_LSTM_PATH
N_CLASSES = 5

def build_lstm(input_shape):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        BatchNormalization(), Dropout(0.3),
        LSTM(64, return_sequences=False),
        BatchNormalization(), Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lstm(Xtr, ytr, Xva, yva):
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    print("[*] Training LSTM...")
    ytr_c = to_categorical(ytr, N_CLASSES)
    yva_c = to_categorical(yva, N_CLASSES)
    model = build_lstm((Xtr.shape[1], Xtr.shape[2]))
    cb = [EarlyStopping(patience=5, restore_best_weights=True),
          ModelCheckpoint(MODEL_LSTM_PATH, save_best_only=True)]
    history = model.fit(Xtr, ytr_c, validation_data=(Xva, yva_c),
                        epochs=LSTM_EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=cb, verbose=1)
    print(f"[+] LSTM saved -> {MODEL_LSTM_PATH}")
    return model, history

def evaluate_lstm(model, X_seq, y_seq):
    from sklearn.metrics import accuracy_score
    yp = model.predict(X_seq, verbose=0)
    y_pred = np.argmax(yp, axis=1)
    acc = accuracy_score(y_seq, y_pred)
    print(f"\n[LSTM] Accuracy: {acc:.4f}")
    return y_pred, yp, acc

def load_lstm():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_LSTM_PATH)