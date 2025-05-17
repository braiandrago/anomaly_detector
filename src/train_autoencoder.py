import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from joblib import dump

# Configuraciones
SEQ_LENGTH = 30
TIPO = "univariado"  # cambiar a "univariado / multivariado" si entrenás ese modelo
tipo_short = "uni" if TIPO == "univariado" else "multi"
data_path = f"data/raw/train_{tipo_short}.csv"
model_path = f"models/model_{tipo_short}.h5"
scaler_path = f"models/scaler_{tipo_short}.pkl"
threshold_path = f"models/threshold_{tipo_short}.pkl"

# Crear carpeta si no existe
os.makedirs("models", exist_ok=True)

# Cargar datos
print(f"[INFO] Cargando datos desde: {data_path}")
df = pd.read_csv(data_path, parse_dates=["timestamp"])

if TIPO == "univariado":
    data = df[["temperature"]].values
else:
    data = df[["temperature", "pressure", "flowrate"]].values

# Escalar datos
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)
dump(scaler, scaler_path)
print(f"[INFO] Scaler guardado en: {scaler_path}")

# Crear secuencias
X = []
for i in range(len(data_scaled) - SEQ_LENGTH):
    X.append(data_scaled[i:i + SEQ_LENGTH])
X = np.array(X)

# Construir modelo autoencoder
n_features = X.shape[2]
model = keras.Sequential([
    layers.Input(shape=(SEQ_LENGTH, n_features)),
    layers.LSTM(16, activation="relu", return_sequences=False),
    layers.RepeatVector(SEQ_LENGTH),
    layers.LSTM(16, activation="relu", return_sequences=True),
    layers.TimeDistributed(layers.Dense(n_features))
])

model.compile(optimizer="adam", loss=MeanSquaredError())
print("[INFO] Entrenando modelo...")
history = model.fit(X, X, epochs=20, batch_size=32, validation_split=0.1, shuffle=False)
model.save(model_path)
print(f"[INFO] Modelo guardado en: {model_path}")

# Calcular y guardar threshold fijo
X_pred = model.predict(X)
recon_errors = np.mean((X - X_pred) ** 2, axis=(1, 2))
threshold = np.mean(recon_errors) + 3 * np.std(recon_errors)
dump(threshold, threshold_path)
print(f"[INFO] Umbral guardado en: {threshold_path} -> {threshold:.6f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(history.history["loss"], label="Loss entrenamiento")
plt.plot(history.history["val_loss"], label="Loss validación")
plt.title("Evolución del entrenamiento")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"models/training_loss_{tipo_short}.png")
plt.savefig(f"reports/training_loss_{tipo_short}.png")
plt.close()

