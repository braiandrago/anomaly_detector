import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import load
from tensorflow import keras
from sklearn.metrics import mean_squared_error

from config import SEQ_LENGTH

from generate_report import generar_reporte

# Configuraciones
#SEQ_LENGTH = 30

def run_inference(tipo, path_csv):
    tipo_short = "uni" if tipo == "univariado" else "multi"
    model_path = f"models/model_{tipo_short}.h5"
    scaler_path = f"models/scaler_{tipo_short}.pkl"
    threshold_path = f"models/threshold_{tipo_short}.pkl"
    output_path = f"reports/anomaly_result_{tipo_short}.csv"

    # Cargar modelo, scaler y umbral
    print(f"[INFO] Cargando modelo desde: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"[INFO] Cargando scaler desde: {scaler_path}")
    scaler = load(scaler_path)

    print(f"[INFO] Cargando umbral desde: {threshold_path}")
    threshold = load(threshold_path)

    # Leer datos
    print(f"[INFO] Leyendo datos de entrada desde: {path_csv}")
    df = pd.read_csv(path_csv, parse_dates=["timestamp"])
    if tipo == "univariado":
        data = df[["temperature"]].values
    else:
        data = df[["temperature", "pressure", "flowrate"]].values

    if len(data) < SEQ_LENGTH:
        raise ValueError("❌ No hay suficientes datos para generar una secuencia.")

    # Escalar datos
    data_scaled = scaler.transform(data)

    # Crear secuencia
    X = []
    for i in range(len(data_scaled) - SEQ_LENGTH + 1):
        X.append(data_scaled[i:i + SEQ_LENGTH])
    X = np.array(X)

    # Predicción
    X_pred = model.predict(X)
    errors = np.mean(np.square(X - X_pred), axis=(1, 2))

    result = pd.DataFrame({
        "timestamp": df["timestamp"].iloc[SEQ_LENGTH - 1:].values,
        "reconstruction_error": errors
    })

    # Extra: calcular contribución por variable en caso multivariado
    if tipo == "multivariado":
        var_names = ["temperature", "pressure", "flowrate"]
        contribs = np.mean(np.square(X - X_pred), axis=1)  # shape: (n_seq, n_features)
        for i, name in enumerate(var_names):
            result[f"error_{name}"] = contribs[:, i]
        result["max_error_var"] = result[[f"error_{v}" for v in var_names]].idxmax(axis=1)

    # Aplicar umbral cargado
    result["anomaly"] = result["reconstruction_error"] > threshold
    result.to_csv(output_path, index=False)

    print(f"[INFO] Resultados guardados en: {output_path}")

    generar_reporte(tipo, threshold=threshold)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python src/inference.py [univariado|multivariado] [ruta_csv]")
        sys.exit(1)

    tipo = sys.argv[1].lower()
    path_csv = sys.argv[2]

    if tipo not in ["univariado", "multivariado"]:
        print("❌ Tipo inválido. Debe ser 'univariado' o 'multivariado'")
        sys.exit(1)

    run_inference(tipo, path_csv)


