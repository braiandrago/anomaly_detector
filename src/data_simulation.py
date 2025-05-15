import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

os.makedirs("data/raw", exist_ok=True)

# Config
FREQ = "1min"
N_DIAS_TRAIN = 7
N_DIAS_TEST = 5
SEQ_LENGTH = 30
np.random.seed(42)

def generar_serie_estable(timestamp_inicio, minutos, ruido=0.3, base=25):
    timestamps = pd.date_range(start=timestamp_inicio, periods=minutos, freq=FREQ)
    temperatura = base + np.random.normal(0, ruido, minutos)
    return pd.DataFrame({"timestamp": timestamps, "temperature": temperatura})

def generar_serie_test(timestamp_inicio, minutos, base=25, ruido_normal=0.3, n_bloques=25, duracion_bloque=15):
    timestamps = pd.date_range(start=timestamp_inicio, periods=minutos, freq=FREQ)
    temperatura = base + np.random.normal(0, ruido_normal, minutos)
    df = pd.DataFrame({"timestamp": timestamps, "temperature": temperatura})

    for i in range(n_bloques):
        start = np.random.randint(0, minutos - duracion_bloque)
        tipo = np.random.choice(["alta", "baja"], p=[0.5, 0.5])
        if tipo == "alta":
            valor_anomalia = np.random.uniform(45, 50)  # 🔺 Picos altos
        else:
            valor_anomalia = np.random.uniform(0, 5)    # 🔻 Valles extremos

        df.loc[start:start + duracion_bloque, "temperature"] = valor_anomalia

    return df


# ---------- UNIVARIADO ----------
min_train = N_DIAS_TRAIN * 24 * 60
min_test = N_DIAS_TEST * 24 * 60

start_train = datetime.now() - timedelta(days=N_DIAS_TRAIN + N_DIAS_TEST)
start_test = datetime.now() - timedelta(days=N_DIAS_TEST)

# Entrenamiento (estable)
df_train_uni = generar_serie_estable(start_train, min_train)
df_train_uni.to_csv("data/raw/train_uni.csv", index=False)

# Test con múltiples anomalías (altas y bajas)
df_test_uni = generar_serie_test(start_test, min_test)
df_test_uni.to_csv("data/raw/test_uni.csv", index=False)

# ---------- MULTIVARIADO ----------
def generar_multivariado(timestamp_inicio, minutos):
    timestamps = pd.date_range(start=timestamp_inicio, periods=minutos, freq=FREQ)
    temperatura = 25 + np.random.normal(0, 0.2, minutos)
    presion = 5 + np.random.normal(0, 0.05, minutos)
    caudal = 100 + np.random.normal(0, 1.0, minutos)
    return pd.DataFrame({
        "timestamp": timestamps,
        "temperature": temperatura,
        "pressure": presion,
        "flowrate": caudal
    })

def insertar_anomalias_multi(df, n_bloques=15, block_size=15):
    for _ in range(n_bloques):
        start = np.random.randint(0, len(df) - block_size)
        for col in ["temperature", "pressure", "flowrate"]:
            tipo = np.random.choice(["alta", "baja"])
            if tipo == "alta":
                if col == "temperature":
                    valor = np.random.uniform(45, 50)
                elif col == "pressure":
                    valor = np.random.uniform(6, 7)
                else:  # flowrate
                    valor = np.random.uniform(115, 130)
            else:
                if col == "temperature":
                    valor = np.random.uniform(0, 5)
                elif col == "pressure":
                    valor = np.random.uniform(3.5, 4.0)
                else:  # flowrate
                    valor = np.random.uniform(80, 85)
            df.loc[start:start + block_size, col] = valor
    return df


# Entrenamiento multivariado
train_multi = generar_multivariado(start_train, min_train)
train_multi.to_csv("data/raw/train_multi.csv", index=False)

# Test multivariado con anomalías
multi_test = generar_multivariado(start_test, min_test)
multi_test = insertar_anomalias_multi(multi_test, n_bloques=15, block_size=15)
multi_test.to_csv("data/raw/test_multi.csv", index=False)

print("\u2705 Datos simulados guardados en data/raw/")





