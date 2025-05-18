import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import numpy as np
from config import SEQ_LENGTH, N_SECUENCIAS

# Configuración
ORIGEN_UNI = "data/raw/test_uni.csv"
ORIGEN_MULTI = "data/raw/test_multi.csv"
DESTINO_UNI = "data/raw/live_uni.csv"
DESTINO_MULTI = "data/raw/live_multi.csv"
#SEQ_LENGTH = 30
#N_SECUENCIAS = 10  # solo 1 secuencia por ejecución / más de una secuencia (cambiar según requerimiento)

# Cargar datasets simulados (test)
df_uni = pd.read_csv(ORIGEN_UNI, parse_dates=["timestamp"])
df_multi = pd.read_csv(ORIGEN_MULTI, parse_dates=["timestamp"])

# Validar que hay suficientes datos
total_necesarios = SEQ_LENGTH + N_SECUENCIAS
if len(df_uni) < total_necesarios or len(df_multi) < total_necesarios:
    raise ValueError("❌ No hay suficientes registros para generar una secuencia.")

# Extraer una ventana aleatoria consecutiva de datos
inicio_uni = np.random.randint(0, len(df_uni) - total_necesarios + 1)
inicio_multi = np.random.randint(0, len(df_multi) - total_necesarios + 1)
ventana_uni = df_uni.iloc[inicio_uni:inicio_uni + total_necesarios].copy()
ventana_multi = df_multi.iloc[inicio_multi:inicio_multi + total_necesarios].copy()

# Guardar archivos simulando consulta al sistema industrial
ventana_uni.to_csv(DESTINO_UNI, index=False)
ventana_multi.to_csv(DESTINO_MULTI, index=False)

print(f"✅ Archivos generados:\n - {DESTINO_UNI}\n - {DESTINO_MULTI}")
print(f"ℹ️  Contienen {total_necesarios} registros consecutivos para 1 secuencia de {SEQ_LENGTH} minutos.")


