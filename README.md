# 🔎 Sistema de Detección de Anomalías en Series de Tiempo con Autoencoders (Univariado y Multivariado)

Este proyecto implementa un sistema de detección de anomalías en series temporales, utilizando Autoencoders Recurrentes (RNN con LSTM) con Keras. Se soportan tanto modelos **univariados** como **multivariados**, pensados para ser utilizados en entornos industriales de monitoreo de procesos.

Particularmente se simula un sistema industrial que genera datos de proceso cada minuto, los cuales son analizados cada 15 minutos por un modelo de detección de anomalías basado en Autoencoders. El sistema cuenta con versiones univariadas y multivariadas.

---

## 📁 Estructura del proyecto

```
anomaly_detector/
├── api_simulation/        # Simula la llamada a una API en producción
│   └── call_api.py
├── data/                  # Datos de entrada (simulados)
│   └── raw/
│       ├── train_uni.csv
│       ├── test_uni.csv
│       ├── train_multi.csv
│       └── test_multi.csv
├── models/                # Modelos entrenados y scalers
│   ├── model_uni.h5
│   ├── model_multi.h5
│   ├── scaler_uni.pkl
│   └── threshold_multi.pkl
├── reports/               # Salidas generadas (HTML, PNG, CSV)
│   ├── report_uni.h5
│   └── report_multi.h5
├── src/                   # Módulos principales
│   ├── train_autoencoder.py
│   ├── inference.py
│   └── generate_report.py
├── run_uni.py             # Ejecuta el flujo completo univariado
├── run_multi.py           # Ejecuta el flujo completo multivariado
├── data_simulation.py     # Script para generar los datasets
├── README.md              # Este archivo
└── .gitignore
```
---

## ⚙️ Flujo de Trabajo

### 1. Generar datos simulados
```bash
python src/data_simulation.py
```
Esto genera los datasets `train` y `test`, tanto para el caso univariado como multivariado. El set de test contiene anomalías realistas.

---

### 2. Entrenar el modelo (ej. univariado)
```bash
python src/train_autoencoder.py
```
Este script:
- Carga `train_uni.csv`
- Escala los datos
- Entrena un Autoencoder LSTM
- Guarda el modelo y el scaler

> Cambiá la variable `TIPO = "multivariado"` si deseás entrenar el modelo multivariado.

---

### 3. Simular datos en vivo y correr inferencia
```bash
python run_uni.py
```
Este script:
- Ejecuta `call_api.py` para generar `live_uni.csv`  (simulación)
- Llama a `inference.py` para detectar anomalías
- Genera el reporte `report_uni.html`

---

### Opcional - Correr inferencia manualmente
```bash
# Univariado
data/raw/test_uni.csv
python src/inference.py univariado data/raw/test_uni.csv

# Multivariado
data/raw/test_multi.csv
python src/inference.py multivariado data/raw/test_multi.csv
```)

## 📊 Reporte generado

Cada vez que se ejecuta una inferencia se genera un HTML con:
- Gráfico de la serie original de entrenamiento
- Zoom de los últimos 20 puntos del entrenamiento
- Gráfico de los datos actuales (live: simulación de datos de producción) 
- Detección de anomalías con umbral y sombreado
- Recomendación en lenguaje natural

---

## 🧠 Modelo utilizado
- Autoencoder con LSTM
- Secuencias de longitud 30 (ej: 30 minutos de datos cada 1 min)

## 🧠 Lógica del Umbral
El sistema detecta anomalías comparando el **error de reconstrucción** de cada secuencia con un umbral fijo calculado así:

```python
threshold = np.mean(errors) + 3 * np.std(errors)
```

Esto se calcula sobre los errores del set de entrenamiento. Todas las secuencias cuyo error supere ese umbral son marcadas como anómalas.

> ✅ Recomendado para producción: mantener el umbral fijo guardado con el modelo para evitar recalibraciones dinámicas.

- Monitoreo de variables de planta
- Datos univariados o multivariados
- Integración con sistemas tipo SCADA o historiadores como Honeywell

---

## 🧪 Cómo probar anomalías
- Modificá el dataset `test_uni.csv` a mano o ajustá `data_simulation.py` para simular condiciones extremas.
- Ejecutá `call_api.py` para traer una secuencia reciente de test.
- Corré `run_uni.py` y visualizá `report_uni.html`.

---

## 🚀 Próximos pasos
- Agregar lógica de recalibración semanal automática
- Integración con una API real (REST)
- Registro en MLflow y alertas en tiempo real

---

## 🧑‍💻 Requisitos

### 🧪 Crear entorno virtual con Conda

```bash
conda create -n anomaly_env python=3.10
conda activate anomaly_env
pip install -r requirements.txt
```bash
pip install -r requirements.txt
```

### `requirements.txt` contiene:
- tensorflow
- pandas
- numpy
- scikit-learn
- matplotlib
- jinja2
- joblib

## 📬 Contacto
Este proyecto fue desarrollado como parte de una arquitectura reproducible para detección temprana de fallos en entornos industriales.
---



