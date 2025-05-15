import os
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte de Anomalías - {{ tipo }}</title>
</head>
<body>
    <h1>Reporte de Anomalías - {{ tipo }}</h1>

    <h2>1. Patrón normal aprendido (train)</h2>
    <img src="{{ grafico_train }}" alt="Gráfico entrenamiento" width="800">
    <h3>Zoom de los últimos 20 registros</h3>
    <img src="{{ grafico_train_zoom }}" alt="Zoom entrenamiento" width="800">

    <h2>2. Datos recientes desde producción</h2>
    <img src="{{ grafico_live }}" alt="Gráfico live" width="800">

    <h2>3. Detección de Anomalías</h2>
    <h3>Umbral de Detección: {{ threshold }}</h3>
    <h3>Total de Anomalías Detectadas: {{ total_anomalias }}</h3>
    <img src="{{ grafico_anomalias }}" alt="Gráfico de Anomalías" width="800">

    <h3>Recomendación:</h3>
    <p>{{ recomendacion }}</p>
</body>
</html>
"""

def generar_reporte(tipo, threshold=None):
    tipo_short = "uni" if tipo == "univariado" else "multi"

    print(f"[INFO] Generando reporte para tipo: {tipo} ({tipo_short})")

    csv_result = os.path.join(REPORTS_DIR, f"anomaly_result_{tipo_short}.csv")
    df_result = pd.read_csv(csv_result, parse_dates=["timestamp"])

    # Usar threshold fijo si se pasó, sino calcularlo
    if threshold is None:
        threshold = df_result["reconstruction_error"].quantile(0.95)

    total_anomalias = df_result["anomaly"].sum()

    train_path = f"data/raw/train_{tipo_short}.csv"
    print(f"[INFO] Leyendo datos de entrenamiento desde: {train_path}")
    df_train = pd.read_csv(train_path, parse_dates=["timestamp"])

    live_path = f"data/raw/live_{tipo_short}.csv"
    print(f"[INFO] Leyendo datos de producción desde: {live_path}")
    df_live = pd.read_csv(live_path, parse_dates=["timestamp"])

    # Gráfico 1: entrenamiento
    path_train = os.path.join(REPORTS_DIR, f"train_plot_{tipo}.png")
    print(f"[INFO] Generando gráfico de entrenamiento: {path_train}")
    plt.figure(figsize=(12, 4))
    try:
        if tipo == "univariado":
            plt.plot(df_train["timestamp"], df_train["temperature"], label="Temperatura")
        else:
            for col in ["temperature", "pressure", "flowrate"]:
                plt.plot(df_train["timestamp"], df_train[col], label=col)
        plt.title("Comportamiento normal - Entrenamiento")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_train)
    finally:
        plt.close()

    # Zoom de entrenamiento
    path_train_zoom = os.path.join(REPORTS_DIR, f"train_plot_zoom_{tipo}.png")
    print(f"[INFO] Generando zoom del entrenamiento: {path_train_zoom}")
    plt.figure(figsize=(12, 4))
    try:
        df_zoom = df_train.tail(20)
        if tipo == "univariado":
            plt.plot(df_zoom["timestamp"], df_zoom["temperature"], label="Temperatura")
        else:
            for col in ["temperature", "pressure", "flowrate"]:
                plt.plot(df_zoom["timestamp"], df_zoom[col], label=col)
        plt.title("Zoom últimos 20 puntos - Entrenamiento")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_train_zoom)
    finally:
        plt.close()

    # Gráfico 2: live
    path_live = os.path.join(REPORTS_DIR, f"live_plot_{tipo}.png")
    print(f"[INFO] Generando gráfico de producción: {path_live}")
    plt.figure(figsize=(12, 4))
    try:
        if tipo == "univariado":
            plt.plot(df_live["timestamp"], df_live["temperature"], label="Temperatura")
        else:
            for col in ["temperature", "pressure", "flowrate"]:
                plt.plot(df_live["timestamp"], df_live[col], label=col)
        plt.title("Datos Recientes desde Producción")
        plt.xlabel("Tiempo")
        plt.ylabel("Valor")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path_live)
    finally:
        plt.close()

    # Gráfico 3: errores y anomalías
    path_anomalias = os.path.join(REPORTS_DIR, f"anomaly_plot_{tipo}.png")
    print(f"[INFO] Generando gráfico de anomalías: {path_anomalias}")
    plt.figure(figsize=(12, 4))
    try:
        plt.plot(df_result["timestamp"], df_result["reconstruction_error"], label="Error de Reconstrucción")
        plt.axhline(threshold, color="r", linestyle="--", label=f"Umbral ({threshold:.4f})")
        plt.fill_between(df_result["timestamp"], 0, df_result["reconstruction_error"], where=df_result["anomaly"] == 1,
                         color="red", alpha=0.3, label="Anomalías")
        plt.title("Detección de Anomalías")
        plt.xlabel("Tiempo")
        plt.ylabel("Error de Reconstrucción")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path_anomalias)
    finally:
        plt.close()

    if total_anomalias > 0:
        recomendacion = "Se recomienda revisar condiciones operativas y sensores involucrados en el proceso."
    else:
        recomendacion = "No se detectaron anomalías relevantes. Continuar operación normal."

    template = Template(HTML_TEMPLATE)
    html = template.render(
        tipo=tipo,
        threshold=f"{threshold:.4f}",
        total_anomalias=int(total_anomalias),
        grafico_train=os.path.basename(path_train),
        grafico_train_zoom=os.path.basename(path_train_zoom),
        grafico_live=os.path.basename(path_live),
        grafico_anomalias=os.path.basename(path_anomalias),
        recomendacion=recomendacion
    )

    output_html = os.path.join(REPORTS_DIR, f"report_{tipo_short}.html")
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"\n🔝 Reporte generado exitosamente: {output_html}")






