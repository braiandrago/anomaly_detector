import os
import subprocess

print("\n🚀 Ejecutando flujo completo univariado...\n")

# Paso 1: Simular datos nuevos desde el sistema industrial
print("[1/2] 🔄 Simulando datos desde test_uni.csv...")
subprocess.run(["python", "api_simulation/call_api.py"], check=True)

# Paso 2: Ejecutar inferencia y generar reporte
print("\n[2/2] 📈 Ejecutando inferencia con modelo univariado...")
subprocess.run(["python", "src/inference.py", "univariado", "data/raw/live_uni.csv"], check=True)

print("\n📈 Flujo completo ejecutado. Reporte disponible en: reports/report_uni.html")


