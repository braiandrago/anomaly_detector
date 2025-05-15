import os
import subprocess

print("\n🚀 Ejecutando flujo completo multivariado...\n")

# Paso 1: Simular datos nuevos desde el sistema industrial
print("[1/2] 🔄 Simulando datos desde test_multi.csv...")
subprocess.run(["python", "api_simulation/call_api.py"], check=True)

# Paso 2: Ejecutar inferencia y generar reporte
print("\n[2/2] 📈 Ejecutando inferencia con modelo multivariado...")
subprocess.run(["python", "src/inference.py", "multivariado", "data/raw/live_multi.csv"], check=True)

print("\n📈 Flujo multivariado ejecutado. Reporte disponible en: reports/report_multi.html")


