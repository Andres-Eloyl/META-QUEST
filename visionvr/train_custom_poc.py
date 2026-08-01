import os
import shutil
from ultralytics import YOLO

print("==================================================")
print("  🤖 INICIANDO ENTRENAMIENTO DE IA (Prueba Rápida)")
print("==================================================")

# 1. Cargamos un modelo pre-entrenado muy ligero (Nano) para que sea rápido
print("\n[1/3] Cargando arquitectura base (yolov8n.pt)...")
modelo = YOLO("yolov8n.pt")

# 2. Entrenamos el modelo
# Para este ejemplo, usamos 'coco8.yaml', que es un dataset minúsculo que 
# Ultralytics descarga automáticamente. Incluye objetos de oficina/laboratorio 
# como: Laptops, Ratones, Libros, Tazas, Teclados y Celulares.
print("\n[2/3] Entrenando con imágenes de escritorio/laboratorio...")
print("      (Configurado a solo 5 iteraciones/epochs para la prueba)")

# Entrenar (en un caso real, cambiaríamos coco8.yaml por tu_dataset_robotica.yaml)
modelo.train(
    data="coco8.yaml",
    epochs=5,           # 5 pasadas de aprendizaje
    imgsz=320,          # Resolución baja para entrenar rápido en CPU
    batch=2,            # Imágenes por lote
    name="entrenamiento_lab",
    verbose=False       # Ocultar exceso de logs
)

# 3. Exportar e integrar a VisionVR
print("\n[3/3] Exportando el modelo personalizado...")
ruta_modelo_entrenado = "runs/detect/entrenamiento_lab/weights/best.pt"
ruta_destino = "visionvr_custom.pt"

if os.path.exists(ruta_modelo_entrenado):
    # Sobrescribir si ya existe un modelo anterior
    if os.path.exists(ruta_destino):
        os.remove(ruta_destino)
        
    shutil.copy(ruta_modelo_entrenado, ruta_destino)
    print("\n✅ ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
    print(f"El archivo cerebral de la IA se guardó en: {ruta_destino}")
    print("\n💡 La próxima vez que inicies 'server.py', el sistema detectará")
    print("   este nuevo archivo y lo usará en tus gafas en lugar del genérico.")
else:
    print("\n❌ Error: No se encontró el modelo entrenado.")
