"""
Script de Entrenamiento YOLO Personalizado para VisionVR
=======================================================
Este script te permite entrenar tu propio modelo de Inteligencia Artificial
(YOLOv8) para reconocer los componentes electrónicos de tu universidad.

PASOS PREVIOS:
1. Necesitas un Dataset (fotos etiquetadas). Puedes conseguir uno en:
   https://universe.roboflow.com/
2. Descarga el dataset en formato "YOLOv8".
3. Coloca la carpeta del dataset descargada junto a este script,
   asegurándote de que exista un archivo `data.yaml` dentro de ella.

CÓMO EJECUTAR:
python train_custom.py

Cuando termine el entrenamiento (puede tardar horas dependiendo de tu PC),
el nuevo modelo se guardará automáticamente como `visionvr_custom.pt` 
y el servidor `server.py` lo detectará la próxima vez que arranque.
"""

import os
import shutil
from ultralytics import YOLO

def main():
    print("🤖 Iniciando Entrenamiento de YOLO Personalizado para VisionVR 🤖")
    
    # 1. Verificar si existe el archivo de configuración del dataset
    # Reemplaza 'dataset/data.yaml' por la ruta real de tu dataset
    dataset_yaml = "dataset/data.yaml"
    
    if not os.path.exists(dataset_yaml):
        print(f"\n❌ ERROR: No se encontró el archivo '{dataset_yaml}'.")
        print("Asegúrate de haber descargado tu dataset (ej: desde Roboflow)")
        print("y de haberlo colocado en una carpeta llamada 'dataset' junto a este script.")
        print("El archivo 'data.yaml' debe estar dentro de esa carpeta.\n")
        return

    print(f"\n✅ Dataset detectado en: {dataset_yaml}")
    print("⏳ Cargando modelo base YOLOv8n (rápido)...")
    
    # 2. Cargar el modelo pre-entrenado base (nano)
    modelo = YOLO("yolov8n.pt")
    
    print("\n🚀 ¡Iniciando entrenamiento! (Esto puede tardar un buen rato...)")
    print("Puedes cancelar en cualquier momento con Ctrl+C.\n")
    
    # 3. Entrenar el modelo
    # Parámetros básicos para empezar:
    # epochs: número de veces que repasará todas las fotos (recomiendo 30-50 para proyectos finales)
    # imgsz: tamaño al que redimensiona las fotos (640 es estándar)
    # device: 'cpu' si no tienes tarjeta gráfica NVIDIA. Si tienes, pon '0' o quita el parámetro.
    resultados = modelo.train(
        data=dataset_yaml,
        epochs=10,        # <- Cambia a 30 o 50 cuando quieras el modelo final
        imgsz=640,
        batch=8,
        name="visionvr_entrenamiento"
    )
    
    print("\n✅ ¡Entrenamiento completado!")
    
    # 4. Mover el modelo resultante a la raíz para que server.py lo detecte
    # YOLO guarda el mejor modelo en runs/detect/visionvr_entrenamiento/weights/best.pt
    try:
        best_model_path = "runs/detect/visionvr_entrenamiento/weights/best.pt"
        if os.path.exists(best_model_path):
            destino = "visionvr_custom.pt"
            shutil.copy(best_model_path, destino)
            print(f"🎯 El nuevo modelo ha sido guardado como: {destino}")
            print("🎉 ¡La próxima vez que inicies start.py, usará tu modelo inteligente!")
        else:
            print("⚠️ No se encontró el modelo final en la ruta esperada.")
    except Exception as e:
        print(f"Error al mover el modelo: {e}")

if __name__ == "__main__":
    main()
