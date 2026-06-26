"""
VisionVR — Servidor Backend
============================
Autor: Andrés
Materia: Inteligencia Artificial
Motor: Flask + YOLOv8 + SQLite + OpenAI (modular)

Módulos activos:
  [x] Núcleo: detección de objetos con YOLO
  [x] Base de datos: registro de sesión en SQLite
  [ ] GPT: modo "explícame" (descomentar cuando llegues a esa capa)
"""

import base64
import os
import time
import sqlite3
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# ─── Configuración ────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Permite peticiones desde el Quest (diferente origen)
socketio = SocketIO(app, cors_allowed_origins="*")

# Modelo YOLO — usa el más ligero para que sea rápido en tu PC
# yolov8n.pt  = nano  (más rápido, menos preciso)  ← recomendado para empezar
# yolov8s.pt  = small (balance)
# yolov8m.pt  = medium (más preciso, más lento)
MODELO_PATH = "yolov8n.pt"
CONFIANZA_MINIMA = 0.5   # Filtra detecciones con menos del 50% de confianza
DB_PATH = "db/sesion.db"

# ─── Carga del modelo (solo una vez al iniciar el servidor) ───────────────────

print("Cargando modelo YOLO...")
modelo = YOLO(MODELO_PATH)
print(f"Modelo listo: {MODELO_PATH}")

# ─── Base de datos ────────────────────────────────────────────────────────────

def init_db():
    """Crea la tabla de detecciones si no existe."""
    os.makedirs("db", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detecciones (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            objeto    TEXT    NOT NULL,
            confianza REAL    NOT NULL,
            timestamp TEXT    NOT NULL
        )
    """)
    con.commit()
    con.close()

def guardar_deteccion(objeto: str, confianza: float):
    """Inserta una detección en la base de datos."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO detecciones (objeto, confianza, timestamp) VALUES (?, ?, ?)",
        (objeto, confianza, datetime.now().isoformat())
    )
    con.commit()
    con.close()

# ─── Utilidades ───────────────────────────────────────────────────────────────

def base64_a_imagen(data_url: str) -> np.ndarray:
    """
    Convierte una imagen en base64 (data URL) a un array de OpenCV.
    El frontend manda algo como: 'data:image/jpeg;base64,/9j/4AAQ...'
    """
    # Quita el prefijo 'data:image/...;base64,' si existe
    if "," in data_url:
        data_url = data_url.split(",")[1]

    img_bytes = base64.b64decode(data_url)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img_cv2

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.route("/ping", methods=["GET"])
def ping():
    """
    Endpoint de prueba. María puede usarlo para verificar
    que el servidor está corriendo antes de conectar el Quest.
    Abrir en browser: http://localhost:5000/ping
    """
    return jsonify({"status": "ok", "mensaje": "Servidor VisionVR activo"})


@app.route("/dashboard")
def dashboard():
    """Sirve el panel de estadísticas en vivo."""
    return send_file("dashboard.html")


@app.route("/detectar", methods=["POST"])
def detectar():
    """
    Endpoint principal.
    Recibe: { "imagen": "data:image/jpeg;base64,..." }
    Devuelve: { "detecciones": [...], "tiempo_ms": 123 }
    """
    datos = request.get_json()

    if not datos or "imagen" not in datos:
        return jsonify({"error": "Falta el campo 'imagen'"}), 400

    try:
        # 1. Decodificar imagen
        img = base64_a_imagen(datos["imagen"])
        if img is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        # 2. Correr YOLO
        inicio = time.time()
        resultados = modelo(img, conf=CONFIANZA_MINIMA, verbose=False)
        tiempo_ms = round((time.time() - inicio) * 1000)

        # 3. Parsear resultados
        detecciones = []
        for resultado in resultados:
            for box in resultado.boxes:
                objeto = modelo.names[int(box.cls)]
                confianza = round(float(box.conf), 3)

                detecciones.append({
                    "objeto": objeto,
                    "confianza": confianza,
                    # Coordenadas del bounding box (útil para el frontend)
                    "bbox": {
                        "x": round(float(box.xyxy[0][0])),
                        "y": round(float(box.xyxy[0][1])),
                        "w": round(float(box.xyxy[0][2]) - float(box.xyxy[0][0])),
                        "h": round(float(box.xyxy[0][3]) - float(box.xyxy[0][1]))
                    }
                })

                # Guardar en base de datos
                guardar_deteccion(objeto, confianza)

        return jsonify({
            "detecciones": detecciones,
            "tiempo_ms": tiempo_ms,
            "total": len(detecciones)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("detectar")
def handle_detectar(datos):
    """
    Recibe por WebSocket: { "imagen": "data:image/jpeg;base64,..." }
    Emite el evento "detecciones_resultado" con los objetos.
    """
    if not datos or "imagen" not in datos:
        emit("detecciones_resultado", {"error": "Falta el campo 'imagen'"})
        return

    try:
        # 1. Decodificar imagen
        img = base64_a_imagen(datos["imagen"])
        if img is None:
            emit("detecciones_resultado", {"error": "No se pudo decodificar la imagen"})
            return

        # 2. Correr YOLO
        inicio = time.time()
        resultados = modelo(img, conf=CONFIANZA_MINIMA, verbose=False)
        tiempo_ms = round((time.time() - inicio) * 1000)

        # 3. Parsear resultados
        detecciones = []
        for resultado in resultados:
            for box in resultado.boxes:
                objeto = modelo.names[int(box.cls)]
                confianza = round(float(box.conf), 3)

                detecciones.append({
                    "objeto": objeto,
                    "confianza": confianza,
                    "bbox": {
                        "x": round(float(box.xyxy[0][0])),
                        "y": round(float(box.xyxy[0][1])),
                        "w": round(float(box.xyxy[0][2]) - float(box.xyxy[0][0])),
                        "h": round(float(box.xyxy[0][3]) - float(box.xyxy[0][1]))
                    }
                })
                # Guardar en base de datos
                guardar_deteccion(objeto, confianza)

        emit("detecciones_resultado", {
            "detecciones": detecciones,
            "tiempo_ms": tiempo_ms,
            "total": len(detecciones)
        })

    except Exception as e:
        emit("detecciones_resultado", {"error": str(e)})


@app.route("/estadisticas", methods=["GET"])
def estadisticas():
    """
    Devuelve estadísticas de la sesión actual.
    El frontend puede mostrarlas en un panel flotante.
    """
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # Total de detecciones
    total = cur.execute("SELECT COUNT(*) FROM detecciones").fetchone()[0]

    # Objetos únicos
    unicos = cur.execute("SELECT COUNT(DISTINCT objeto) FROM detecciones").fetchone()[0]

    # Top 5 más detectados
    top = cur.execute("""
        SELECT objeto, COUNT(*) as veces, ROUND(AVG(confianza), 2) as conf_promedio
        FROM detecciones
        GROUP BY objeto
        ORDER BY veces DESC
        LIMIT 5
    """).fetchall()

    # Timeline: agrupado por hora:minuto:segundo (últimos 20 segundos con datos)
    timeline = cur.execute("""
        SELECT substr(timestamp, 12, 8) as hora, COUNT(*) 
        FROM detecciones 
        GROUP BY substr(timestamp, 12, 8)
        ORDER BY timestamp DESC
        LIMIT 20
    """).fetchall()
    timeline.reverse() # Cronológico de viejo a nuevo

    con.close()

    return jsonify({
        "total_detecciones": total,
        "objetos_unicos": unicos,
        "top_objetos": [
            {"objeto": r[0], "veces": r[1], "confianza_promedio": r[2]}
            for r in top
        ],
        "timeline": [
            {"hora": r[0], "conteo": r[1]}
            for r in timeline
        ]
    })


@app.route("/limpiar", methods=["POST"])
def limpiar():
    """Borra el historial de la sesión actual."""
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM detecciones")
    con.commit()
    con.close()
    return jsonify({"mensaje": "Historial borrado"})


@app.route("/exportar", methods=["GET"])
def exportar():
    """Genera un archivo CSV con todo el historial de la sesión."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    filas = cur.execute("SELECT id, objeto, confianza, timestamp FROM detecciones ORDER BY timestamp ASC").fetchall()
    con.close()

    def generar():
        yield "ID,Objeto,Confianza,Timestamp\n"
        for fila in filas:
            yield f"{fila[0]},{fila[1]},{fila[2]},{fila[3]}\n"

    # app.response_class permite enviar la respuesta generada sobre la marcha como un archivo descargable
    return app.response_class(
        generar(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sesion_visionvr.csv'}
    )


# ─── Módulo GPT — descomentar cuando llegues a esa capa ──────────────────────
#
# from openai import OpenAI
# cliente_gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
# @app.route("/explicar", methods=["POST"])
# def explicar():
#     """
#     Recibe un objeto detectado y devuelve una explicación de GPT.
#     Recibe: { "objeto": "laptop", "contexto": "laboratorio de computación" }
#     Devuelve: { "explicacion": "Una laptop es..." }
#     """
#     datos = request.get_json()
#     objeto = datos.get("objeto", "objeto desconocido")
#     contexto = datos.get("contexto", "entorno universitario")
#
#     respuesta = cliente_gpt.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{
#             "role": "user",
#             "content": (
#                 f"Explica brevemente qué es '{objeto}' en el contexto de '{contexto}'. "
#                 f"Máximo 2 oraciones, en español, lenguaje simple."
#             )
#         }],
#         max_tokens=100
#     )
#     return jsonify({"explicacion": respuesta.choices[0].message.content})

# ─── Arranque ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("\n" + "="*50)
    print("  VisionVR Backend corriendo")
    print("  http://localhost:5000/ping")
    print("  Comparte esta IP con María para el frontend")
    import socket
    ip = socket.gethostbyname(socket.gethostname())
    print(f"  IP en red local: http://{ip}:5000")
    print("="*50 + "\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)
