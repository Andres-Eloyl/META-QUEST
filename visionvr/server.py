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
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# ─── Configuración & Hilos CPU ────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Permite peticiones desde el Quest (diferente origen)
socketio = SocketIO(app, cors_allowed_origins="*")

# Optimizar hilos CPU
num_cpus = os.cpu_count() or 4
torch.set_num_threads(num_cpus)
print(f"⚡ Hilos CPU PyTorch optimizados a: {num_cpus}")

# Modelo YOLO & ONNX
CUSTOM_MODEL_PATH = "visionvr_custom.pt"
DEFAULT_MODEL_PATH = "yolov8m.pt"
ONNX_MODEL_PATH = "visionvr_model.onnx"

if os.path.exists(CUSTOM_MODEL_PATH):
    BASE_MODEL_PATH = CUSTOM_MODEL_PATH
    print(f"🚀 ¡Modelo personalizado detectado! Cargando {BASE_MODEL_PATH}")
else:
    BASE_MODEL_PATH = DEFAULT_MODEL_PATH
    print(f"Usando modelo genérico por defecto: {BASE_MODEL_PATH}")

ONNX_MODEL_PATH = BASE_MODEL_PATH.replace(".pt", ".onnx")
CONFIANZA_MINIMA = 0.5   # Filtra detecciones con menos del 50% de confianza
DB_PATH = "db/sesion.db"

# ─── Estado Multi-Usuario ─────────────────────────────────────────────────────
clientes_conectados = set()

# ─── Carga del modelo (solo una vez al iniciar el servidor) ───────────────────

print("Cargando modelo YOLO...")
# Intentar cargar ONNX si existe
if os.path.exists(ONNX_MODEL_PATH):
    MODELO_PATH = ONNX_MODEL_PATH
    print(f"⚡ Modelo ONNX detectado: {MODELO_PATH}")
else:
    MODELO_PATH = BASE_MODEL_PATH
    # Intentar exportación automática a ONNX si es posible
    try:
        print("⚡ Intentando exportar modelo YOLO a ONNX para aceleración 2-3x...")
        temp_m = YOLO(BASE_MODEL_PATH)
        temp_m.export(format="onnx", imgsz=640, dynamic=True)
        onnx_exportado = BASE_MODEL_PATH.replace(".pt", ".onnx")
        if os.path.exists(onnx_exportado):
            MODELO_PATH = onnx_exportado
            print(f"✅ Exportación a ONNX exitosa: {MODELO_PATH}")
    except Exception as e:
        print(f"⚠️ Exportación ONNX omitida ({e}). Usando PyTorch model: {BASE_MODEL_PATH}")
        MODELO_PATH = BASE_MODEL_PATH

modelo = YOLO(MODELO_PATH)
print(f"Modelo listo: {MODELO_PATH}")

# ─── Preprocesamiento, Auto-Rotación, ROI y Caché CV ────────────────────────

from collections import deque, Counter

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
MSE_UMBRAL_SIMILITUD = 35.0  # Umbral de diferencia para reusar detecciones

estado_sesiones = {}

def init_estado_sesion(session_id: str):
    if session_id not in estado_sesiones:
        estado_sesiones[session_id] = {
            "ultimo_frame_gray": None,
            "ultimas_detecciones_cache": None,
            "buffer_historico_frames": deque(maxlen=5)
        }

# Lista de objetos clasificados como de riesgo/precaución
OBJETOS_PELIGROSOS = {"knife", "scissors", "bottle", "syringe", "fire", "hazardous"}

def corregir_orientacion(img_cv2: np.ndarray) -> np.ndarray:
    """
    Auto-rotación (#14): Si la imagen se recibe en orientación vertical (h > w * 1.2),
    la rota 90° a la derecha para estandarizar la inferencia.
    """
    h, w = img_cv2.shape[:2]
    if h > w * 1.2:
        return cv2.rotate(img_cv2, cv2.ROTATE_90_CLOCKWISE)
    return img_cv2

def calcular_zona(bbox: dict) -> str:
    """Calcula el cuadrante o zona espacial basado en el centroide del bbox 640x640."""
    cx = bbox["x"] + bbox["w"] / 2
    cy = bbox["y"] + bbox["h"] / 2
    
    if 160 <= cx <= 480 and 160 <= cy <= 480:
        return "centro"
    elif cx < 320 and cy < 320:
        return "top_left"
    elif cx >= 320 and cy < 320:
        return "top_right"
    elif cx < 320 and cy >= 320:
        return "bottom_left"
    else:
        return "bottom_right"

def preprocesar_imagen(img_cv2: np.ndarray, solo_roi: bool = False) -> np.ndarray:
    """
    1. Aplica auto-rotación si el frame viene vertical (#14).
    2. Si solo_roi es True (#9), recorta el 70% central de la imagen.
    3. Redimensiona a 640x640 y aplica CLAHE en espacio LAB.
    """
    img_cv2 = corregir_orientacion(img_cv2)
    
    if solo_roi:
        h, w = img_cv2.shape[:2]
        crop_x1 = int(w * 0.15)
        crop_x2 = int(w * 0.85)
        crop_y1 = int(h * 0.15)
        crop_y2 = int(h * 0.85)
        img_cv2 = img_cv2[crop_y1:crop_y2, crop_x1:crop_x2]

    if img_cv2.shape[0] != 640 or img_cv2.shape[1] != 640:
        img_cv2 = cv2.resize(img_cv2, (640, 640), interpolation=cv2.INTER_LINEAR)
    
    lab = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_procesada = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return img_procesada

def obtener_detecciones_cache(img_cv2: np.ndarray, session_id: str):
    """
    Compara el frame actual en escala de grises con el frame anterior.
    Si la variación (MSE) es menor al umbral, retorna la caché.
    """
    init_estado_sesion(session_id)
    estado = estado_sesiones[session_id]
    gray = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
    
    if estado["ultimo_frame_gray"] is not None and estado["ultimas_detecciones_cache"] is not None:
        mse = float(np.mean((gray.astype("float") - estado["ultimo_frame_gray"].astype("float")) ** 2))
        if mse < MSE_UMBRAL_SIMILITUD:
            return estado["ultimas_detecciones_cache"], True
            
    estado["ultimo_frame_gray"] = gray
    return None, False

def aplicar_filtro_ensemble(detecciones_raw: list, session_id: str) -> list:
    """
    Ensemble de frames (#6): Exige que un objeto aparezca al menos en 3 de los últimos 5 frames
    para considerarlo consistente y eliminar falsos positivos parpadeantes.
    Mejora: usa id_track si está disponible para evitar ghosting entre múltiples objetos.
    """
    init_estado_sesion(session_id)
    estado = estado_sesiones[session_id]
    
    objetos_actuales = set()
    for d in detecciones_raw:
        # Usamos solo el nombre del objeto para el filtro ensemble,
        # ya que los IDs de tracking pueden ser inestables en cámaras móviles
        uid = d["objeto"]
        objetos_actuales.add(uid)
        
    estado["buffer_historico_frames"].append(objetos_actuales)
    
    if len(estado["buffer_historico_frames"]) < 3:
        return detecciones_raw  # Aún no hay suficiente historial
        
    conteo_presencia = Counter()
    for frame_set in estado["buffer_historico_frames"]:
        for uid in frame_set:
            conteo_presencia[uid] += 1
            
    # Solo mantener objetos presentes en al menos 3 de los 5 frames
    objetos_validos = set(uid for uid, count in conteo_presencia.items() if count >= 3)
    
    filtradas = []
    for d in detecciones_raw:
        uid = d["objeto"]
        if uid in objetos_validos:
            filtradas.append(d)
    return filtradas

def procesar_frame(img_raw: np.ndarray, usar_tracking: bool = True, session_id: str = "anonimo", usar_ensemble: bool = True, solo_roi: bool = False):
    """
    Pipeline optimizado de inferencia:
    - Preprocesamiento con Auto-rotación (#14) y ROI (#9).
    - Verificación de caché por MSE de frame.
    - Inferencia YOLO con iou=0.45, conf=CONFIANZA_MINIMA, imgsz=640.
    - Filtro Ensemble de 3 de 5 frames (#6).
    """
    init_estado_sesion(session_id)
    estado = estado_sesiones[session_id]
    
    img = preprocesar_imagen(img_raw, solo_roi=solo_roi)
    cached_dets, es_cache = obtener_detecciones_cache(img, session_id)
    if es_cache and cached_dets is not None:
        # BUG FIX: Guardar las detecciones cacheadas en la DB para no perder la continuidad
        for d in cached_dets:
            guardar_deteccion(d["objeto"], d["confianza"], session_id, d["zona"])
        return cached_dets, 5, True
        
    inicio = time.time()
    if usar_tracking:
        resultados = modelo.track(img, persist=True, conf=CONFIANZA_MINIMA, iou=0.45, imgsz=640, verbose=False)
    else:
        resultados = modelo(img, conf=CONFIANZA_MINIMA, iou=0.45, imgsz=640, verbose=False)
        
    tiempo_ms = round((time.time() - inicio) * 1000)
    
    detecciones_raw = []
    for resultado in resultados:
        for box in resultado.boxes:
            objeto = modelo.names[int(box.cls)]
            confianza = round(float(box.conf), 3)
            
            track_id = None
            if usar_tracking and box.id is not None:
                track_id = int(box.id[0])
                
            bbox = {
                "x": round(float(box.xyxy[0][0])),
                "y": round(float(box.xyxy[0][1])),
                "w": round(float(box.xyxy[0][2]) - float(box.xyxy[0][0])),
                "h": round(float(box.xyxy[0][3]) - float(box.xyxy[0][1]))
            }
            
            zona = calcular_zona(bbox)
            es_peligroso = objeto.lower() in OBJETOS_PELIGROSOS

            detecciones_raw.append({
                "id_track": track_id,
                "objeto": objeto,
                "confianza": confianza,
                "bbox": bbox,
                "zona": zona,
                "peligro": es_peligroso
            })

    # Aplicar filtro ensemble (3 de 5 frames) si se solicita
    if usar_ensemble:
        detecciones = aplicar_filtro_ensemble(detecciones_raw, session_id)
    else:
        detecciones = detecciones_raw

    for d in detecciones:
        guardar_deteccion(d["objeto"], d["confianza"], session_id, d["zona"])
            
    estado["ultimas_detecciones_cache"] = detecciones
    return detecciones, tiempo_ms, False



# ─── Base de datos ────────────────────────────────────────────────────────────

def init_db():
    """Crea la tabla de detecciones si no existe y asegura el esquema con session_id y zona."""
    os.makedirs("db", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS detecciones (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            objeto    TEXT    NOT NULL,
            confianza REAL    NOT NULL,
            timestamp TEXT    NOT NULL,
            session_id TEXT   DEFAULT 'anonimo',
            zona      TEXT    DEFAULT 'centro'
        )
    """)
    try:
        con.execute("ALTER TABLE detecciones ADD COLUMN session_id TEXT DEFAULT 'anonimo'")
    except sqlite3.OperationalError:
        pass
    try:
        con.execute("ALTER TABLE detecciones ADD COLUMN zona TEXT DEFAULT 'centro'")
    except sqlite3.OperationalError:
        pass
    con.commit()
    con.close()

# Ejecutar siempre al cargar el módulo para garantizar esquema
init_db()


def guardar_deteccion(objeto: str, confianza: float, session_id: str = "anonimo", zona: str = "centro"):
    """Inserta una detección en la base de datos asociada a una sesión y zona espacial."""
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO detecciones (objeto, confianza, timestamp, session_id, zona) VALUES (?, ?, ?, ?, ?)",
        (objeto, confianza, datetime.now().isoformat(), session_id, zona)
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


@app.route("/movil")
def movil():
    """Sirve la vista simplificada 2D para probar desde el celular."""
    return send_file("test_movil.html")


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
        img = base64_a_imagen(datos["imagen"])
        if img is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        usar_tracking = len(clientes_conectados) <= 1
        detecciones, tiempo_ms, es_cache = procesar_frame(img, usar_tracking=usar_tracking, session_id="anonimo")

        return jsonify({
            "detecciones": detecciones,
            "tiempo_ms": tiempo_ms,
            "total": len(detecciones),
            "cache": es_cache
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@socketio.on("connect")
def handle_connect():
    clientes_conectados.add(request.sid)
    print(f"✅ Nuevo cliente conectado: {request.sid}. Total: {len(clientes_conectados)}")

@socketio.on("disconnect")
def handle_disconnect():
    if request.sid in clientes_conectados:
        clientes_conectados.remove(request.sid)
    print(f"❌ Cliente desconectado: {request.sid}. Total: {len(clientes_conectados)}")

@socketio.on("detectar")
def handle_detectar(datos):
    """
    Recibe por WebSocket: { "imagen": "data:image/jpeg;base64,..." }
    Emite el evento "detecciones_resultado" con los objetos.
    """
    session_id = request.sid
    if not datos or "imagen" not in datos:
        emit("detecciones_resultado", {"error": "Falta el campo 'imagen'"})
        return

    try:
        img = base64_a_imagen(datos["imagen"])
        if img is None:
            emit("detecciones_resultado", {"error": "No se pudo decodificar la imagen"})
            return

        usar_tracking = len(clientes_conectados) <= 1
        detecciones, tiempo_ms, es_cache = procesar_frame(img, usar_tracking=usar_tracking, session_id=session_id)

        emit("detecciones_resultado", {
            "detecciones": detecciones,
            "tiempo_ms": tiempo_ms,
            "total": len(detecciones),
            "cache": es_cache
        })

    except Exception as e:
        emit("detecciones_resultado", {"error": str(e)})



@app.route("/estadisticas", methods=["GET"])
def estadisticas():
    """
    Devuelve estadísticas de la sesión actual incluyendo distribución espacial de mapa de calor.
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

    # Mapa de calor espacial por zonas
    zonas = cur.execute("""
        SELECT zona, COUNT(*) as conteo
        FROM detecciones
        GROUP BY zona
    """).fetchall()
    mapa_calor = {row[0]: row[1] for row in zonas}

    con.close()

    return jsonify({
        "total_detecciones": total,
        "objetos_unicos": unicos,
        "top_objetos": [
            {"objeto": row[0], "veces": row[1], "confianza_promedio": row[2]} 
            for row in top
        ],
        "timeline": [
            {"hora": row[0], "conteo": row[1]}
            for row in timeline
        ],
        "mapa_calor": mapa_calor,
        "usuarios_activos": len(clientes_conectados)
    })


@app.route("/sesiones", methods=["GET"])
def listar_sesiones():
    """Devuelve las distintas sesiones registradas en SQLite para comparación."""
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    filas = cur.execute("""
        SELECT session_id, COUNT(*) as total, MIN(timestamp) as inicio, MAX(timestamp) as fin
        FROM detecciones
        GROUP BY session_id
        ORDER BY inicio DESC
        LIMIT 10
    """).fetchall()
    con.close()
    
    sesiones = [
        {"session_id": r[0], "total_detecciones": r[1], "inicio": r[2], "fin": r[3]}
        for r in filas
    ]
    return jsonify({"sesiones": sesiones})


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
    filas = cur.execute("SELECT id, objeto, confianza, timestamp, session_id, zona FROM detecciones ORDER BY timestamp ASC").fetchall()
    con.close()

    def generar():
        yield "ID,Objeto,Confianza,Timestamp,SessionID,Zona\n"
        for fila in filas:
            zona_val = fila[5] if len(fila) > 5 else "centro"
            yield f"{fila[0]},{fila[1]},{fila[2]},{fila[3]},{fila[4]},{zona_val}\n"

    return app.response_class(
        generar(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=sesion_visionvr.csv'}
    )


# ─── Módulo IA (NVIDIA NIM) ──────────────────────────────────────────────────

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() # Cargar API key de .env
cliente_ia = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

@app.route("/explicar", methods=["POST"])
def explicar():
    """
    Recibe un objeto detectado y devuelve una explicación de Llama-3.
    """
    datos = request.get_json()
    objeto = datos.get("objeto", "objeto desconocido")
    contexto = datos.get("contexto", "entorno universitario")

    try:
        respuesta = cliente_ia.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{
                "role": "user",
                "content": (
                    f"Eres un asistente virtual integrado en unas gafas de realidad aumentada. "
                    f"El usuario está apuntando a un '{objeto}'. Explica muy brevemente (1 o 2 oraciones) "
                    f"qué es y cómo podría usarse en el contexto de '{contexto}'. Habla en español, de forma "
                    f"directa, clara e inteligente. No uses comillas."
                )
            }],
            max_tokens=150
        )
        texto = respuesta.choices[0].message.content.strip()
        return jsonify({"explicacion": texto})
    except Exception as e:
        print("Error en NVIDIA API:", e)
        return jsonify({"explicacion": "Lo siento, hubo un error de conexión con la IA de NVIDIA."}), 500


@app.route("/analizar_escena", methods=["POST"])
def analizar_escena():
    """
    Recibe la lista de objetos en pantalla y genera una descripción analítica global con Llama-3.
    """
    datos = request.get_json() or {}
    objetos = datos.get("objetos", [])

    if not objetos:
        return jsonify({"resumen": "Escena despejada. No se detectan objetos significativos en este momento."})

    lista_str = ", ".join(list(set(objetos)))
    try:
        respuesta = cliente_ia.chat.completions.create(
            model="meta/llama-3.1-70b-instruct",
            messages=[{
                "role": "user",
                "content": (
                    f"Eres una IA de visión por computador integrada en Meta Quest 3. "
                    f"Los objetos detectados en la escena actual son: [{lista_str}]. "
                    f"Proporciona un resumen sintético de 2 oraciones sobre el estado y composición de la escena en español. "
                    f"Si hay algún objeto de precaución (cuchillo, tijeras, etc.), adviértelo explícitamente."
                )
            }],
            max_tokens=150
        )
        resumen = respuesta.choices[0].message.content.strip()
        return jsonify({"resumen": resumen})
    except Exception as e:
        print("Error en analizar_escena NVIDIA:", e)
        return jsonify({"resumen": f"Objetos visibles en escena: {lista_str}."})


@app.route("/modelo_info", methods=["GET"])
def modelo_info():
    """
    Devuelve información sobre el modelo cargado (#12):
    si es un modelo custom de Roboflow o el yolov8m por defecto, y la lista de clases.
    """
    es_custom = os.path.exists(CUSTOM_MODEL_PATH)
    nombres_clases = list(modelo.names.values()) if hasattr(modelo, "names") else []
    return jsonify({
        "modelo_path": MODELO_PATH,
        "es_custom": es_custom,
        "total_clases": len(nombres_clases),
        "clases_muestra": nombres_clases[:15]
    })


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
