"""
test_servidor.py
================
Prueba el servidor sin necesitar el Quest ni el frontend.
Manda una imagen real a /detectar y muestra los resultados.

Uso:
    python test_servidor.py
    python test_servidor.py ruta/a/foto.jpg
"""

import sys
import base64
import json
import urllib.request

URL = "http://localhost:5000"

def imagen_a_base64(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def post_json(endpoint: str, datos: dict) -> dict:
    body = json.dumps(datos).encode()
    req = urllib.request.Request(
        URL + endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())

def get_json(endpoint: str) -> dict:
    with urllib.request.urlopen(URL + endpoint) as resp:
        return json.loads(resp.read())

if __name__ == "__main__":
    print("\n-- Prueba 1: ping -------------------------")
    try:
        res = get_json("/ping")
        print("[OK]", res)
    except Exception as e:
        print("[X] Servidor no responde:", e)
        print("  Corriste 'python server.py' primero?")
        sys.exit(1)

    print("\n-- Prueba 2: deteccion ---------------------")
    ruta = sys.argv[1] if len(sys.argv) > 1 else None

    if ruta:
        print(f"Usando imagen: {ruta}")
        imagen = imagen_a_base64(ruta)
    else:
        # Genera un frame de prueba (imagen gris sintetica) si no das una foto
        print("Sin imagen proporcionada - usando frame sintetico gris")
        import numpy as np
        import cv2
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        _, buf = cv2.imencode(".jpg", frame)
        imagen = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    res = post_json("/detectar", {"imagen": imagen})
    print(f"[OK] Tiempo: {res.get('tiempo_ms')}ms")
    print(f"[OK] Detecciones: {res.get('total')}")
    for d in res.get("detecciones", []):
        print(f"   > {d['objeto']} ({d['confianza']*100:.1f}%)")

    print("\n-- Prueba 3: estadisticas ------------------")
    res = get_json("/estadisticas")
    print(f"[OK] Total detecciones en sesion: {res['total_detecciones']}")
    print(f"[OK] Objetos unicos: {res['objetos_unicos']}")
    for obj in res.get("top_objetos", []):
        print(f"   > {obj['objeto']}: {obj['veces']} veces ({obj['confianza_promedio']*100:.0f}% conf. prom.)")

    print("\n-- Todo OK ---------------------------------\n")
