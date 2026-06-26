# VisionVR — Detección de Objetos en Realidad Aumentada

Proyecto de Inteligencia Artificial que detecta objetos en tiempo real usando el Meta Quest 3 con passthrough AR.

**Backend:** Flask + YOLOv8 (detección de objetos)
**Frontend:** A-Frame WebXR (visualización AR en el Quest)

---

## Arranque rápido

```bash
# 1. Clona el repositorio
git clone <url-del-repo>
cd visionvr

# 2. Arranca todo
python start.py
```

Eso es todo. El script:
- ✓ Verifica Python 3.10+
- ✓ Instala dependencias automáticamente
- ✓ Arranca el backend (puerto 5000)
- ✓ Arranca el frontend (puerto 8080)
- ✓ Muestra la IP para el Quest
- ✓ Abre el navegador

---

## Conectar el Meta Quest 3

1. Conecta el Quest al **mismo WiFi** que la PC
2. Abre el **Meta Horizon Browser** en el Quest
3. Escribe la IP que mostró `start.py` (formato `http://192.168.x.x:8080`)
4. En el campo "IP del servidor" escribe `http://192.168.x.x:5000`
5. Presiona **Conectar**
6. Escucharás: *"Sistema VisionVR activado"*
7. Presiona **Enter VR** para ver las detecciones en passthrough AR

---

## Estructura del proyecto

```
visionvr/
├── start.py           ← un comando arranca todo
├── server.py          ← backend Flask + YOLO
├── index.html         ← frontend WebXR (A-Frame)
├── requirements.txt   ← dependencias Python
├── test_servidor.py   ← pruebas del backend
├── db/                ← base de datos SQLite (se crea sola)
└── README.md
```

---

## Endpoints del backend

| Endpoint | Método | Descripción |
|---|---|---|
| `/ping` | GET | Verifica que el servidor está activo |
| `/detectar` | POST | Recibe frame base64, devuelve objetos detectados |
| `/estadisticas` | GET | Estadísticas de la sesión |
| `/limpiar` | POST | Borra historial de la sesión |

---

## Probar sin el Quest

Abre `http://localhost:8080` en Chrome, escribe `http://localhost:5000` como IP del servidor y presiona Conectar. No verás el passthrough pero sí las etiquetas de detección y la voz.

---

## Probar solo el backend

```bash
python test_servidor.py                  # con imagen sintética
python test_servidor.py ruta/a/foto.jpg  # con foto real
```

---

## Requisitos

- Python 3.10+
- Conexión a internet (primera vez, para descargar modelo YOLO ~6MB)
- Meta Quest 3 en el mismo WiFi (para pruebas AR)

---

## Problema más común

Si el Quest no carga la página → PC y Quest están en **redes diferentes**. Usa el hotspot de un celular como router compartido si el WiFi institucional bloquea comunicación entre dispositivos.

---

## Autores

- **Andrés** — Backend (servidor, detección YOLO, base de datos)
- **María** — Frontend (interfaz WebXR, etiquetas AR, voz)
