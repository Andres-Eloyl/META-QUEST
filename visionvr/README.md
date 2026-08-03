# VisionVR — Detección de Objetos en Realidad Aumentada

Proyecto de Inteligencia Artificial que detecta objetos en tiempo real usando el **Meta Quest 3** con passthrough AR.

**Backend:** Flask + YOLOv8 + SQLite + Groq (Llama-3)  
**Frontend:** A-Frame WebXR + Socket.IO  
**Universidad:** UFT — Materia: Inteligencia Artificial

---

## 🚀 Arranque Rápido

### Opción 1: Un solo comando (recomendado)

```bash
cd visionvr
py start.py
```

El script automáticamente:
- ✅ Verifica Python 3.10+
- ✅ Instala dependencias si faltan
- ✅ Arranca el servidor Flask (backend + frontend) en el puerto 5000
- ✅ Muestra la IP local para conectar el Quest
- ✅ Abre el navegador

### Opción 2: Arranque manual (paso a paso)

```bash
# 1. Instalar dependencias
cd visionvr
pip install -r requirements.txt

# 2. Arrancar el servidor
py server.py
```

El servidor estará en: `https://localhost:5000`

### Opción 3: Solo probar el backend (sin Quest)

```bash
# Primero arranca el servidor en una terminal:
py server.py

# En otra terminal, corre las pruebas:
py test_servidor.py                  # con imagen sintética
py test_servidor.py ruta/a/foto.jpg  # con foto real
```

---

## 🥽 Conectar el Meta Quest 3

1. Conecta el Quest al **mismo WiFi** que tu PC
2. Arranca el servidor con `py start.py` (te mostrará la IP local)
3. En el Quest, abre el **Meta Horizon Browser**
4. Escribe: `https://<IP-de-tu-PC>:5000`
5. **Acepta el certificado** auto-firmado (es normal, el servidor usa HTTPS local)
6. Escribe la IP del servidor en el campo y presiona **CONNECT**
7. Escucharás: *"Sistema VisionVR activado"*
8. Presiona **Enter VR** para ver las detecciones flotando en passthrough AR

---

## 📱 Probar desde el Celular

1. Asegúrate de estar en la misma red WiFi que la PC
2. Abre en el navegador del celular: `https://<IP-de-tu-PC>:5000/movil`
3. Acepta el certificado y da permisos de cámara
4. Toca los recuadros verdes de detección para consultar a la IA

---

## 📊 Dashboard de Estadísticas

Abre `https://localhost:5000/dashboard` para ver en tiempo real:
- Total de detecciones y objetos únicos
- Gráfica de actividad por segundo
- Mapa de calor por zonas espaciales
- Top 5 objetos más detectados
- Exportar sesión como CSV

---

## 📁 Estructura del Proyecto

```
visionvr/
├── start.py              ← Un comando arranca todo
├── server.py             ← Backend Flask + YOLO + SQLite + Groq
├── index.html            ← Frontend WebXR (A-Frame) — interfaz principal
├── dashboard.html        ← Panel de estadísticas con Chart.js
├── test_movil.html       ← Vista simplificada 2D para celular
├── sw.js                 ← Service Worker para modo offline
├── requirements.txt      ← Dependencias Python
├── .env                  ← API keys (Groq) — NO subir al repo
├── test_servidor.py      ← Tests del backend sin necesitar el Quest
├── train_custom.py       ← Entrenamiento personalizado con Roboflow
├── train_custom_poc.py   ← Entrenamiento rápido de prueba (5 epochs)
├── js/
│   ├── aframe.min.js     ← A-Frame local (modo offline)
│   └── socket.io.min.js  ← Socket.IO local (modo offline)
└── db/
    └── sesion.db          ← SQLite (se crea automáticamente)
```

---

## 🔌 Endpoints del Backend

| Endpoint | Método | Descripción |
|---|---|---|
| `/` | GET | Interfaz principal WebXR |
| `/ping` | GET | Verifica que el servidor está activo |
| `/detectar` | POST | Recibe frame base64, devuelve objetos detectados |
| `/explicar` | POST | IA explica un objeto (Groq / Llama-3) |
| `/analizar_escena` | POST | Análisis global de la escena completa |
| `/estadisticas` | GET | Estadísticas de la sesión actual |
| `/sesiones` | GET | Lista de todas las sesiones registradas |
| `/modelo_info` | GET | Info del modelo YOLO cargado |
| `/exportar` | GET | Descarga CSV con el historial |
| `/limpiar` | POST | Borra historial de detecciones |
| `/dashboard` | GET | Panel de estadísticas |
| `/movil` | GET | Vista simplificada para celular |

**WebSocket** (Socket.IO):
| Evento | Dirección | Descripción |
|---|---|---|
| `detectar` | Cliente → Servidor | Envía frame para detección |
| `detecciones_resultado` | Servidor → Cliente | Devuelve objetos detectados |

---

## ⚙️ Requisitos

- **Python 3.10+** 
- **Meta Quest 3** en el mismo WiFi (para pruebas AR)
- **Groq API Key** en el archivo `.env` (para el modo "explícame" con Llama-3)
- Primera ejecución necesita internet (descarga modelo YOLO ~52MB)

---

## 🔧 Configuración del .env

Copia el archivo de ejemplo y rellena con tu clave real:

```bash
cp .env.example .env
```

Edita el `.env` con:

```
GROQ_API_KEY=tu-clave-groq-aqui
```

Obtén tu clave gratis en: https://console.groq.com/

---

## ❓ Problemas Comunes

| Problema | Solución |
|---|---|
| Quest no carga la página | PC y Quest están en **redes WiFi diferentes**. Usa hotspot del celular. |
| Certificado rechazado | Es un certificado auto-firmado. En el Quest, presiona "Continuar de todas formas". |
| No detecta objetos | Verifica que hay buena iluminación. El modelo necesita luz para funcionar bien. |
| `py` no se encuentra | Usa `python` o `python3` en vez de `py`. |
| Error de Groq API | Verifica que tu `.env` tiene la clave GROQ_API_KEY correcta y que tienes internet. |

---

## 👨‍💻 Autores

- **Andrés** — Backend (servidor, detección YOLO, base de datos, IA)
- **María** — Frontend (interfaz WebXR, etiquetas AR, voz)
