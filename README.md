# 🥽 VisionVR — Detección de Objetos en Realidad Aumentada

Sistema de **Inteligencia Artificial** que detecta objetos en tiempo real usando el **Meta Quest 3** con passthrough AR, desarrollado como proyecto universitario para la materia de IA en la UFT.

## 🎯 ¿Qué hace?

Un usuario con el Meta Quest 3 puede ver objetos del mundo real con etiquetas flotantes superpuestas en 3D, mostrando qué son y con qué confianza fueron detectados. También puede:

- **Preguntar a la IA** sobre cualquier objeto señalándolo con la mano (Groq Llama-3)
- **Ver estadísticas en tiempo real** en un dashboard web
- **Probar desde el celular** sin necesidad de Quest
- **Exportar sesiones** como CSV para análisis

## 🏗️ Arquitectura

```
┌──────────────────────┐     WebSocket      ┌──────────────────────┐
│   Meta Quest 3       │ ◄────────────────► │   Backend Flask      │
│   (A-Frame WebXR)    │   frames base64    │   + YOLOv8           │
│                      │   detecciones JSON │   + SQLite            │
│   - Passthrough AR   │                    │   + Groq (Llama-3)   │
│   - Hand Tracking    │                    │                      │
│   - Depth Sensing    │                    │   Puerto: 5000       │
└──────────────────────┘                    └──────────────────────┘
```

## 📂 Estructura del Proyecto

| Carpeta | Descripción |
|---|---|
| [`visionvr/`](visionvr/) | **Proyecto principal** — Backend + Frontend integrado |
| [`visionvr-connect/`](visionvr-connect/) | App complementaria (React + TanStack Start) |

## 🚀 Inicio Rápido

```bash
cd visionvr
py start.py
```

El script automáticamente verifica Python, instala dependencias, arranca el servidor, y abre el navegador. Para más detalles, consulta el [README de visionvr](visionvr/README.md).

## ⚙️ Stack Tecnológico

| Componente | Tecnología |
|---|---|
| **Detección de objetos** | YOLOv8 (Ultralytics) con ONNX Runtime |
| **Backend** | Flask + Flask-SocketIO |
| **Frontend AR** | A-Frame (WebXR) |
| **IA Generativa** | Groq API (Llama-3.1-70B) |
| **Base de datos** | SQLite |
| **Dashboard** | Chart.js |
| **Comunicación** | WebSocket (Socket.IO) |

## 👨‍💻 Autores

- **Andrés** — Backend (servidor, detección YOLO, base de datos, IA)
- **María** — Frontend (interfaz WebXR, etiquetas AR, voz)

## 📄 Licencia

Proyecto universitario — UFT, Inteligencia Artificial.