# VisionVR — Backend

Sistema de detección de objetos en tiempo real para Meta Quest 3.

## Instalación

```bash
# 1. Clona o descarga el proyecto
cd visionvr

# 2. Instala dependencias (Python 3.10+ recomendado)
pip install -r requirements.txt

# 3. El modelo YOLO se descarga automáticamente al primer arranque
```

## Uso

```bash
# Arrancar el servidor
python server.py

# En otra terminal, probar que funciona
python test_servidor.py

# Probar con una foto real
python test_servidor.py foto.jpg
```

## Endpoints

| Endpoint | Método | Descripción |
|---|---|---|
| `/ping` | GET | Verifica que el servidor está activo |
| `/detectar` | POST | Recibe imagen, devuelve detecciones |
| `/estadisticas` | GET | Estadísticas de la sesión |
| `/limpiar` | POST | Borra historial de sesión |
| `/explicar` | POST | Explicación GPT (activar en server.py) |

## Formato del JSON que recibe /detectar

```json
{ "imagen": "data:image/jpeg;base64,/9j/4AAQ..." }
```

## Formato del JSON que devuelve /detectar

```json
{
  "detecciones": [
    {
      "objeto": "laptop",
      "confianza": 0.97,
      "bbox": { "x": 120, "y": 80, "w": 200, "h": 150 }
    }
  ],
  "tiempo_ms": 45,
  "total": 1
}
```

## Módulos por activar

Para activar el módulo GPT en `server.py`:
1. Descomenta el bloque `# ─── Módulo GPT`
2. Descomenta `openai` en requirements.txt y reinstala
3. Crea variable de entorno: `export OPENAI_API_KEY=tu_clave`

## Estructura del proyecto

```
visionvr/
├── server.py          ← servidor principal (tú)
├── test_servidor.py   ← pruebas sin Quest (tú)
├── requirements.txt   ← dependencias
├── db/
│   └── sesion.db      ← se crea automáticamente
└── README.md
```

## Para María (frontend)

La IP del servidor se muestra al arrancar. Ejemplo:
```
IP en red local: http://192.168.1.105:5000
```

Usar esa IP en el frontend para las peticiones fetch().
