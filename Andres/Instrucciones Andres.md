# VisionVR — Instrucciones Backend (Andrés)

## Archivos que necesitas

- `server.py`
- `test_servidor.py`
- `requirements.txt`
- `README.md`

Mételos todos en una carpeta llamada `visionvr`.

## Requisitos previos

Necesitas tener Python 3.10 o superior instalado. Para verificarlo abre una terminal y escribe:

```bash
python --version
```

Si no lo tienes, descárgalo desde [python.org](https://www.python.org/).

## Instalación

Abre una terminal dentro de la carpeta `visionvr` y corre:

```bash
pip install -r requirements.txt
```

Esto instala Flask, YOLO, OpenCV y todo lo necesario. Tarda unos minutos la primera vez.

## Arrancar el servidor

```bash
python server.py
```

La primera vez descarga el modelo YOLO automáticamente (unos 6MB, necesitas internet). Cuando esté listo verás esto en consola:

```
==================================================
  VisionVR Backend corriendo
  http://localhost:5000/ping
  IP en red local: http://192.168.x.x:5000
==================================================
```

Esa **IP en red local** es la que le tienes que pasar a María para que el frontend del Quest se conecte a tu PC.

## Verificar que todo funciona

Sin cerrar la terminal del servidor, abre una **segunda terminal** en la misma carpeta y corre:

```bash
python test_servidor.py
```

Deberías ver tres pruebas con ✓. Si las tres pasan, el backend está funcionando correctamente.

También puedes probar con una foto real:

```bash
python test_servidor.py ruta/a/tu/foto.jpg
```

## Endpoints disponibles

Una vez corriendo, el servidor expone estos endpoints:

| Endpoint | Método | Descripción |
|---|---|---|
| `/ping` | GET | Verifica que el servidor está activo |
| `/detectar` | POST | Recibe un frame en base64 y devuelve los objetos detectados |
| `/estadisticas` | GET | Devuelve estadísticas de la sesión actual |
| `/limpiar` | POST | Borra el historial de la sesión |
| `/explicar` | POST | Explicación con GPT (se activa más adelante) |

## Formato del contrato con María

Este es el JSON que el servidor recibe en `/detectar`:

```json
{ "imagen": "data:image/jpeg;base64,/9j/4AAQ..." }
```

Y este es el JSON que devuelve:

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

María solo necesita saber ese formato para construir el frontend independientemente.

## Módulos futuros (no tocar todavía)

El servidor tiene comentado el módulo de GPT. Cuando lleguen a esa capa del proyecto:

1. En `server.py` descomenta el bloque marcado como **Módulo GPT**
2. En `requirements.txt` descomenta la línea de `openai`
3. Reinstala dependencias: `pip install -r requirements.txt`
4. Configura tu API key: `set OPENAI_API_KEY=tu_clave` (en Windows)

Por ahora no es necesario.

## Problema más común

Si el Quest no se conecta al servidor, casi siempre es porque están en **redes diferentes**. Asegúrate de que tu PC y el Quest estén en el mismo WiFi. Si el WiFi de la universidad bloquea comunicación entre dispositivos, usa el hotspot de un celular como router compartido para las pruebas.