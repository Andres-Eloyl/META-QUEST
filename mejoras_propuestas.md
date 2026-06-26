# VisionVR — Propuestas de Mejora

## 🟢 Victorias rápidas (1-2 horas cada una)

### 1. WebSocket en vez de HTTP polling
**Impacto: alto** — Ahora mismo el frontend hace `POST /detectar` cada 500ms y espera respuesta. Con WebSocket la comunicación sería bidireccional y en tiempo real, reduciendo latencia a la mitad.

- Modifica: `server.py` (agregar Flask-SocketIO) + `index.html` (usar socket.io client)
- Beneficio: latencia ~50ms en vez de ~200ms por frame

---

### 2. Posicionamiento de etiquetas usando bbox
**Impacto: alto** — El backend YA devuelve coordenadas `bbox` pero el frontend las ignora. Se pueden usar para posicionar las etiquetas 3D aproximadamente donde está el objeto real, en vez de distribuirlas en un arco genérico.

- Modifica: `index.html` (función `crearEtiqueta3D`)
- Usa `bbox.x, bbox.y` mapeados al espacio 3D de la escena

---

### 3. Modo "explícame" con GPT
**Impacto: alto** — Ya está 90% listo. El backend tiene el endpoint `/explicar` comentado. Solo falta descomentarlo y agregar un botón en el frontend.

- Modifica: `server.py` (descomentar bloque GPT) + `index.html` (agregar botón + leer respuesta en voz)
- Requiere: API key de OpenAI

---

### 4. Dashboard de sesión en tiempo real
**Impacto: medio** — Una página `dashboard.html` separada que muestre gráficas en vivo: objetos detectados por minuto, distribución de confianza, timeline de la sesión. Útil para la presentación de la materia.

- Archivo nuevo: `dashboard.html` (con Chart.js)
- Consume: `GET /estadisticas` del backend existente

---

### 5. Exportar sesión como PDF/CSV
**Impacto: medio** — Botón en el dashboard que descargue un reporte con todas las detecciones, tiempos y estadísticas. Perfecto para entregar como evidencia del proyecto.

- Modifica: `server.py` (nuevo endpoint `GET /exportar`)
- Genera CSV o usa una librería ligera de PDF

---

## 🟡 Mejoras intermedias (3-6 horas)

### 6. Tracking de objetos entre frames
**Impacto: alto** — Ahora cada frame es independiente: detecta objetos desde cero. Con tracking (SORT o ByteTrack, integrados en ultralytics), los objetos mantienen identidad entre frames. La etiqueta de "laptop" no parpadea, se mueve suavemente.

- Modifica: `server.py` (usar `modelo.track()` en vez de `modelo()`)
- Beneficio: etiquetas estables, conteo real de objetos únicos

---

### 7. Interacción por gestos de mano
**Impacto: alto** — El Quest 3 tiene hand tracking nativo. Se puede agregar:
  - Señalar un objeto → ver su etiqueta ampliada
  - Pellizcar → pedir explicación GPT
  - Puño → pausar detección

- Modifica: `index.html` (componentes A-Frame de hand-tracking)
- Ya tenemos `optionalFeatures: hand-tracking` declarado en la escena

---

### 8. Historial visual flotante
**Impacto: medio** — Panel 3D flotante que muestra los últimos 10 objetos detectados con thumbnails. Ya mencionado en las instrucciones de María como mejora futura.

- Modifica: `index.html` (nuevo panel A-Frame)

---

### 9. Modelo YOLO personalizado
**Impacto: alto** — Entrenar YOLOv8 para detectar objetos específicos del contexto universitario: Arduino, Raspberry Pi, protoboard, multímetro, osciloscopio. El frontend ya los muestra automáticamente porque solo lee el campo `objeto` del JSON.

- Herramienta: [Roboflow](https://roboflow.com) para anotar imágenes + `ultralytics` para entrenar
- Modifica: `server.py` (cambiar `MODELO_PATH` al modelo custom)

---

### 10. Modo offline / cache de modelo
**Impacto: medio** — Ahora el backend requiere internet la primera vez para descargar YOLO. Se puede incluir el `.pt` en el repo (ya lo tienen: `yolov8n.pt` son solo 6MB) y agregar un Service Worker al frontend para que funcione sin internet después del primer acceso.

---

## 🔴 Mejoras avanzadas (1-2 días)

### 11. Multi-usuario
Varios Quest conectados al mismo backend, cada uno con su sesión independiente. El backend ya usa SQLite pero sin concepto de sesión por usuario.

### 12. Detección de profundidad
Usar la API de profundidad del Quest 3 (`depth-sensing`) para posicionar las etiquetas en la posición 3D exacta del objeto, no solo proyectadas en un plano.

### 13. Deploy en la nube
Subir el backend a un servidor (Railway, Render, o una VM con GPU) para que funcione sin necesitar la PC de Andrés encendida. Requiere GPU en la nube para YOLO.

---

## Mi recomendación: top 3 para empezar

| Prioridad | Mejora | Por qué |
|-----------|--------|---------|
| **1** | Tracking de objetos (#6) | Un cambio de 1 línea en el backend (`modelo.track()`) que mejora drásticamente la experiencia visual |
| **2** | Posicionar etiquetas con bbox (#2) | Los datos ya existen, solo falta usarlos en el frontend |
| **3** | Dashboard (#4) | Impresiona mucho en una presentación y es independiente del resto |

> [!TIP]
> Si quieres que implemente alguna de estas mejoras, dime cuáles te interesan y las hago.
