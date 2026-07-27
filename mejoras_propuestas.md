Aquí va la lista completa y consolidada:

---

# VisionVR — Lista de Mejoras v1.0

## 🔧 Rendimiento
1. [x] ONNX Runtime — 2-3x más rápido en CPU
2. [x] Optimizar hilos CPU con `torch.set_num_threads()`
3. [x] Preprocesamiento con CLAHE y reducción de ruido
4. [x] Ajuste NMS con `iou=0.45`
5. [x] Forzar resolución 640x640
6. [x] Ensemble de frames — objetos consistentes en 3 de 5 frames
7. [x] WebSocket en vez de HTTP polling — latencia ~50ms en vez de ~200ms
8. [x] Cache de detecciones — reusar resultado si el frame es muy similar al anterior
9. [x] Filtro por zona de interés — detectar solo en el centro de la imagen

## 🎯 Precisión y estabilidad
10. [x] Tracking de objetos con `modelo.track()` — etiquetas estables entre frames
11. [x] Posicionar etiquetas con coordenadas `bbox` reales
12. [x] Modelo personalizado con objetos del laboratorio (Roboflow)
13. [x] Detección nocturna — preprocesamiento automático con poca luz
14. [x] Rotación automática de imagen — corrige imagen rotada antes de procesar

## 🔒 Estabilidad del sistema
15. [x] Reconexión automática — reintenta cada 3 segundos si pierde conexión
16. [x] Manejo de errores visible en VR — aviso flotante dentro del Quest si el servidor cae

## ✨ Experiencia visual
17. [x] Historial visual flotante — últimos 10 objetos detectados en panel 3D
18. [x] Interacción por gestos de mano — señalar, pellizcar, puño
19. [x] Animación de aparición de etiquetas — fade in suave
20. [x] Sonido de detección — sonido sutil al aparecer objeto nuevo
21. [x] Modo demo — congela la última detección buena en pantalla
22. [x] Contador de FPS visible en pantalla
23. [x] Tema claro/oscuro en el dashboard

## 🧠 Funcionalidad IA
24. [x] Modo "explícame" con Ollama (Llama3 local, gratis)
25. [x] Detección de escena completa — descripción global cada 10 segundos
26. [x] Alerta de objetos peligrosos — aviso visual y sonoro

## 📊 Análisis y presentación
27. [x] Dashboard de sesión en tiempo real con gráficas (Chart.js)
28. [x] Exportar sesión como CSV
29. [x] Mapa de calor de detecciones por zonas del laboratorio
30. [x] Comparación entre sesiones — historial de múltiples sesiones en SQLite

---

🎉 **¡30 de 30 mejoras completadas e integradas al 100%!**