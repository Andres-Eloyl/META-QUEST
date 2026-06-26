VisionVR — Instrucciones Frontend (María)
Archivos que necesitas

index.html

Solo ese archivo. No necesitas instalar Node.js, npm ni nada por el estilo. Es HTML puro.

Requisitos previos
Necesitas tener acceso a un servidor web local para servir el archivo. La forma más simple es con Python, que probablemente ya tienes instalado. Verifica con:
bashpython --version

Cómo servir el archivo
El archivo index.html no puede abrirse directo desde el explorador de archivos porque WebXR requiere un servidor. Usa esto:
bash# Entra a la carpeta donde está el index.html
cd visionvr

# Levanta un servidor local en el puerto 8080
python -m http.server 8080
Verás algo así:
Serving HTTP on 0.0.0.0 port 8080 ...
Deja esa terminal abierta mientras trabajas.

Cómo acceder desde el Meta Quest 3
Una vez que el servidor de Andrés esté corriendo y el tuyo también, sigue estos pasos:
1. Conecta el Quest 3 al mismo WiFi que la PC de Andrés.
2. Abre el Meta Horizon Browser dentro del Quest.
3. En la barra de direcciones escribe la IP de tu PC (no la de Andrés) seguida del puerto 8080. Ejemplo:
http://192.168.1.102:8080
Para saber tu IP abre una terminal y escribe:
bash# En Windows
ipconfig

# En Mac/Linux
ifconfig
Busca la dirección que empieza con 192.168.x.x.
4. El navegador del Quest cargará la página con un panel de configuración en la parte superior.

Conectar con el servidor de Andrés
Cuando cargue la página verás un campo que dice IP del servidor. Ahí escribe la IP que Andrés te pase, que tiene este formato:
http://192.168.1.105:5000
Luego presiona el botón Conectar. Si todo está bien, el estado cambiará a:
✓ Conectado — iniciando detección
Y escucharás por el audio del Quest: "Sistema VisionVR activado".

Lo que hace el frontend
Una vez conectado y dentro del modo VR:

Captura un frame del canvas cada 500 milisegundos
Lo manda al servidor de Andrés en formato base64
Recibe de vuelta los objetos detectados con su porcentaje de confianza
Muestra etiquetas flotantes 3D sobre el entorno real
Las etiquetas cambian de color según la confianza: verde alta, amarillo media, rojo baja
Anuncia en voz alta el primer objeto detectado de cada frame
Muestra un panel de estadísticas de sesión en la esquina superior derecha


Estructura del código
El index.html tiene tres secciones principales:
Panel de configuración — barra superior donde escribes la IP del servidor. Desaparece automáticamente cuando entras al modo VR.
Escena A-Frame — el entorno WebXR con passthrough activado. Aquí viven las etiquetas flotantes y el panel de estadísticas.
Script JavaScript — maneja la captura de frames, comunicación con el servidor, generación de etiquetas y síntesis de voz. Está dividido en funciones claras y comentadas.

Formato del JSON que recibes de Andrés
Cada vez que mandas un frame, el servidor responde con esto:
json{
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
Usas objeto y confianza para las etiquetas. El campo bbox está disponible para mejoras futuras si quieren posicionar las etiquetas más precisamente.

Prueba sin el Quest
Puedes probar el frontend en Chrome de escritorio antes de llevarlo al Quest. Abre http://localhost:8080 en Chrome, escribe la IP del servidor de Andrés y presiona Conectar. No verás el passthrough pero sí podrás verificar que las etiquetas aparecen y que la voz funciona.

Módulos que se agregarán después
El frontend está preparado para recibir estas mejoras en capas sin reescribir nada:
Modo "explícame" — cuando Andrés active el endpoint /explicar, se agrega un botón en la escena VR que el usuario activa con la mirada. El servidor devuelve una explicación de GPT que se lee en voz alta.
Historial visual — panel flotante adicional con los últimos 10 objetos detectados.
Clase personalizada — cuando Andrés entrene el modelo con objetos nuevos como Arduino o Raspberry Pi, el frontend los muestra automáticamente sin cambios porque solo lee el campo objeto del JSON.

Problema más común
Si el Quest no carga la página, casi siempre es porque la PC y el Quest están en redes diferentes. Asegúrate de que ambos estén en el mismo WiFi. Si el WiFi de la universidad bloquea comunicación entre dispositivos, usa el hotspot de un celular como router compartido para las pruebas.
Si la voz no funciona en el Quest, es normal que el primer intento falle porque el navegador requiere una interacción del usuario primero. Presiona el botón Conectar manualmente dentro del Quest y después de eso la voz debería funcionar.