"""
start.py — Script de arranque único para VisionVR
===================================================
Un solo comando arranca todo el proyecto:
    python start.py

Hace lo siguiente:
  1. Verifica que Python 3.10+ esté instalado
  2. Verifica/instala las dependencias de requirements.txt
  3. Arranca el backend Flask (Andrés) en el puerto 5000
  4. Arranca el servidor del frontend (María) en el puerto 8080
  5. Muestra la IP local para conectar el Quest
  6. Abre el navegador automáticamente
"""

import sys
import os
import subprocess
import socket
import time
import signal
import webbrowser
import threading

# ─── Colores para la consola ──────────────────────────────────────────────────

class Color:
    VERDE   = "\033[92m"
    AMARILLO = "\033[93m"
    ROJO    = "\033[91m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RESET   = "\033[0m"

def info(msg):
    print(f"  {Color.CYAN}>{Color.RESET} {msg}")

def ok(msg):
    print(f"  {Color.VERDE}[OK]{Color.RESET} {msg}")

def warn(msg):
    print(f"  {Color.AMARILLO}[!]{Color.RESET} {msg}")

def error(msg):
    print(f"  {Color.ROJO}[X]{Color.RESET} {msg}")

# ─── Verificaciones ──────────────────────────────────────────────────────────

def verificar_python():
    """Verifica que Python sea 3.10 o superior."""
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        error(f"Se requiere Python 3.10+, tienes {version_str}")
        sys.exit(1)
    ok(f"Python {version_str}")

def verificar_dependencias():
    """Verifica que las dependencias estén instaladas. Si no, las instala."""
    req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if not os.path.exists(req_path):
        error("No se encontro requirements.txt")
        sys.exit(1)

    # Leer dependencias (ignorar comentarios y lineas vacias)
    with open(req_path, "r") as f:
        deps = [
            line.strip().split(">=")[0].split("==")[0].lower()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    # Obtener paquetes instalados via pip (rapido, no importa modulos)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=columns"],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=15
        )
        instalados = set()
        for line in result.stdout.splitlines()[2:]:  # Saltar encabezados
            parts = line.split()
            if parts:
                instalados.add(parts[0].lower())
    except Exception:
        instalados = set()

    # Verificar cuales faltan
    faltantes = [dep for dep in deps if dep.lower() not in instalados]

    if faltantes:
        warn(f"Faltan dependencias: {', '.join(faltantes)}")
        info("Instalando dependencias (esto puede tardar)...")
        resultado = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", req_path],
            encoding='utf-8', errors='replace'
        )
        if resultado.returncode != 0:
            error("Error instalando dependencias")
            sys.exit(1)
        ok("Dependencias instaladas")
    else:
        ok(f"Dependencias OK ({len(deps)} paquetes)")

def obtener_ip_local():
    """Obtiene la IP local de la máquina en la red WiFi."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())

# ─── Arranque de servidores ──────────────────────────────────────────────────

procesos = []

def arrancar_backend():
    """Arranca el servidor Flask de Andrés en el puerto 5000."""
    server_path = os.path.join(os.path.dirname(__file__), "server.py")
    if not os.path.exists(server_path):
        error("No se encontró server.py")
        sys.exit(1)

    proc = subprocess.Popen(
        [sys.executable, server_path],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    procesos.append(proc)

    # Esperar a que el backend esté listo (buscar el mensaje de "corriendo")
    def leer_salida():
        for linea in iter(proc.stdout.readline, ''):
            linea = linea.rstrip()
            if linea:
                print(f"  {Color.DIM}[backend]{Color.RESET} {linea}")

    hilo = threading.Thread(target=leer_salida, daemon=True)
    hilo.start()

    return proc

def arrancar_frontend():
    """Arranca un servidor HTTP para el frontend de María en el puerto 8080."""
    # Verificar que index.html existe
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    if not os.path.exists(index_path):
        error("No se encontró index.html")
        sys.exit(1)

    proc = subprocess.Popen(
        [sys.executable, "-m", "http.server", "8080"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        encoding='utf-8',
        errors='replace'
    )
    procesos.append(proc)

    # Leer salida en segundo plano
    def leer_salida():
        for linea in iter(proc.stdout.readline, ''):
            linea = linea.rstrip()
            if linea:
                print(f"  {Color.DIM}[frontend]{Color.RESET} {linea}")

    hilo = threading.Thread(target=leer_salida, daemon=True)
    hilo.start()

    return proc

def limpiar(*args):
    """Detiene ambos servidores al salir."""
    print(f"\n{Color.AMARILLO}Apagando servidores...{Color.RESET}")
    for proc in procesos:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()
    ok("Servidores detenidos")
    sys.exit(0)

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    os.system("")  # Habilita colores ANSI en Windows

    ip = obtener_ip_local()

    print()
    print(f"  {Color.BOLD}{Color.VERDE}============================================{Color.RESET}")
    print(f"  {Color.BOLD}{Color.VERDE}       VisionVR -- Arranque                {Color.RESET}")
    print(f"  {Color.BOLD}{Color.VERDE}============================================{Color.RESET}")
    print()

    # Paso 1: Verificaciones
    info("Verificando entorno...")
    verificar_python()
    verificar_dependencias()
    print()

    # Registrar señal para limpiar al salir
    signal.signal(signal.SIGINT, limpiar)
    signal.signal(signal.SIGTERM, limpiar)

    # Paso 2: Arrancar backend
    info("Arrancando backend (Flask + YOLO) en puerto 5000...")
    arrancar_backend()

    # Dar tiempo al backend para cargar el modelo YOLO
    time.sleep(3)

    # Paso 3: Arrancar frontend
    info("Arrancando frontend (HTTP) en puerto 8080...")
    arrancar_frontend()
    time.sleep(1)

    # Paso 4: Mostrar resumen
    print()
    print(f"  {Color.BOLD}{Color.VERDE}============================================{Color.RESET}")
    print(f"  {Color.BOLD}{Color.VERDE}       [OK] Todo corriendo                 {Color.RESET}")
    print(f"  {Color.BOLD}{Color.VERDE}============================================{Color.RESET}")
    print()
    print(f"  {Color.BOLD}Backend  (Andres):{Color.RESET}  http://localhost:5000")
    print(f"  {Color.BOLD}Frontend (Maria):{Color.RESET}   http://localhost:8080")
    print()
    print(f"  {Color.BOLD}{Color.CYAN}--- Para el Meta Quest 3 ---{Color.RESET}")
    print(f"  {Color.BOLD}Abre en el Quest:{Color.RESET}   http://{ip}:8080")
    print(f"  {Color.BOLD}IP del servidor:{Color.RESET}    http://{ip}:5000")
    print(f"  {Color.DIM}(escribe esa IP en el campo del frontend){Color.RESET}")
    print()
    print(f"  {Color.DIM}Presiona Ctrl+C para detener ambos servidores{Color.RESET}")
    print()

    # Paso 5: Abrir navegador
    webbrowser.open(f"http://localhost:8080")

    # Mantener vivo hasta Ctrl+C
    try:
        while True:
            # Verificar que los procesos siguen vivos
            for proc in procesos:
                if proc.poll() is not None:
                    error(f"Un servidor se detuvo inesperadamente (código: {proc.returncode})")
                    limpiar()
            time.sleep(2)
    except KeyboardInterrupt:
        limpiar()

if __name__ == "__main__":
    main()
