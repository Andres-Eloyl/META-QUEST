import os
from groq import Groq
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

print("Verificando GROQ_API_KEY...")
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: No se encontro GROQ_API_KEY en el archivo .env")
    exit(1)

print("Clave de API encontrada.")
print("Iniciando conexion con Groq...")

try:
    cliente_ia = Groq()
    
    objeto = "mochila"
    contexto = "entorno universitario"
    modelo_actual = "llama-3.3-70b-versatile" # Actualizado al modelo nuevo
    
    print(f"Enviando solicitud de prueba...")
    print(f"Modelo: {modelo_actual}")
    print(f"Objeto simulado: '{objeto}'\n")
    
    respuesta = cliente_ia.chat.completions.create(
        model=modelo_actual,
        messages=[{
            "role": "user",
            "content": (
                f"Eres un asistente virtual integrado en unas gafas de realidad aumentada. "
                f"El usuario esta apuntando a un '{objeto}'. Explica muy brevemente (1 o 2 oraciones) "
                f"que es y como podria usarse en el contexto de '{contexto}'. Habla en espanol, de forma "
                f"directa, clara e inteligente. No uses comillas."
            )
        }],
        max_tokens=150
    )
    texto = respuesta.choices[0].message.content.strip()
    
    print("--- RESPUESTA DE LA IA ---")
    print(texto)
    print("-----------------------------")
    print("\nExito! La conexion con Groq y el modelo esta funcionando perfectamente.")
    
except Exception as e:
    print("\nHubo un error al intentar conectar con Groq:")
    print(e)
