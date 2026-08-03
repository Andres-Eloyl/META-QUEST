import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
cliente = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)

try:
    respuesta = cliente.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[{"role": "user", "content": "Hola, respondeme con la palabra 'si'"}],
        max_tokens=10
    )
    print("Exito:", respuesta.choices[0].message.content)
except Exception as e:
    print("Error:", e)
