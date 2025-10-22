# list_models.py
import os
from dotenv import load_dotenv
import google.generativeai as genai

# .env dosyasını açıkça yükle
load_dotenv(dotenv_path=".env")

api_key = os.getenv("GOOGLE_API_KEY")
assert api_key, "GOOGLE_API_KEY bulunamadı (.env dosyanı kontrol et)."

genai.configure(api_key=api_key)

print("Desteklenen generateContent modelleri:")
for m in genai.list_models():
    if "generateContent" in getattr(m, "supported_generation_methods", []):
        print("-", m.name)
