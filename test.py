import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv("API_KEY.env")
genai.configure(api_key=os.getenv("Gemini_API_key"))

models = genai.list_models()
for m in models:
    print(m.name, " - ", m.supported_generation_methods)
