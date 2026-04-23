import os
import google.generativeai as genai
from dotenv import load_dotenv
import traceback

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if api_key:
    api_key = api_key.strip("'").strip('"')
    genai.configure(api_key=api_key)
    try:
        models_to_try = ["gemini-2.0-flash", "gemini-2.5-flash"]
        long_prompt = "Esta es una película ficticia con una sinopsis extremadamente larga que describe todos los detalles imaginables de la trama. " * 300
        for m in models_to_try:
            try:
                print(f"Testing {m} (Short)...")
                model = genai.GenerativeModel(m)
                response = model.generate_content("Hola, dime TEST")
                print(f" - SUCCESS Short! {m}")
                
                print(f"Testing {m} (Long)...")
                response = model.generate_content(long_prompt)
                print(f" - SUCCESS Long! {m}")
            except Exception as e:
                print(f" - FAILED {m}: {repr(e)[:200]}")
    except Exception as e:
        print("Error en Gemini:")
        traceback.print_exc()

