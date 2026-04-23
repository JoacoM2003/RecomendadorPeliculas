# 🎬 TP Ciencia de Datos — Recomendador de Películas RAG

Motor de recomendación de películas inteligente que combina **búsqueda vectorial semántica (FAISS)** con un modelo de lenguaje **(Google Gemini)** para entender peticiones complejas en lenguaje natural.

---

## ¿Cómo funciona?

El sistema usa una arquitectura **RAG (Retrieval-Augmented Generation) híbrida** en dos etapas:

1. **Pipeline Offline (`scripts/setup_data.py`):**  
   Descarga el dataset de Kaggle, limpia los datos, genera embeddings multilingües con `SentenceTransformers` y los indexa en **FAISS** para búsquedas ultrarrápidas.

2. **Pipeline Online (cada consulta del usuario):**  
   - `spaCy` extrae actores, géneros y años del prompt.  
   - Se filtra el dataset de Pandas con esas restricciones exactas.  
   - Se busca semánticamente en FAISS dentro del subconjunto filtrado.  
   - Las películas candidatas se envían a **Gemini**, que genera una respuesta natural y justificada.

---

## Estructura del Proyecto

```text
Recomendador/
├── backend/
│   ├── main.py            # API FastAPI + serving del frontend
│   └── recommender.py     # Motor RAG: FAISS, spaCy, Pandas, Gemini
├── frontend/
│   ├── index.html         # UI
│   ├── app.js             # Lógica del cliente
│   └── style.css          # Estilos glassmorphism
├── scripts/
│   └── setup_data.py      # ETL: descarga, limpieza, embeddings y FAISS
├── data/                  # (Generado automáticamente) índices y modelos
├── .env                   # Tu API Key de Gemini (NO subir a git)
├── .env.example           # Plantilla del .env
└── requirements.txt       # Dependencias Python
```

---

## Instalación

### 1. Crear y activar el entorno virtual

> **Importante:** El entorno virtual **debe crearse dentro de la carpeta del proyecto**. No copies un venv de otra ubicación.

```powershell
# Crear el venv (usar py en Windows)
py -m venv venv

# Activar en PowerShell
.\venv\Scripts\Activate.ps1

# (Alternativa) Activar en CMD
venv\Scripts\activate.bat
```

### 2. Instalar dependencias

Con el venv activo:

```powershell
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### 3. Configurar la API Key de Gemini

Crea el archivo `.env` en la raíz del proyecto:

```env
GEMINI_API_KEY="tu_api_key_aqui"
```

Podés obtener una API Key gratis en: https://aistudio.google.com/app/apikey

### 4. Generar la base de datos vectorial (solo la primera vez)

```powershell
python scripts/setup_data.py
```

> Esto tarda varios minutos la primera vez. Genera los archivos en `/data/` que el backend usa para buscar películas. Las próximas veces que lo corras, detectará que ya existen y terminará instantáneamente.

### 5. Levantar el servidor

```powershell
# Desde la raíz del proyecto, con el venv activo:
python -m uvicorn backend.main:app --reload
```

> **El servidor debe iniciarse con `python -m uvicorn`** (no directamente con `uvicorn` ni con `python backend/main.py`), para que Python resuelva correctamente el módulo `backend`.

La aplicación estará disponible en: **http://localhost:8000**

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'backend'`
Estás ejecutando el servidor de forma incorrecta. Usá siempre:
```powershell
python -m uvicorn backend.main:app --reload
```

### `Fatal error in launcher: Unable to create process`
El entorno virtual está **roto** (fue copiado de otra ubicación). Recrealo desde cero:
```powershell
# Cierra el terminal que tiene el venv activo primero, luego:
Remove-Item -Recurse -Force .\venv
py -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

### `FileNotFoundError: Missing movies_processed.pkl`
Aún no generaste la base de datos vectorial. Ejecutá el paso 4:
```powershell
python scripts/setup_data.py
```

### Error de Gemini (429 / quota exceeded)
Significa que tu API Key superó el límite gratuito. El sistema igual devuelve las películas encontradas por FAISS. Podés obtener una API Key nueva en https://aistudio.google.com/app/apikey y reemplazarla en `.env`.

---

## Tecnologías

| Capa | Herramienta |
|---|---|
| Backend API | FastAPI + Uvicorn |
| LLM (Generación) | Google Gemini 2.5 Flash |
| Embeddings | SentenceTransformers (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Base de datos vectorial | FAISS (Meta) |
| NLP / NER | spaCy (`es_core_news_sm`) |
| Datos | Pandas + Scikit-Learn |
| Frontend | Vanilla JS, HTML5, CSS3 |