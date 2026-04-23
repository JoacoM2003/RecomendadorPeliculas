import os
import re
import unicodedata
import pandas as pd
import faiss
import spacy
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

load_dotenv()

# Configuración Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    # Some versions of dotenv keep the quotes, so we strip them just in case
    GEMINI_API_KEY = GEMINI_API_KEY.strip("'").strip('"')
    genai.configure(api_key=GEMINI_API_KEY)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

GENRES = [
    "acción", "aventura", "comedia", "drama", "ciencia ficción", "terror", "thriller",
    "romance", "animación", "documental", "misterio", "fantasía", "crimen"
]

class RecommenderEngine:
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.movies_df = None
        self.index = None
        self.scaler = None
        self.nlp = None
        self.embedding_model = None
        self.is_loaded = False

    def load(self):
        if self.is_loaded:
            return
        print("Cargando modelo y datos en memoria...")
        
        df_path = os.path.join(self.data_dir, "movies_processed.pkl")
        faiss_path = os.path.join(self.data_dir, "movie_embeddings.faiss")
        scaler_path = os.path.join(self.data_dir, "scaler.joblib")
        
        if not os.path.exists(df_path):
            raise FileNotFoundError(f"Missing {df_path}. Run setup_data.py first.")
            
        self.movies_df = pd.read_pickle(df_path)
        self.index = faiss.read_index(faiss_path)
        self.scaler = joblib.load(scaler_path)
        
        self.nlp = spacy.load("es_core_news_sm")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.is_loaded = True
        print("Modelos y datos cargados exitosamente.")

    def quitar_tildes(self, texto):
        return ''.join(
            c for c in unicodedata.normalize('NFD', texto)
            if unicodedata.category(c) != 'Mn'
        ).lower()

    def extract_actor_genres_years(self, prompt):
        doc = self.nlp(prompt)
        actores = [ent.text for ent in doc.ents if ent.label_ == "PER"]

        años_spacy = []
        for ent in doc.ents:
            if ent.label_ == "DATE":
                matches = re.findall(r'\b(19\d{2}|20\d{2})\b', ent.text)
                años_spacy.extend(matches)

        años_manual = re.findall(r'\b(19\d{2}|20\d{2})\b', prompt)
        años = list(set(años_spacy + años_manual))

        generos_encontrados = []
        prompt_normalizado = self.quitar_tildes(prompt.lower())
        for genero in GENRES:
            if self.quitar_tildes(genero.lower()) in prompt_normalizado:
                generos_encontrados.append(genero.capitalize())

        return actores, generos_encontrados, años

    def recommend_movies(self, prompt, k=5):
        actores, generos, years = self.extract_actor_genres_years(prompt)
        generos_lower = [g.lower() for g in generos]
        actores = [a for a in actores if a.lower() not in generos_lower]

        df_filtrado = self.movies_df.copy()

        if actores:
            df_filtrado = df_filtrado[
                df_filtrado['CleanCast'].str.contains('|'.join(actores), case=False, na=False)
            ]
        if generos:
            df_filtrado = df_filtrado[
                df_filtrado['Genres'].str.lower().apply(
                    lambda x: any(gen in x for gen in generos_lower)
                )
            ]

        if years:
            years_int = []
            for y in years:
                try:
                    years_int.append(int(y))
                except ValueError:
                    pass
            if years_int:
                df_filtrado = df_filtrado[df_filtrado['Year'].isin(years_int)]

        if df_filtrado.empty:
            df_filtrado = self.movies_df.copy()

        indices_filtrados = df_filtrado.index.tolist()
        
        # Recuperamos los embeddings originales desde el DataFrame original y reconstruimos temp index
        # El DataFrame original index mapeado a FAISS
        # Sin embargo, FAISS no permite extraer los vectores fácilmente en IndexFlatL2 para pandas filtrado
        # Alternative: The original code loaded scaled_embeddings from memory, which is heavy.
        # Instead, let's re-encode only the filtered dataset text OR rebuild index.
        # It's faster to just keep the FAISS index intact and filter AFTER searching, OR read vectors.
        # For simplicity, since the original code kept `scaled_embeddings` in memory, we will read the vectors 
        # from FAISS index. IndexFlatL2 allows reconstruct_n.
        
        # En FAISS Flat, ntotal es el num total
        all_embeddings = np.array([self.index.reconstruct(i) for i in indices_filtrados]) if len(indices_filtrados) < 50000 else None
        
        if all_embeddings is not None and len(all_embeddings) > 0:
            temp_index = faiss.IndexFlatL2(all_embeddings.shape[1])
            temp_index.add(all_embeddings)
        else:
            # Fallback a index completo si es que todo
            temp_index = self.index

        prompt_emb = self.embedding_model.encode([prompt])
        scaled_prompt_emb = self.scaler.transform(prompt_emb)

        distances, indices = temp_index.search(scaled_prompt_emb, k)

        # Si usamos el index completo:
        if all_embeddings is None:
            # Fallback
            indices_totales = indices[0]
            recomendadas = self.movies_df.iloc[indices_totales].copy()
        else:
            # Los indices devueltos por temp_index se refieren a df_filtrado
            indices_reales = indices[0]
            recomendadas = df_filtrado.iloc[indices_reales].copy()
            
        recomendadas["Distancia"] = distances[0]

        scaler_minmax = MinMaxScaler()
        norm_rating = scaler_minmax.fit_transform(recomendadas[["AvgRating"]])
        norm_dist = scaler_minmax.fit_transform(recomendadas[["Distancia"]])

        recomendadas["Score"] = 0.3 * norm_rating.flatten() - 0.7 * norm_dist.flatten()
        recomendadas = recomendadas.sort_values(by="Score", ascending=False).reset_index(drop=True)

        return recomendadas

    def llamar_a_gemini(self, prompt_texto):
        if not GEMINI_API_KEY:
            return "No se ha configurado la API Key de Gemini en el archivo .env."
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        try:
            response = model.generate_content(prompt_texto)
            return response.text
        except Exception as e:
            print(f"Error Gemini: {e}")
            return "No pude generar una recomendación dinámica en este momento. Por favor revisa tu consola."

    def process_request(self, user_prompt, k=5):
        if not self.is_loaded:
            self.load()
            
        recommendations = self.recommend_movies(user_prompt, k)
        
        resumen = ""
        movies_list = []
        for i, row in recommendations.iterrows():
            movie_year = int(row['Year']) if pd.notna(row['Year']) and row['Year'] != 'año desconocido' else 'desconocido'
            
            resumen += (
                f"Título: {row['CleanTitle']} ({movie_year}), "
                f"Géneros: {row['Genres']}, "
                f"Actores: {row['CleanCast']}, "
                f"Rating Promedio: {round(row['AvgRating'], 1)}.\n"
                f"Sinopsis: {row['overview']}\n\n"
            )
            movies_list.append({
                "title": row['CleanTitle'],
                "year": movie_year,
                "genres": row['Genres'],
                "cast": row['CleanCast'],
                "rating": round(row['AvgRating'], 1),
                "overview": str(row['overview']) if pd.notna(row['overview']) else ""
            })
            
        prompt_llm = f"""
Actuá como un recomendador de películas experto y conciso. El usuario te pidió lo siguiente:
"{user_prompt}"

El sistema de búsqueda vectorial arrojó las siguientes {k} opciones candidatas:
{resumen}

INSTRUCCIONES CRÍTICAS:
1. **Sé muy breve y directo.** Ve al grano, no uses introducciones largas.
2. **Fíltralas:** Si alguna de las opciones arrojadas NO cumple realmente con lo que pidió el usuario (ejemplo: pidió zombies y la película no es de zombies, u otra temática específica), IGNÓRALA por completo y no la menciones en tu respuesta.
3. Si solo 1 o 2 películas encajan perfectamente, recomienda solo esas. No intentes rellenar recomendando las que no tienen que ver o que fueron devueltas de relleno por el buscador.
4. Justifica brevemente por qué recomiendas las que sí elegiste.
"""
        print("\n=== DEBUG: LONGITUD DEL PROMPT (CARACTERES) ===")
        print(len(prompt_llm))
        print("Esto equivale a aprox", len(prompt_llm)//4, "tokens.")
        print("===============================================\n")
        
        model_reply = self.llamar_a_gemini(prompt_llm)
        
        return {
            "reply": model_reply,
            "movies": movies_list
        }
