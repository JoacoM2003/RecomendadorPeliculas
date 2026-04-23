import pandas as pd
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import faiss
import os
import wget
import ast
import unicodedata
import joblib

# ==========================================
# ⚙️ CONFIGURACIÓN DE RUTAS Y CONSTANTES
# ==========================================
# URL de la API de Kaggle para el dataset de películas
DATASET_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/utkarshx27/movies-dataset"
DATASET_FILE_NAME = "movies-dataset.zip"
DATASET_CSV_NAME = "movie_dataset.csv"

# Resolviendo la ruta base del proyecto (subimos un nivel desde la carpeta "scripts")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data") # Carpeta donde se guardará todo el caché
DATASET_LOCAL_PATH = os.path.join(DATA_DIR, DATASET_CSV_NAME)

# Archivos de salida que usará nuestro Backend (API) de manera ligera
PROCESSED_DF_PATH = os.path.join(DATA_DIR, "movies_processed.pkl")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "movie_embeddings.faiss")
SCALER_PATH = os.path.join(DATA_DIR, "scaler.joblib")

def main():
    # 1️⃣ Crear directorio de datos si no existe
    os.makedirs(DATA_DIR, exist_ok=True)

    # 2️⃣ Descarga del Dataset (Si es la primera vez)
    if not os.path.exists(DATASET_LOCAL_PATH):
        print(f"Descargando el dataset de películas a: {DATA_DIR}")
        zip_path = os.path.join(DATA_DIR, DATASET_FILE_NAME)
        # Bajar el ZIP usando wget
        wget.download(DATASET_DOWNLOAD_URL, out=zip_path)
        print("\nDescarga del ZIP completada. Descomprimiendo...")
        # Descomprimir extrañando únicamente el CSV principal
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract(DATASET_CSV_NAME, DATA_DIR)
        os.remove(zip_path) # Limpiar el ZIP para ahorrar espacio
        print(f"Dataset '{DATASET_CSV_NAME}' descomprimido.")

    # 3️⃣ Cargar el dataset crudo a Memoria
    print(f"Cargando el dataset desde: {DATASET_LOCAL_PATH}")
    movies_df = pd.read_csv(DATASET_LOCAL_PATH)

    # ==========================================
    # 🧹 PREPROCESAMIENTO DE DATOS
    # ==========================================
    # Limpiamos el título
    movies_df.rename(columns={'title': 'CleanTitle'}, inplace=True)
    
    # Extraemos el año de la fecha de lazamiento y llenamos vacíos
    movies_df.loc[:, 'Year'] = pd.to_datetime(movies_df['release_date'], errors='coerce').dt.year
    movies_df.loc[:, 'Year'] = movies_df['Year'].astype('Int64').fillna(0)
    movies_df.loc[movies_df['Year'] == 0, 'Year'] = 'año desconocido'

    # Función robusta para sacar los nombres de la lista de diccionarios JSON que viene en 'genres'
    def parse_genres_robust(genres_str):
        if pd.isna(genres_str) or genres_str == '[]' or genres_str == '': return []
        try:
            genres_list = ast.literal_eval(genres_str)
            if isinstance(genres_list, list): return [d['name'] for d in genres_list if isinstance(d, dict) and 'name' in d]
        except:
            if ',' in genres_str: return [g.strip() for g in genres_str.split(',') if g.strip()]
            else: return [g.strip() for g in genres_str.split(' ') if g.strip()]
        return []

    # Función robusta para parsear la lista de actores ('cast')
    def parse_cast_robust(cast_str):
        if pd.isna(cast_str) or cast_str == '[]' or cast_str == '': return []
        try:
            cast_list = ast.literal_eval(cast_str)
            if isinstance(cast_list, list): return [d['name'] for d in cast_list if isinstance(d, dict) and 'name' in d]
        except:
            if ',' in cast_str: return [c.strip() for c in cast_str.split(',') if c.strip()]
            else: return [cast_str.strip()]
        return []

    # Aplicamos parseo en bruto
    movies_df['genres_parsed'] = movies_df['genres'].apply(parse_genres_robust)

    # Diccionario para unificar el idioma de géneros principales
    genre_translation = {
        "Action": "Acción", "Adventure": "Aventura", "Comedy": "Comedia", "Drama": "Drama",
        "Science Fiction": "Ciencia Ficción", "Horror": "Terror", "Thriller": "Thriller",
        "Romance": "Romance", "Animation": "Animación", "Documentary": "Documental",
        "Mystery": "Misterio", "Fantasy": "Fantasía", "Crime": "Crimen"
    }

    # Traducimos la lista resultante
    def traducir_generos(generos):
        return [genre_translation.get(g, g) for g in generos]

    movies_df['genres_traducidos'] = movies_df['genres_parsed'].apply(traducir_generos)
    movies_df['Genres'] = movies_df['genres_traducidos'].apply(lambda x: ', '.join(x) if x else '')

    # Extraemos elenco final a un String separado por comas
    movies_df['cast_parsed'] = movies_df['cast'].apply(parse_cast_robust)
    movies_df['CleanCast'] = movies_df['cast_parsed'].apply(lambda x: ', '.join(x) if x else '')

    movies_df.rename(columns={'vote_average': 'AvgRating'}, inplace=True)

    # 4️⃣ Creación del CORPUS: Concatenamos los datos para vectorizar su representación semántica
    movies_df['combined_text'] = (
        "Título: " + movies_df['CleanTitle'].fillna('') + ". " +
        "Año: " + movies_df['Year'].astype(str).fillna('') + ". " +
        "Géneros: " + movies_df['Genres'].fillna('') + ". " +
        "Actores: " + movies_df['CleanCast'].fillna('') + ". " +
        "Sinopsis: " + movies_df['overview'].fillna('')
    )

    # Limpiamos remanentes duplicados basados en Título y Año
    movies_df.drop_duplicates(subset=['CleanTitle', 'Year'], inplace=True, ignore_index=True)
    print(f"Dataset preprocesado. Filas totales limpidas: {len(movies_df)}")
    
    # Bypass: Si los 3 archivos de base de datos final ya existen, terminamos aquí para ganar tiempo.
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(PROCESSED_DF_PATH):
        print("Embeddings y dataset preprocesado ya existen, no hace falta regererar. Terminando...")
        return

    # CREACIÓN DE EMBEDDINGS Y FAISS INDEX
    print("Cargando modelo de embeddings (SentenceTransformer multi-idioma)...")
    embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Generando embeddings para todas las películas... (Esto tardará unos minutos).")
    # Codifica nuestro 'corpus' combinado en vectores de más de 300 dimensiones
    movie_embeddings = embedding_model.encode(movies_df['combined_text'].tolist(), show_progress_bar=True)

    # Escalamos (estandarizamos) los vectores para que el cálculo matemático de distancias en FAISS no tenga sesgos
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(movie_embeddings)

    # Obtenemos la dimensión generada
    D = scaled_embeddings.shape[1]
    
    # Creamos el índice topográfico FAISS usando distancia euclidiana (L2)
    index = faiss.IndexFlatL2(D)
    index.add(scaled_embeddings) # Inyectamos los vectores
    
    # GUARDADO A DISCO PARA CACHÉ DEL BACKEND
    print("Guardando base de datos y modelos matemáticos localmente...")
    faiss.write_index(index, FAISS_INDEX_PATH)    # Guarda el motor de búsqueda vectorial
    joblib.dump(scaler, SCALER_PATH)             # Guarda las reglas de escalado de la matriz
    movies_df.to_pickle(PROCESSED_DF_PATH)       # Guarda el dataframe pandas nativo (mucho más rápido y tipado que CSV)
    
    print("¡Preprocesamiento finalizado y archivos guardados limpiamente en la carpeta /data! ✨")

if __name__ == '__main__':
    main()
