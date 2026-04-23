#!/usr/bin/env bash
# Salir si hay algún error
set -o errexit

echo "Instalando dependencias de Python..."
# Instalamos la versión CPU de PyTorch para ahorrar muchísimo espacio en el servidor y evitar que crashee
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalamos el resto de las dependencias
pip install -r requirements.txt

echo "Descargando modelo de NLP (spaCy)..."
python -m spacy download es_core_news_sm

echo "Generando base de datos vectorial y procesando embeddings (puede tardar un poco)..."
python scripts/setup_data.py

echo "Build completado exitosamente!"
