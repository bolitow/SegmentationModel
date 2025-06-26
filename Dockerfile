# Dockerfile (à la racine)
FROM python:3.12-slim

WORKDIR /app

# Copier les fichiers de requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p api/models api/data/images api/data/masks

# Script de démarrage
COPY start.sh .
RUN chmod +x start.sh

# Exposer le port que Heroku va utiliser
EXPOSE $PORT

CMD ["./start.sh"]