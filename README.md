# Segmentation Cityscapes - Démo Web

Ce dépôt contient une application web complète pour démontrer un modèle de segmentation sémantique sur des images Cityscapes.

## Structure du projet

```
webapp/
│
├── api/                         # API FastAPI
│   ├── main.py                  # Code principal de l'API
│   ├── requirements.txt         # Dépendances Python
│   ├── Dockerfile               # Configuration Docker
│   ├── Procfile                 # Configuration pour Heroku
│   └── data/                    # Données d'exemple
│       ├── images/              # Images d'entrée
│       └── masks/               # Masques de vérité terrain
│
├── app/                         # Frontend Streamlit
│   ├── app.py                   # Application Streamlit
│   ├── requirements.txt         # Dépendances Python
│   ├── Dockerfile               # Configuration Docker
│   └── Procfile                 # Configuration pour Heroku
│
└── README.md                    # Ce fichier
```

## Prérequis

- Python 3.11+
- Docker (optionnel, pour le déploiement)
- Un modèle entraîné au format `.keras` ou `.h5`

## Installation locale

### API FastAPI

1. Placez votre modèle entraîné dans le dossier `api/` (par défaut, le nom attendu est `best_model.keras`)
2. Placez quelques images d'exemple et leurs masques dans `api/data/images/` et `api/data/masks/`
3. Installez les dépendances :
   ```bash
   cd api
   pip install -r requirements.txt
   ```
4. Lancez l'API :
   ```bash
   uvicorn main:app --reload
   ```
   L'API sera accessible à l'adresse http://localhost:8000

### Frontend Streamlit

1. Installez les dépendances :
   ```bash
   cd app
   pip install -r requirements.txt
   ```
2. Lancez l'application :
   ```bash
   streamlit run app.py
   ```
   L'application sera accessible à l'adresse http://localhost:8501

## Déploiement

### Heroku

```bash
# API
cd webapp/api
heroku create my-seg-api
heroku stack:set container
git add . && git commit -m "API"
git push heroku main
heroku config:set MODEL_PATH=Model attention final.keras
heroku ps:scale web=1

# Frontend
cd ../app
heroku create my-seg-app
heroku stack:set container
git add . && git commit -m "Frontend"
git push heroku main
heroku config:set API_URL=https://my-seg-api.herokuapp.com
```

### Azure App Service (Docker)

1. Construisez et poussez les images Docker :
   ```bash
   docker build -t ghcr.io/<user>/seg-api:latest ./api
   docker push ghcr.io/<user>/seg-api:latest
   
   docker build -t ghcr.io/<user>/seg-app:latest ./app
   docker push ghcr.io/<user>/seg-app:latest
   ```

2. Créez des Web App for Containers sur Azure pointant vers ces images.

## Points d'attention

| Élément                                  | Où ajuster ?                                       |
| ---------------------------------------- | -------------------------------------------------- |
| **Chemin du modèle**                     | var d'env `MODEL_PATH` (+ monter le fichier)       |
| **Taille d'entrée**                      | `IMG_SIZE` dans `main.py` & preprocessing          |
| **Nombre de classes / palette couleurs** | fonction `postprocess` (mapping couleur ↔ classe)  |