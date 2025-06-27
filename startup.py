# startup.py (À CRÉER à la racine)
import sys
import os
from pathlib import Path

# Ajouter le répertoire courant au PATH Python
sys.path.insert(0, str(Path(__file__).parent))

# Configuration des chemins pour Azure
os.environ.setdefault("MODELS_DIR", "/home/site/wwwroot/api/models")
os.environ.setdefault("DATA_DIR", "/home/site/wwwroot/api/data")

# Importer l'application FastAPI
from api.main import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
