import os, io, glob
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import jaccard_score, accuracy_score
import time

# ──────────────────────────────────────────────
# Configuration pour Azure App Service
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # Racine du projet
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "api" / "models"))
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "api" / "data"))
IMG_SIZE = (1024, 512)

print(f"BASE_DIR: {BASE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"Modèles trouvés: {glob.glob(os.path.join(MODELS_DIR, '*.keras'))}")

# Dictionnaire des modèles chargés
loaded_models = {}
current_model_name = None


def load_available_models():
    global loaded_models, current_model_name
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.keras"))

    for model_path in model_files:
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        try:
            loaded_models[model_name] = tf.keras.models.load_model(model_path, compile=False)
            if current_model_name is None:
                current_model_name = model_name
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name}: {e}")

    return loaded_models


# Charger tous les modèles au démarrage
load_available_models()


def get_current_model():
    return loaded_models.get(current_model_name) if current_model_name else None


DATA_DIR = os.getenv("DATA_DIR", "data")  # images & masques GT

# 1) Palette Cityscapes trainId (25 couleurs indexées 0…24)
CITYSCAPES_TRAINID_PALETTE = [
    (128, 64, 128),  # 0 road
    (244, 35, 232),  # 1 sidewalk
    (250, 170, 160),  # 2 parking
    (230, 150, 140),  # 3 rail track
    (70, 70, 70),  # 4 building
    (102, 102, 156),  # 5 wall
    (190, 153, 153),  # 6 fence
    (180, 165, 180),  # 7 guard rail
    (150, 100, 100),  # 8 bridge
    (150, 120, 90),  # 9 tunnel
    (153, 153, 153),  # 10 pole
    (153, 153, 153),  # 11 polegroup
    (250, 170, 30),  # 12 traffic light
    (220, 220, 0),  # 13 traffic sign
    (107, 142, 35),  # 14 vegetation
    (152, 251, 152),  # 15 terrain
    (70, 130, 180),  # 16 sky
    (220, 20, 60),  # 17 person
    (255, 0, 0),  # 18 rider
    (0, 0, 142),  # 19 car
    (0, 0, 70),  # 20 truck
    (0, 60, 100),  # 21 bus
    (0, 80, 100),  # 22 train
    (0, 0, 230),  # 23 motorcycle
    (119, 11, 32),  # 24 bicycle
]

# 1) Aplatir la liste de tuples en une seule liste d'entiers
flat_palette = []
for color in CITYSCAPES_TRAINID_PALETTE:
    flat_palette.extend(color)
flat_palette.extend([0] * (256 * 3 - len(flat_palette)))


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def postprocess(mask: np.ndarray, image_id: str) -> Image.Image:
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    mask = mask.astype(np.uint8)

    pred = Image.fromarray(mask, mode="P")

    # tentative de copier la palette du GT coloré…
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}_gtFine_color.png")
    if os.path.exists(gt_path):
        gt = Image.open(gt_path)
        palette = gt.getpalette()
        if palette is None:
            # si None (mode RGB), on retombe sur la palette statique
            palette = flat_palette
    else:
        palette = flat_palette

    pred.putpalette(palette)
    return pred


def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Calcule les métriques de performance entre GT et prédiction"""
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    accuracy = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, average='weighted')

    return {
        "accuracy": accuracy,
        "iou": iou
    }


# ──────────────────────────────────────────────
app = FastAPI(
    title="Cityscapes Segmentation API",
    description="Prédit un masque de segmentation U-Net-MobileNetV2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # front Streamlit hébergé ailleurs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/models", response_model=List[str])
def list_models():
    """Retourne la liste des modèles disponibles"""
    return list(loaded_models.keys())


@app.get("/current_model")
def get_current_model_info():
    """Retourne des informations sur le modèle actuellement sélectionné"""
    return {"model": current_model_name}


@app.post("/set_model/{model_name}")
def set_model(model_name: str):
    """Change le modèle actuellement utilisé"""
    global current_model_name
    if model_name not in loaded_models:
        raise HTTPException(404, f"Modèle {model_name} non trouvé")
    current_model_name = model_name
    return {"message": f"Modèle {model_name} sélectionné", "model": current_model_name}


@app.get("/images", response_model=List[str])
def list_images():
    paths = glob.glob(os.path.join(DATA_DIR, "images", "*.png"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


@app.get("/images/{image_id}")
def get_image(image_id: str):
    path = os.path.join(DATA_DIR, "images", f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "Image inconnue")
    return Response(open(path, "rb").read(), media_type="image/png")


@app.get("/masks/{image_id}")
def get_gt_mask(image_id: str):
    path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "Masque GT inconnu")
    return Response(open(path, "rb").read(), media_type="image/png")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Vérifier qu'un modèle est disponible
    model = get_current_model()
    if model is None:
        raise HTTPException(500, "Aucun modèle disponible")

    image_id = Path(file.filename).stem

    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    x = preprocess(img)
    y = model.predict(x, verbose=0)
    mask_img = postprocess(y, image_id)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/predict_with_metrics")
async def predict_with_metrics(file: UploadFile = File(...)):
    """Fait une prédiction et calcule les métriques si le GT est disponible"""
    # Vérifier qu'un modèle est disponible
    model = get_current_model()
    if model is None:
        raise HTTPException(500, "Aucun modèle disponible")

    image_id = Path(file.filename).stem

    # Lire et traiter l'image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    x = preprocess(img)

    # Prédiction avec mesure du temps
    start_time = time.time()
    y = model.predict(x, verbose=0)
    prediction_time = time.time() - start_time

    # Convertir la prédiction en masque
    pred_mask = np.squeeze(y)
    if pred_mask.ndim == 3:
        pred_mask = np.argmax(pred_mask, axis=-1)

    result = {
        "model_used": current_model_name,
        "prediction_time": prediction_time,
    }

    # Essayer de calculer les métriques si le GT existe
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if os.path.exists(gt_path):
        try:
            gt_img = Image.open(gt_path)
            gt_mask = np.array(gt_img.resize(IMG_SIZE, Image.NEAREST))

            metrics = calculate_metrics(gt_mask, pred_mask)
            result["metrics"] = metrics
        except Exception as e:
            result["metrics"] = {"error": f"Erreur lors du calcul des métriques: {str(e)}"}
    else:
        result["metrics"] = {"error": "Ground truth non disponible"}

    return result
