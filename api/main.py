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


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)


def postprocess(mask: np.ndarray, image_id: str) -> Image.Image:
    """
    Convertit la sortie du modèle en image masque avec la palette appropriée
    """
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)
    mask = mask.astype(np.uint8)
    
    # Créer l'image en mode palette
    pred = Image.fromarray(mask, mode="P")
    
    # Essayer de récupérer la palette depuis le masque GT
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    palette_applied = False
    
    if os.path.exists(gt_path):
        try:
            gt = Image.open(gt_path)
            
            # Si le GT est en mode palette, récupérer sa palette
            if gt.mode == 'P':
                palette = gt.getpalette()
                if palette:
                    pred.putpalette(palette)
                    palette_applied = True
                    print(f"Palette récupérée depuis le GT pour {image_id}")
            
            # Si le GT est en RGB, on doit extraire la palette
            elif gt.mode == 'RGB':
                # Convertir en array numpy
                gt_array = np.array(gt)
                
                # Trouver toutes les couleurs uniques
                unique_colors = {}
                gt_indexed = np.zeros((gt.size[1], gt.size[0]), dtype=np.uint8)
                
                # Créer un mapping couleur -> index
                for y in range(gt_array.shape[0]):
                    for x in range(gt_array.shape[1]):
                        color = tuple(gt_array[y, x])
                        if color not in unique_colors:
                            unique_colors[color] = len(unique_colors)
                        gt_indexed[y, x] = unique_colors[color]
                
                # Créer la palette à partir des couleurs uniques
                palette = []
                for i in range(256):
                    if i < len(unique_colors):
                        color = list(unique_colors.keys())[i]
                        palette.extend(color)
                    else:
                        palette.extend([0, 0, 0])
                
                pred.putpalette(palette)
                palette_applied = True
                print(f"Palette extraite du GT RGB pour {image_id}")
                
        except Exception as e:
            print(f"Erreur lors de la récupération de la palette: {e}")
    
    # Si aucune palette n'a pu être appliquée, créer une palette par défaut
    if not palette_applied:
        # Palette de secours basée sur ce qui semble être dans votre image
        # (vous pouvez ajuster ces couleurs selon vos besoins)
        default_palette = [
            (128, 64, 128),   # 0 - route (violet)
            (244, 35, 232),   # 1 - trottoir (rose)
            (70, 70, 70),     # 2 - bâtiment (gris foncé)
            (102, 102, 156),  # 3 - mur (gris bleu)
            (107, 142, 35),   # 4 - végétation (vert)
            (152, 251, 152),  # 5 - terrain (vert clair)
            (70, 130, 180),   # 6 - ciel (bleu)
            (220, 20, 60),    # 7 - personne (rouge)
            (0, 0, 142),      # 8 - voiture (bleu foncé)
            (0, 0, 70),       # 9 - camion (bleu très foncé)
            (0, 60, 100),     # 10 - bus (bleu foncé)
            (0, 0, 230),      # 11 - moto (bleu)
            (119, 11, 32),    # 12 - vélo (bordeaux)
            (250, 170, 30),   # 13 - feu de circulation (jaune)
            (220, 220, 0),    # 14 - panneau (jaune)
            (190, 153, 153),  # 15 - clôture (gris rosé)
            (153, 153, 153),  # 16 - poteau (gris)
            (180, 165, 180),  # 17 - barrière (gris clair)
            (150, 100, 100),  # 18 - pont (brun)
            (150, 120, 90),   # 19 - tunnel (brun clair)
            (250, 170, 160),  # 20 - parking (rose clair)
            (255, 0, 0),      # 21 - cycliste (rouge vif)
            (0, 80, 100),     # 22 - train (bleu-vert)
            (230, 150, 140),  # 23 - rail (rose-brun)
            (0, 0, 0),        # 24 - autre (noir)
        ]
        
        # Créer la palette complète (256 couleurs)
        flat_palette = []
        for i in range(256):
            if i < len(default_palette):
                flat_palette.extend(default_palette[i])
            else:
                flat_palette.extend([0, 0, 0])
        
        pred.putpalette(flat_palette)
        print(f"Palette par défaut appliquée pour {image_id}")
    
    return pred


def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """Calcule les métriques de performance entre GT et prédiction"""
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    accuracy = accuracy_score(gt_flat, pred_flat)
    iou = jaccard_score(gt_flat, pred_flat, average='weighted', zero_division=0)

    return {
        "accuracy": float(accuracy),
        "iou": float(iou)
    }


# ──────────────────────────────────────────────
app = FastAPI(
    title="Cityscapes Segmentation API",
    description="Prédit un masque de segmentation U-Net-MobileNetV2",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    with open(path, "rb") as f:
        return Response(f.read(), media_type="image/png")


@app.get("/masks/{image_id}")
def get_gt_mask(image_id: str):
    path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "Masque GT inconnu")
    with open(path, "rb") as f:
        return Response(f.read(), media_type="image/png")


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
    
    # Prédiction
    y = model.predict(x, verbose=0)
    
    # Post-traitement avec récupération de la palette
    mask_img = postprocess(y, image_id)

    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
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
        "prediction_time": float(prediction_time),
    }

    # Essayer de calculer les métriques si le GT existe
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if os.path.exists(gt_path):
        try:
            gt_img = Image.open(gt_path)
            
            # Si le GT est en mode P, on peut directement récupérer les indices
            if gt_img.mode == 'P':
                gt_mask = np.array(gt_img.resize(IMG_SIZE, Image.NEAREST))
            else:
                # Si c'est en RGB, il faut le convertir en indices
                # Cette partie pourrait nécessiter un ajustement selon votre format exact
                gt_img = gt_img.resize(IMG_SIZE, Image.NEAREST)
                gt_mask = np.array(gt_img)
                # Conversion RGB vers indices (à adapter selon votre mapping)
                # Pour l'instant, on suppose que le GT est déjà en mode P
                raise ValueError("GT en mode RGB non supporté pour les métriques")

            metrics = calculate_metrics(gt_mask, pred_mask)
            result["metrics"] = metrics
        except Exception as e:
            result["metrics"] = {"error": f"Erreur lors du calcul des métriques: {str(e)}"}
    else:
        result["metrics"] = {"error": "Ground truth non disponible"}

    return result


@app.get("/debug/palette/{image_id}")
def debug_palette(image_id: str):
    """Endpoint de debug pour vérifier la palette d'un masque GT"""
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if not os.path.exists(gt_path):
        raise HTTPException(404, "Masque GT inconnu")
    
    gt = Image.open(gt_path)
    info = {
        "mode": gt.mode,
        "size": gt.size,
    }
    
    if gt.mode == 'P':
        palette = gt.getpalette()
        if palette:
            # Afficher les 25 premières couleurs de la palette
            colors = []
            for i in range(min(25, len(palette)//3)):
                r = palette[i*3]
                g = palette[i*3 + 1]
                b = palette[i*3 + 2]
                colors.append(f"Index {i}: RGB({r}, {g}, {b})")
            info["palette_colors"] = colors
            
            # Trouver les indices utilisés dans l'image
            gt_array = np.array(gt)
            unique_indices = np.unique(gt_array)
            info["used_indices"] = unique_indices.tolist()
    
    elif gt.mode == 'RGB':
        # Analyser les couleurs uniques
        gt_array = np.array(gt)
        unique_colors = {}
        for y in range(gt_array.shape[0]):
            for x in range(gt_array.shape[1]):
                color = tuple(gt_array[y, x])
                if color not in unique_colors:
                    unique_colors[color] = 0
                unique_colors[color] += 1
        
        # Trier par fréquence
        sorted_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)
        info["unique_colors"] = [
            f"RGB{color}: {count} pixels" 
            for color, count in sorted_colors[:25]
        ]
        info["total_unique_colors"] = len(unique_colors)
    
    return info
