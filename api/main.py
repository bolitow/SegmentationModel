import os, io, glob
from pathlib import Path
from typing import List, Dict, Tuple
from fastapi import FastAPI, UploadFile, File, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.metrics import jaccard_score, accuracy_score
import time

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = os.getenv("MODELS_DIR", str(BASE_DIR / "api" / "models"))
DATA_DIR = os.getenv("DATA_DIR", str(BASE_DIR / "api" / "data"))
IMG_SIZE = (1024, 512)

print(f"BASE_DIR: {BASE_DIR}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"DATA_DIR: {DATA_DIR}")
print(f"Modèles trouvés: {glob.glob(os.path.join(MODELS_DIR, '*.keras'))}")

# ──────────────────────────────────────────────
# Gestion des palettes de couleurs
# ──────────────────────────────────────────────

# Définissons clairement la correspondance entre les indices et les classes
CLASS_NAMES = [
    "road",           # 0
    "sidewalk",       # 1
    "building",       # 2
    "wall",           # 3
    "fence",          # 4
    "pole",           # 5
    "traffic_light",  # 6
    "traffic_sign",   # 7
    "vegetation",     # 8
    "terrain",        # 9
    "sky",            # 10
    "person",         # 11
    "rider",          # 12
    "car",            # 13
    "truck",          # 14
    "bus",            # 15
    "train",          # 16
    "motorcycle",     # 17
    "bicycle",        # 18
    "parking",        # 19
    "rail_track",     # 20
    "guard_rail",     # 21
    "bridge",         # 22
    "tunnel",         # 23
    "polegroup",      # 24
]

# Palette de couleurs basée sur votre masque réel
# J'ai analysé les couleurs visibles dans votre image
COLOR_PALETTE = {
    "road": (128, 64, 128),        # Violet/mauve pour la route
    "sidewalk": (244, 35, 232),    # Rose pour le trottoir
    "building": (70, 70, 70),      # Gris foncé pour les bâtiments
    "wall": (102, 102, 156),       # Gris bleuté
    "fence": (190, 153, 153),      # Gris rosé
    "pole": (153, 153, 153),       # Gris
    "traffic_light": (250, 170, 30), # Jaune
    "traffic_sign": (220, 220, 0),   # Jaune vif
    "vegetation": (107, 142, 35),    # Vert
    "terrain": (152, 251, 152),      # Vert clair
    "sky": (70, 130, 180),          # Bleu ciel
    "person": (220, 20, 60),        # Rouge
    "rider": (255, 0, 0),           # Rouge vif
    "car": (0, 0, 142),             # Bleu foncé
    "truck": (0, 0, 70),            # Bleu très foncé
    "bus": (0, 60, 100),            # Bleu-vert foncé
    "train": (0, 80, 100),          # Bleu-vert
    "motorcycle": (0, 0, 230),      # Bleu
    "bicycle": (119, 11, 32),       # Bordeaux
    "parking": (250, 170, 160),     # Rose clair
    "rail_track": (230, 150, 140),  # Rose-brun
    "guard_rail": (180, 165, 180),  # Gris clair
    "bridge": (150, 100, 100),      # Brun
    "tunnel": (150, 120, 90),       # Brun clair
    "polegroup": (153, 153, 153),   # Gris
}

def create_palette_from_mapping():
    """
    Crée une palette de 256*3 valeurs pour PIL à partir de notre mapping
    """
    palette = []
    
    # Pour chaque indice possible (0-255)
    for i in range(256):
        if i < len(CLASS_NAMES):
            class_name = CLASS_NAMES[i]
            if class_name in COLOR_PALETTE:
                color = COLOR_PALETTE[class_name]
            else:
                color = (0, 0, 0)  # Noir par défaut
        else:
            color = (0, 0, 0)  # Noir pour les indices non utilisés
        
        palette.extend(color)
    
    return palette

# Créer la palette une seule fois
SEGMENTATION_PALETTE = create_palette_from_mapping()

# ──────────────────────────────────────────────
# Gestion des modèles
# ──────────────────────────────────────────────
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
            print(f"Modèle {model_name} chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle {model_name}: {e}")

    return loaded_models

# Charger les modèles au démarrage
load_available_models()

def get_current_model():
    return loaded_models.get(current_model_name) if current_model_name else None

# ──────────────────────────────────────────────
# Fonctions de traitement d'image
# ──────────────────────────────────────────────

def preprocess(img: Image.Image) -> np.ndarray:
    """
    Prépare l'image pour la prédiction
    """
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def postprocess(mask: np.ndarray, debug_info: bool = False) -> Tuple[Image.Image, Dict]:
    """
    Convertit la sortie du modèle en image masque colorée
    
    Args:
        mask: Sortie du modèle (peut être 3D avec probabilités ou 2D avec indices)
        debug_info: Si True, retourne des informations de debug
    
    Returns:
        Image PIL colorée et dictionnaire d'informations de debug
    """
    # S'assurer que le masque est 2D avec des indices de classe
    mask = np.squeeze(mask)
    if mask.ndim == 3:
        # Si c'est une sortie softmax, prendre l'argmax
        mask = np.argmax(mask, axis=-1)
    
    # Convertir en uint8
    mask = mask.astype(np.uint8)
    
    # Créer des informations de debug
    debug_data = {}
    if debug_info:
        unique_classes = np.unique(mask)
        debug_data['unique_predicted_classes'] = unique_classes.tolist()
        debug_data['class_distribution'] = {
            int(cls): int(np.sum(mask == cls)) 
            for cls in unique_classes
        }
        debug_data['predicted_class_names'] = {
            int(cls): CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else f"unknown_{cls}"
            for cls in unique_classes
        }
    
    # Créer l'image en mode palette
    pred_img = Image.fromarray(mask, mode="P")
    
    # Appliquer notre palette de couleurs
    pred_img.putpalette(SEGMENTATION_PALETTE)
    
    return pred_img, debug_data

def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques entre le GT et la prédiction
    """
    # S'assurer que les deux masques ont la même forme
    if gt_mask.shape != pred_mask.shape:
        # Redimensionner si nécessaire
        if len(gt_mask.shape) == 3:
            # Si le GT est en couleur, le convertir en indices
            # Cette partie nécessiterait un mapping inverse des couleurs vers indices
            raise ValueError("GT mask doit être en mode indices, pas en RGB")
    
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # Calculer l'IoU pour chaque classe présente
    unique_classes = np.unique(np.concatenate([gt_flat, pred_flat]))
    iou_per_class = {}
    
    for cls in unique_classes:
        gt_binary = (gt_flat == cls)
        pred_binary = (pred_flat == cls)
        
        intersection = np.sum(gt_binary & pred_binary)
        union = np.sum(gt_binary | pred_binary)
        
        if union > 0:
            iou_per_class[int(cls)] = float(intersection / union)
        else:
            iou_per_class[int(cls)] = 0.0
    
    # IoU moyen
    mean_iou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0

    return {
        "accuracy": float(accuracy),
        "mean_iou": float(mean_iou),
        "iou_per_class": iou_per_class
    }

# ──────────────────────────────────────────────
# API FastAPI
# ──────────────────────────────────────────────

app = FastAPI(
    title="Cityscapes Segmentation API",
    description="API de segmentation sémantique avec gestion des palettes de couleurs",
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

@app.get("/")
def root():
    """Point d'entrée de l'API"""
    return {
        "message": "API de segmentation Cityscapes",
        "endpoints": {
            "/models": "Liste des modèles disponibles",
            "/current_model": "Modèle actuellement sélectionné",
            "/images": "Liste des images disponibles",
            "/predict": "Prédire un masque de segmentation",
            "/debug/classes": "Voir le mapping des classes et couleurs"
        }
    }

@app.get("/models", response_model=List[str])
def list_models():
    """Retourne la liste des modèles disponibles"""
    return list(loaded_models.keys())

@app.get("/current_model")
def get_current_model_info():
    """Retourne des informations sur le modèle actuellement sélectionné"""
    if current_model_name and current_model_name in loaded_models:
        model = loaded_models[current_model_name]
        return {
            "model": current_model_name,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_classes": model.output_shape[-1] if len(model.output_shape) > 3 else "unknown"
        }
    return {"model": None, "error": "Aucun modèle sélectionné"}

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
    """Liste les images disponibles"""
    paths = glob.glob(os.path.join(DATA_DIR, "images", "*.png"))
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]

@app.get("/images/{image_id}")
def get_image(image_id: str):
    """Récupère une image spécifique"""
    path = os.path.join(DATA_DIR, "images", f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "Image non trouvée")
    with open(path, "rb") as f:
        return Response(f.read(), media_type="image/png")

@app.get("/masks/{image_id}")
def get_gt_mask(image_id: str):
    """Récupère le masque ground truth"""
    path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if not os.path.exists(path):
        raise HTTPException(404, "Masque GT non trouvé")
    with open(path, "rb") as f:
        return Response(f.read(), media_type="image/png")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Effectue une prédiction sur une image uploadée
    """
    model = get_current_model()
    if model is None:
        raise HTTPException(500, "Aucun modèle disponible")

    # Lire l'image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    
    # Prétraitement
    x = preprocess(img)
    
    # Prédiction
    y = model.predict(x, verbose=0)
    
    # Post-traitement
    mask_img, _ = postprocess(y, debug_info=False)
    
    # Retourner l'image
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    return Response(buf.getvalue(), media_type="image/png")

@app.post("/predict_with_debug")
async def predict_with_debug(file: UploadFile = File(...)):
    """
    Effectue une prédiction avec informations de debug
    """
    model = get_current_model()
    if model is None:
        raise HTTPException(500, "Aucun modèle disponible")

    image_id = Path(file.filename).stem

    # Lire l'image
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    
    # Prétraitement
    x = preprocess(img)
    
    # Prédiction avec timing
    start_time = time.time()
    y = model.predict(x, verbose=0)
    prediction_time = time.time() - start_time
    
    # Post-traitement avec debug
    mask_img, debug_info = postprocess(y, debug_info=True)
    
    # Préparer la réponse
    result = {
        "model_used": current_model_name,
        "prediction_time": float(prediction_time),
        "debug_info": debug_info,
        "image_id": image_id
    }
    
    # Sauvegarder l'image pour référence
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    
    # Encoder l'image en base64 pour l'inclure dans la réponse JSON
    import base64
    result["mask_base64"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return result

@app.get("/debug/classes")
def debug_classes():
    """
    Affiche le mapping entre indices, noms de classes et couleurs
    """
    mapping = []
    for i, class_name in enumerate(CLASS_NAMES):
        if class_name in COLOR_PALETTE:
            color = COLOR_PALETTE[class_name]
            mapping.append({
                "index": i,
                "class_name": class_name,
                "color_rgb": color,
                "color_hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
            })
    
    return {
        "total_classes": len(CLASS_NAMES),
        "class_mapping": mapping
    }

@app.get("/debug/analyze_gt/{image_id}")
def analyze_gt(image_id: str):
    """
    Analyse un masque GT pour comprendre son format
    """
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if not os.path.exists(gt_path):
        raise HTTPException(404, "Masque GT non trouvé")
    
    gt_img = Image.open(gt_path)
    gt_array = np.array(gt_img)
    
    info = {
        "image_id": image_id,
        "mode": gt_img.mode,
        "size": gt_img.size,
        "shape": gt_array.shape,
        "dtype": str(gt_array.dtype),
    }
    
    if gt_img.mode == 'P':
        # Mode palette
        unique_indices = np.unique(gt_array)
        info["unique_indices"] = unique_indices.tolist()
        info["num_unique_indices"] = len(unique_indices)
        
        # Récupérer la palette
        palette = gt_img.getpalette()
        if palette:
            colors_used = {}
            for idx in unique_indices:
                if idx < len(palette) // 3:
                    r = palette[idx * 3]
                    g = palette[idx * 3 + 1]
                    b = palette[idx * 3 + 2]
                    colors_used[int(idx)] = f"RGB({r}, {g}, {b})"
            info["colors_used"] = colors_used
    
    elif gt_img.mode == 'RGB':
        # Mode RGB - analyser les couleurs uniques
        h, w, c = gt_array.shape
        pixels = gt_array.reshape(-1, c)
        unique_colors = np.unique(pixels, axis=0)
        
        info["num_unique_colors"] = len(unique_colors)
        info["unique_colors"] = [
            f"RGB({r}, {g}, {b})" 
            for r, g, b in unique_colors[:20]  # Limiter à 20 pour la lisibilité
        ]
        
        if len(unique_colors) > 20:
            info["note"] = f"Affichage limité aux 20 premières couleurs sur {len(unique_colors)}"
    
    else:
        # Autre mode
        unique_values = np.unique(gt_array)
        info["unique_values"] = unique_values.tolist()
        info["num_unique_values"] = len(unique_values)
    
    return info
