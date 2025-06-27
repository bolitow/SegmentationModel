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
import base64

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
# Configuration des 8 macro-classes
# ──────────────────────────────────────────────

# Nombre de classes après réduction
N_CLASSES = 8

# Noms des 8 macro-classes
MACRO_CLASS_NAMES = [
    "Route/Trottoir",  # 0
    "Construction",    # 1
    "Objet",          # 2
    "Nature",         # 3
    "Ciel",           # 4
    "Humain",         # 5
    "Véhicule",       # 6
    "Void"            # 7
]

# Palette de couleurs pour les 8 macro-classes
# J'ai choisi des couleurs distinctes et représentatives pour chaque classe
MACRO_CLASS_COLORS = [
    (128, 64, 128),   # 0 - Route/Trottoir : violet/mauve
    (70, 70, 70),     # 1 - Construction : gris foncé
    (220, 220, 0),    # 2 - Objet : jaune (panneaux, feux)
    (107, 142, 35),   # 3 - Nature : vert
    (70, 130, 180),   # 4 - Ciel : bleu ciel
    (220, 20, 60),    # 5 - Humain : rouge
    (0, 0, 142),      # 6 - Véhicule : bleu foncé
    (0, 0, 0)         # 7 - Void : noir
]

# Mapping des IDs Cityscapes originaux vers les 8 macro-classes
# Ce mapping est extrait directement de votre notebook
CS2MACRO = {
    # Route/Trottoir (classe 0)
    7: 0,   # road
    8: 0,   # sidewalk
    9: 0,   # parking
    10: 0,  # rail track
    6: 0,   # ground
    
    # Construction (classe 1)
    11: 1,  # building
    12: 1,  # wall
    13: 1,  # fence
    15: 1,  # bridge
    14: 1,  # guard rail
    16: 1,  # tunnel
    
    # Objet (classe 2)
    17: 2,  # pole
    19: 2,  # traffic light
    20: 2,  # traffic sign
    18: 2,  # polegroup
    4: 2,   # static
    
    # Nature (classe 3)
    21: 3,  # vegetation
    22: 3,  # terrain
    
    # Ciel (classe 4)
    23: 4,  # sky
    
    # Humain (classe 5)
    24: 5,  # person
    25: 5,  # rider
    
    # Véhicule (classe 6)
    26: 6,  # car
    27: 6,  # truck
    28: 6,  # bus
    31: 6,  # train
    32: 6,  # motorcycle
    33: 6,  # bicycle
    1: 6,   # ego vehicle
    -1: 6,  # license plate
    29: 6,  # caravan
    30: 6,  # trailer
    5: 6,   # dynamic
    
    # Void (classe 7)
    0: 7,   # unlabeled
    2: 7,   # rectification border
    3: 7    # out of roi
}

def create_palette_for_8_classes():
    """
    Crée une palette PIL pour les 8 macro-classes.
    La palette doit contenir 256*3 valeurs (RGB pour chaque index de 0 à 255).
    """
    palette = []
    
    for i in range(256):
        if i < N_CLASSES:
            # Utiliser la couleur définie pour cette macro-classe
            color = MACRO_CLASS_COLORS[i]
        else:
            # Noir pour les indices non utilisés
            color = (0, 0, 0)
        
        palette.extend(color)
    
    return palette

# Créer la palette une seule fois
PALETTE_8_CLASSES = create_palette_for_8_classes()

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
            model = tf.keras.models.load_model(model_path, compile=False)
            loaded_models[model_name] = model
            
            if current_model_name is None:
                current_model_name = model_name
            
            # Vérifier que le modèle a bien 8 classes en sortie
            output_shape = model.output_shape
            if len(output_shape) >= 4:  # (batch, height, width, classes)
                num_classes = output_shape[-1]
                print(f"Modèle {model_name} chargé - Nombre de classes en sortie: {num_classes}")
                if num_classes != N_CLASSES:
                    print(f"⚠️  Attention: Le modèle prédit {num_classes} classes au lieu de {N_CLASSES}")
            
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
    Prépare l'image pour la prédiction.
    Redimensionne et normalise l'image.
    """
    img = img.resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def postprocess(mask: np.ndarray, debug_info: bool = False) -> Tuple[Image.Image, Dict]:
    """
    Convertit la sortie du modèle (8 classes) en image masque colorée.
    
    Args:
        mask: Sortie du modèle - shape (1, H, W, 8) avec probabilités
        debug_info: Si True, retourne des informations de debug
    
    Returns:
        Image PIL colorée et dictionnaire d'informations de debug
    """
    # Enlever la dimension batch
    mask = np.squeeze(mask)  # (H, W, 8)
    
    # Si le masque a 3 dimensions (H, W, C), prendre l'argmax pour obtenir les indices de classe
    if mask.ndim == 3:
        mask = np.argmax(mask, axis=-1)  # (H, W) avec valeurs 0-7
    
    # S'assurer que les valeurs sont bien entre 0 et 7
    mask = np.clip(mask, 0, N_CLASSES - 1)
    
    # Convertir en uint8
    mask = mask.astype(np.uint8)
    
    # Créer des informations de debug si demandé
    debug_data = {}
    if debug_info:
        unique_classes = np.unique(mask)
        debug_data['unique_predicted_classes'] = unique_classes.tolist()
        debug_data['class_distribution'] = {
            int(cls): {
                'name': MACRO_CLASS_NAMES[cls],
                'pixel_count': int(np.sum(mask == cls)),
                'percentage': float(np.sum(mask == cls) / mask.size * 100)
            }
            for cls in unique_classes
        }
    
    # Créer l'image en mode palette
    pred_img = Image.fromarray(mask, mode="P")
    
    # Appliquer la palette de couleurs pour les 8 classes
    pred_img.putpalette(PALETTE_8_CLASSES)
    
    return pred_img, debug_data

def convert_gt_to_8_classes(gt_mask: np.ndarray) -> np.ndarray:
    """
    Convertit un masque GT Cityscapes (avec IDs originaux) vers les 8 macro-classes.
    Utilise le mapping CS2MACRO.
    """
    # Créer un masque de sortie avec la valeur par défaut 7 (Void)
    macro_mask = np.full_like(gt_mask, 7, dtype=np.uint8)
    
    # Appliquer le mapping pour chaque ID Cityscapes
    for cityscapes_id, macro_id in CS2MACRO.items():
        macro_mask[gt_mask == cityscapes_id] = macro_id
    
    return macro_mask

def calculate_metrics(gt_mask: np.ndarray, pred_mask: np.ndarray) -> Dict[str, float]:
    """
    Calcule les métriques entre le GT et la prédiction.
    Les deux masques doivent être en format 8 classes.
    """
    # S'assurer que les deux masques sont 2D
    if gt_mask.ndim > 2:
        gt_mask = gt_mask.squeeze()
    if pred_mask.ndim > 2:
        pred_mask = pred_mask.squeeze()
    
    # S'assurer qu'ils ont la même forme
    if gt_mask.shape != pred_mask.shape:
        # Redimensionner le GT si nécessaire
        gt_img = Image.fromarray(gt_mask.astype(np.uint8), mode='L')
        gt_img = gt_img.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
        gt_mask = np.array(gt_img)
    
    gt_flat = gt_mask.flatten()
    pred_flat = pred_mask.flatten()

    # Accuracy globale
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    # IoU par classe
    iou_per_class = {}
    for cls in range(N_CLASSES):
        gt_binary = (gt_flat == cls)
        pred_binary = (pred_flat == cls)
        
        intersection = np.sum(gt_binary & pred_binary)
        union = np.sum(gt_binary | pred_binary)
        
        if union > 0:
            iou = float(intersection / union)
        else:
            iou = 0.0
            
        iou_per_class[MACRO_CLASS_NAMES[cls]] = iou
    
    # IoU moyen (en excluant les classes non présentes)
    ious = [iou for iou in iou_per_class.values() if iou > 0]
    mean_iou = np.mean(ious) if ious else 0.0

    return {
        "accuracy": float(accuracy),
        "mean_iou": float(mean_iou),
        "iou_per_class": iou_per_class
    }

# ──────────────────────────────────────────────
# API FastAPI
# ──────────────────────────────────────────────

app = FastAPI(
    title="Cityscapes Segmentation API - 8 Classes",
    description="API de segmentation sémantique avec réduction à 8 macro-classes",
    version="2.0"
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
        "message": "API de segmentation Cityscapes - 8 macro-classes",
        "classes": MACRO_CLASS_NAMES,
        "endpoints": {
            "/models": "Liste des modèles disponibles",
            "/current_model": "Modèle actuellement sélectionné",
            "/images": "Liste des images disponibles",
            "/predict": "Prédire un masque de segmentation",
            "/predict_with_metrics": "Prédiction avec métriques",
            "/debug/classes": "Informations sur les classes"
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
            "input_shape": list(model.input_shape),
            "output_shape": list(model.output_shape),
            "num_classes": model.output_shape[-1] if len(model.output_shape) > 3 else "unknown",
            "expected_classes": N_CLASSES
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
    """Récupère le masque ground truth avec conversion en 8 classes"""
    # Essayer différents chemins de fichiers
    possible_paths = [
        os.path.join(DATA_DIR, "masks", f"{image_id}.png"),
        os.path.join(DATA_DIR, "masks", f"{image_id}_gtFine_color.png"),
        os.path.join(DATA_DIR, "masks", f"{image_id}_gtFine_labelIds.png")
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        # Liste des fichiers disponibles pour debug
        try:
            mask_files = os.listdir(os.path.join(DATA_DIR, "masks"))
            available_files = [f for f in mask_files if image_id in f]
        except:
            available_files = []
        
        raise HTTPException(404, f"Masque GT non trouvé pour {image_id}. Fichiers disponibles: {available_files}")
    
    try:
        # Lire le masque GT
        gt_img = Image.open(path)
        
        # Debug: afficher les informations sur l'image
        print(f"DEBUG: Image path: {path}")
        print(f"DEBUG: Image mode: {gt_img.mode}")
        print(f"DEBUG: Image size: {gt_img.size}")
        
        # Gérer différents types de masques Cityscapes
        if "_gtFine_color.png" in path:
            # C'est un masque coloré Cityscapes - on doit le convertir en IDs de classes
            # Les masques colorés utilisent des couleurs spécifiques pour chaque classe
            # On va d'abord extraire les IDs depuis les couleurs
            
            # Dictionnaire des couleurs Cityscapes vers IDs
            CITYSCAPES_COLORS_TO_IDS = {
                (128, 64, 128): 7,    # road
                (244, 35, 232): 8,    # sidewalk
                (250, 170, 160): 9,   # parking
                (230, 150, 140): 10,  # rail track
                (70, 70, 70): 11,     # building
                (102, 102, 156): 12,  # wall
                (190, 153, 153): 13,  # fence
                (153, 153, 153): 17,  # pole
                (250, 170, 30): 19,   # traffic light
                (220, 220, 0): 20,    # traffic sign
                (107, 142, 35): 21,   # vegetation
                (152, 251, 152): 22,  # terrain
                (70, 130, 180): 23,   # sky
                (220, 20, 60): 24,    # person
                (255, 0, 0): 25,      # rider
                (0, 0, 142): 26,      # car
                (0, 0, 70): 27,       # truck
                (0, 60, 100): 28,     # bus
                (0, 80, 100): 31,     # train
                (0, 0, 230): 32,      # motorcycle
                (119, 11, 32): 33,    # bicycle
                (0, 0, 0): 0,         # unlabeled
                (81, 0, 81): 6,       # ground
                (150, 100, 100): 15,  # bridge
                (180, 165, 180): 14,  # guard rail
                (150, 120, 90): 16,   # tunnel
                (153, 153, 153): 18,  # polegroup
                (111, 74, 0): 4,      # static
                (255, 255, 255): -1,  # license plate
                (0, 0, 90): 29,       # caravan
                (0, 0, 110): 30,      # trailer
                (111, 74, 0): 5,      # dynamic
                (0, 0, 142): 1,       # ego vehicle
                (0, 0, 0): 2,         # rectification border
                (0, 0, 0): 3          # out of roi
            }
            
            # Convertir en RGB (ignorer le canal alpha si présent)
            gt_img = gt_img.convert('RGB')
                
            rgb_array = np.array(gt_img)
            print(f"DEBUG: RGB array shape after conversion: {rgb_array.shape}")
            
            # S'assurer que c'est bien un array 3D avec 3 canaux
            if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
                raise ValueError(f"Invalid RGB array shape: {rgb_array.shape}")
                
            height, width = rgb_array.shape[:2]
            gt_array = np.zeros((height, width), dtype=np.uint8)
            
            # Mapper chaque couleur vers son ID Cityscapes
            for color, cs_id in CITYSCAPES_COLORS_TO_IDS.items():
                mask = np.all(rgb_array == color, axis=2)
                gt_array[mask] = cs_id if cs_id >= 0 else 0  # Remplacer -1 par 0
                
        elif "_gtFine_labelIds.png" in path or gt_img.mode in ['L', 'I']:
            # C'est déjà un masque avec IDs de classes
            if gt_img.mode == 'I':
                gt_array = np.array(gt_img, dtype=np.int32)
            else:
                gt_array = np.array(gt_img)
        else:
            # Autre type de masque - essayer de le traiter comme un masque d'IDs
            # Gérer tous les modes possibles
            if gt_img.mode == 'RGBA':
                # Si RGBA, ignorer le canal alpha
                gt_img = gt_img.convert('RGB').convert('L')
            elif gt_img.mode == 'RGB':
                # Si RGB, convertir en niveaux de gris
                gt_img = gt_img.convert('L')
            elif gt_img.mode == 'P':
                # Si palette, convertir en niveaux de gris
                gt_img = gt_img.convert('L')
            
            gt_array = np.array(gt_img)
        
        print(f"DEBUG: Array shape après conversion: {gt_array.shape}")
        print(f"DEBUG: Array unique values: {np.unique(gt_array)}")
        print(f"DEBUG: Array min/max: {gt_array.min()}/{gt_array.max()}")
        
        # S'assurer que c'est bien un array 2D
        if gt_array.ndim != 2:
            print(f"ERROR: Array has {gt_array.ndim} dimensions, expected 2")
            gt_array = gt_array.squeeze()
            if gt_array.ndim != 2:
                raise ValueError(f"Impossible de convertir l'array en 2D. Shape: {gt_array.shape}")
        
        # Convertir les IDs Cityscapes en 8 macro-classes
        print("DEBUG: Converting Cityscapes IDs to 8 classes")
        gt_array_8classes = convert_gt_to_8_classes(gt_array)
        
        # Vérifier la distribution des classes après conversion
        unique_classes = np.unique(gt_array_8classes)
        print(f"DEBUG: Classes après conversion (8 classes): {unique_classes}")
        for cls in unique_classes:
            count = np.sum(gt_array_8classes == cls)
            print(f"  - Classe {cls} ({MACRO_CLASS_NAMES[cls]}): {count} pixels")
        
        # Créer une nouvelle image avec la palette 8 classes
        gt_8classes = Image.fromarray(gt_array_8classes.astype(np.uint8), mode='P')
        gt_8classes.putpalette(PALETTE_8_CLASSES)
        
        # Vérifier que la palette est bien appliquée
        print(f"DEBUG: Image finale mode: {gt_8classes.mode}")
        print(f"DEBUG: Palette length: {len(gt_8classes.getpalette()) if gt_8classes.mode == 'P' else 0}")
        
        # Retourner l'image
        buf = io.BytesIO()
        gt_8classes.save(buf, format="PNG")
        buf.seek(0)
        return Response(buf.getvalue(), media_type="image/png")
        
    except Exception as e:
        print(f"ERROR: Exception type: {type(e).__name__}")
        print(f"ERROR: Exception message: {str(e)}")
        import traceback
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        raise HTTPException(500, f"Erreur lors du traitement du masque: {str(e)}")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Effectue une prédiction sur une image uploadée.
    Retourne un masque de segmentation avec 8 classes.
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

@app.post("/predict_with_metrics")
async def predict_with_metrics(file: UploadFile = File(...)):
    """
    Fait une prédiction et calcule les métriques si le GT est disponible
    """
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

    # Post-traitement pour obtenir le masque prédit
    pred_img, debug_info = postprocess(y, debug_info=True)
    pred_mask = np.array(pred_img)

    result = {
        "model_used": current_model_name,
        "prediction_time": float(prediction_time),
        "debug_info": debug_info
    }

    # Essayer de calculer les métriques si le GT existe
    gt_path = os.path.join(DATA_DIR, "masks", f"{image_id}.png")
    if os.path.exists(gt_path):
        try:
            gt_img = Image.open(gt_path)
            gt_mask = np.array(gt_img)
            
            # Convertir le GT en 8 classes s'il est en format Cityscapes original
            if gt_mask.max() > 7:
                gt_mask = convert_gt_to_8_classes(gt_mask)
            
            # Redimensionner le GT à la taille de la prédiction
            if gt_mask.shape != pred_mask.shape:
                gt_img_resized = Image.fromarray(gt_mask.astype(np.uint8), mode='L')
                gt_img_resized = gt_img_resized.resize((pred_mask.shape[1], pred_mask.shape[0]), Image.NEAREST)
                gt_mask = np.array(gt_img_resized)

            metrics = calculate_metrics(gt_mask, pred_mask)
            result["metrics"] = metrics
        except Exception as e:
            result["metrics"] = {"error": f"Erreur lors du calcul des métriques: {str(e)}"}
    else:
        result["metrics"] = {"error": "Ground truth non disponible"}

    return result

@app.get("/debug/classes")
def debug_classes():
    """
    Affiche les informations sur les 8 macro-classes
    """
    class_info = []
    for i in range(N_CLASSES):
        color = MACRO_CLASS_COLORS[i]
        class_info.append({
            "index": i,
            "name": MACRO_CLASS_NAMES[i],
            "color_rgb": color,
            "color_hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        })
    
    # Ajouter le mapping Cityscapes -> macro-classes
    cityscapes_mapping = {}
    for cs_id, macro_id in CS2MACRO.items():
        if cs_id not in cityscapes_mapping:
            cityscapes_mapping[cs_id] = {
                "cityscapes_id": cs_id,
                "macro_class_id": macro_id,
                "macro_class_name": MACRO_CLASS_NAMES[macro_id]
            }
    
    return {
        "num_classes": N_CLASSES,
        "macro_classes": class_info,
        "cityscapes_to_macro_mapping": dict(sorted(cityscapes_mapping.items()))
    }

@app.post("/predict_with_debug")
async def predict_with_debug(file: UploadFile = File(...)):
    """
    Effectue une prédiction avec informations détaillées pour le debug
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
    
    # Informations sur la sortie du modèle
    model_output_info = {
        "shape": y.shape,
        "dtype": str(y.dtype),
        "min_value": float(y.min()),
        "max_value": float(y.max()),
        "contains_nan": bool(np.isnan(y).any())
    }
    
    # Post-traitement avec debug
    mask_img, debug_info = postprocess(y, debug_info=True)
    
    # Préparer la réponse
    result = {
        "model_used": current_model_name,
        "prediction_time": float(prediction_time),
        "image_id": image_id,
        "model_output_info": model_output_info,
        "prediction_info": debug_info
    }
    
    # Encoder l'image en base64
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)
    result["mask_base64"] = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return result

@app.get("/debug/mask/{image_id}")
def debug_gt_mask(image_id: str):
    """
    Debug endpoint pour analyser le masque GT et sa conversion
    """
    # Essayer différents chemins de fichiers
    possible_paths = [
        os.path.join(DATA_DIR, "masks", f"{image_id}.png"),
        os.path.join(DATA_DIR, "masks", f"{image_id}_gtFine_color.png"),
        os.path.join(DATA_DIR, "masks", f"{image_id}_gtFine_labelIds.png")
    ]
    
    path = None
    for p in possible_paths:
        if os.path.exists(p):
            path = p
            break
    
    if path is None:
        raise HTTPException(404, f"Masque GT non trouvé pour {image_id}")
    
    try:
        # Lire le masque GT
        gt_img = Image.open(path)
        
        debug_info = {
            "file_path": path,
            "image_mode": gt_img.mode,
            "image_size": gt_img.size,
            "file_type": "color" if "_gtFine_color.png" in path else "labelIds" if "_gtFine_labelIds.png" in path else "unknown"
        }
        
        # Analyser selon le type
        if "_gtFine_color.png" in path:
            # Analyser les couleurs uniques dans l'image
            if gt_img.mode != 'RGB':
                gt_img = gt_img.convert('RGB')
            rgb_array = np.array(gt_img)
            
            # Trouver toutes les couleurs uniques
            unique_colors = np.unique(rgb_array.reshape(-1, 3), axis=0)
            debug_info["unique_colors_count"] = len(unique_colors)
            debug_info["unique_colors"] = [tuple(color) for color in unique_colors[:20]]  # Limiter à 20
            
        else:
            # Masque avec IDs
            if gt_img.mode == 'P':
                gt_img = gt_img.convert('L')
            gt_array = np.array(gt_img)
            unique_ids = np.unique(gt_array)
            debug_info["unique_ids"] = unique_ids.tolist()
            debug_info["id_range"] = f"{gt_array.min()} - {gt_array.max()}"
        
        # Convertir en 8 classes
        if "_gtFine_color.png" in path:
            # Pour les masques colorés, utiliser la conversion couleur -> ID -> 8 classes
            rgb_array = np.array(gt_img.convert('RGB'))
            height, width = rgb_array.shape[:2]
            gt_array = np.zeros((height, width), dtype=np.uint8)
            
            # Mapper les couleurs vers les IDs (simplifié pour le debug)
            # On va juste montrer la conversion finale
            gt_array_8classes = convert_gt_to_8_classes(gt_array)
        else:
            gt_array = np.array(gt_img.convert('L') if gt_img.mode == 'P' else gt_img)
            gt_array_8classes = convert_gt_to_8_classes(gt_array)
        
        # Analyser la distribution après conversion
        unique_classes_8 = np.unique(gt_array_8classes)
        class_distribution = {}
        for cls in unique_classes_8:
            count = np.sum(gt_array_8classes == cls)
            class_distribution[int(cls)] = {
                "name": MACRO_CLASS_NAMES[cls],
                "pixel_count": int(count),
                "percentage": float(count / gt_array_8classes.size * 100),
                "color": MACRO_CLASS_COLORS[cls]
            }
        
        debug_info["converted_to_8_classes"] = {
            "unique_classes": unique_classes_8.tolist(),
            "class_distribution": class_distribution
        }
        
        return debug_info
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
