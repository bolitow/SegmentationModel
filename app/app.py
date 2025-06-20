import streamlit as st
import requests, io
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Démo Segmentation", layout="wide")
st.title("Segmentation Cityscapes – U-Net / MobileNetV2")

# Sélection du modèle
try:
    models = requests.get(f"{API_URL}/models", timeout=10).json()
    current_model = requests.get(f"{API_URL}/current_model", timeout=10).json()["model"]
except Exception as e:
    st.error(f"Impossible de récupérer les modèles : {e}")
    st.stop()

selected_model = st.selectbox(
    "Choisissez un modèle :", 
    models, 
    index=models.index(current_model) if current_model in models else 0
)

if selected_model != current_model:
    try:
        requests.post(f"{API_URL}/set_model/{selected_model}", timeout=10)
        st.success(f"Modèle {selected_model} sélectionné")
        st.rerun()
    except Exception as e:
        st.error(f"Erreur lors du changement de modèle : {e}")

try:
    ids = requests.get(f"{API_URL}/images", timeout=10).json()
except Exception as e:
    st.error(f"Impossible de contacter l'API : {e}")
    st.stop()

image_id = st.selectbox("Choisissez une image :", sorted(ids))

col1, col2 = st.columns(2)

with col1:
    predict_only = st.button("Lancer la prédiction", use_container_width=True)

with col2:
    predict_with_metrics = st.button("Prédiction + Métriques", use_container_width=True)

if predict_only or predict_with_metrics:
    col1, col2, col3 = st.columns(3)

    # Image brute
    img_bytes = requests.get(f"{API_URL}/images/{image_id}").content
    col1.subheader("Image")
    col1.image(img_bytes)

    # Masque Ground-Truth
    gt_bytes = requests.get(f"{API_URL}/masks/{image_id}").content
    col2.subheader("Masque réel")
    col2.image(gt_bytes)

    # Masque prédit
    files = {
        "file": (f"{image_id}.png", img_bytes, "image/png"),
    }
    
    if predict_with_metrics:
        try:
            response = requests.post(f"{API_URL}/predict_with_metrics", files=files)
            result = response.json()
            
            # Récupérer l'image de prédiction pour l'affichage
            pred_response = requests.post(f"{API_URL}/predict", files=files)
            pred_bytes = pred_response.content
            
            col3.subheader("Masque prédit")
            col3.image(pred_bytes)
            
            # Afficher les métriques
            st.markdown("---")
            st.subheader("📊 Métriques de performance")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Modèle utilisé", result["model_used"])
            
            with metrics_col2:
                st.metric("Temps de prédiction", f"{result['prediction_time']:.3f}s")
            
            with metrics_col3:
                if "metrics" in result and "error" not in result["metrics"]:
                    st.metric("Précision", f"{result['metrics']['accuracy']:.3f}")
                    st.metric("IoU (Jaccard)", f"{result['metrics']['iou']:.3f}")
                else:
                    st.warning("Métriques non disponibles (pas de ground truth)")
                    
        except Exception as e:
            st.error(f"Erreur lors de la prédiction avec métriques : {e}")
    else:
        pred_bytes = requests.post(f"{API_URL}/predict", files=files).content
        col3.subheader("Masque prédit")
        col3.image(pred_bytes)
        
        # Afficher juste le modèle utilisé et le temps
        model_info = requests.get(f"{API_URL}/current_model").json()
        st.info(f"Modèle utilisé : {model_info['model']}")