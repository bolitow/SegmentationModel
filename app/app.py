import streamlit as st
import requests, io
from PIL import Image
import numpy as np
import time

# Configuration de l'API
API_URL = "https://segmentation-model-api-czfuckgqcpfacdct.francecentral-01.azurewebsites.net"

# Configuration de la page
st.set_page_config(page_title="Démo Segmentation - 8 Classes", layout="wide")
st.title("Segmentation Cityscapes – U-Net / MobileNetV2")
st.markdown("### Système de segmentation avec réduction à 8 macro-classes")


# Fonction helper pour les requêtes avec gestion d'erreur
def safe_request(method, url, **kwargs):
    """Effectue une requête HTTP avec gestion d'erreur et retry"""
    kwargs.setdefault('timeout', 10)
    kwargs.setdefault('verify', False)

    try:
        if method == 'get':
            response = requests.get(url, **kwargs)
        elif method == 'post':
            response = requests.post(url, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion : {e}")
        return None


# Afficher les informations sur les classes dans la sidebar
with st.sidebar:
    st.header("Informations sur les classes")

    # Récupérer les infos sur les classes
    classes_response = safe_request('get', f"{API_URL}/debug/classes")
    if classes_response:
        classes_info = classes_response.json()

        st.write(f"**Nombre de classes :** {classes_info['num_classes']}")

        # Afficher chaque classe avec sa couleur
        st.write("**Classes et couleurs :**")
        for class_info in classes_info['macro_classes']:
            color_hex = class_info['color_hex']
            # Créer un petit carré de couleur en HTML
            color_box = f'<div style="display:inline-block;width:20px;height:20px;background-color:{color_hex};margin-right:10px;vertical-align:middle;"></div>'
            st.markdown(f"{color_box} {class_info['index']}: {class_info['name']}", unsafe_allow_html=True)

        # Bouton pour afficher le mapping complet
        if st.button("Voir le mapping Cityscapes → Macro-classes"):
            with st.expander("Mapping détaillé"):
                mapping = classes_info.get('cityscapes_to_macro_mapping', {})
                for cs_id, info in mapping.items():
                    st.write(f"ID {cs_id} → Classe {info['macro_class_id']} ({info['macro_class_name']})")

# Section principale : Sélection du modèle
st.header("1. Sélection du modèle")
col1, col2 = st.columns([3, 1])

with col1:
    # Récupération des modèles
    models_response = safe_request('get', f"{API_URL}/models")
    current_model_response = safe_request('get', f"{API_URL}/current_model")

    if not models_response or not current_model_response:
        st.error("Impossible de récupérer les modèles")
        st.stop()

    models = models_response.json()
    current_model_info = current_model_response.json()
    current_model = current_model_info.get("model", None)

    # Afficher les infos du modèle actuel
    if current_model:
        st.info(f"**Modèle actif :** {current_model}")
        if "num_classes" in current_model_info:
            expected = current_model_info.get("expected_classes", 8)
            actual = current_model_info.get("num_classes", "?")
            if actual != expected:
                st.warning(f"⚠️ Le modèle prédit {actual} classes au lieu de {expected}")

    # Sélection du modèle
    selected_model = st.selectbox(
        "Choisissez un modèle :",
        models,
        index=models.index(current_model) if current_model in models else 0
    )

with col2:
    # Bouton pour changer de modèle
    if st.button("Changer le modèle", disabled=(selected_model == current_model)):
        change_response = safe_request('post', f"{API_URL}/set_model/{selected_model}")
        if change_response:
            st.success(f"Modèle {selected_model} sélectionné")
            st.rerun()

# Section 2 : Sélection de l'image
st.header("2. Sélection de l'image")

# Récupération de la liste des images
images_response = safe_request('get', f"{API_URL}/images")
if not images_response:
    st.error("Impossible de récupérer la liste des images")
    st.stop()

ids = images_response.json()
image_id = st.selectbox("Choisissez une image :", sorted(ids))

# Section 3 : Prédiction
st.header("3. Prédiction et résultats")

col1, col2 = st.columns([1, 1])
with col1:
    predict_button = st.button("🚀 Lancer la prédiction", use_container_width=True, type="primary")
with col2:
    predict_metrics_button = st.button("📊 Prédiction avec métriques", use_container_width=True)

# Prédiction simple
if predict_button:
    with st.spinner("Prédiction en cours..."):
        start_time = time.time()

        # Créer trois colonnes pour l'affichage
        col1, col2, col3 = st.columns(3)

        # Image originale
        with col1:
            st.subheader("Image originale")
            img_response = safe_request('get', f"{API_URL}/images/{image_id}")
            if img_response:
                img_bytes = img_response.content
                st.image(img_bytes, use_container_width=True)

                # Afficher les dimensions de l'image
                img_pil = Image.open(io.BytesIO(img_bytes))
                st.caption(f"Dimensions : {img_pil.size[0]}×{img_pil.size[1]}")

        # Masque Ground Truth
        with col2:
            st.subheader("Masque réel (8 classes)")
            gt_response = safe_request('get', f"{API_URL}/masks/{image_id}")
            if gt_response:
                gt_bytes = gt_response.content
                st.image(gt_bytes, use_container_width=True)

                # Analyser le masque GT
                gt_pil = Image.open(io.BytesIO(gt_bytes))
                if gt_pil.mode == 'P':
                    gt_array = np.array(gt_pil)
                    unique_classes = np.unique(gt_array)
                    st.caption(f"Classes présentes : {list(unique_classes)}")

        # Masque prédit
        with col3:
            st.subheader("Masque prédit")
            if img_response:
                files = {"file": (f"{image_id}.png", img_bytes, "image/png")}
                pred_response = safe_request('post', f"{API_URL}/predict", files=files)

                if pred_response:
                    pred_bytes = pred_response.content
                    st.image(pred_bytes, use_container_width=True)

                    # Temps de prédiction
                    elapsed = time.time() - start_time
                    st.caption(f"Temps : {elapsed:.2f}s")

                    # Analyser le masque prédit
                    pred_pil = Image.open(io.BytesIO(pred_bytes))
                    if pred_pil.mode == 'P':
                        pred_array = np.array(pred_pil)
                        unique_classes = np.unique(pred_array)
                        st.caption(f"Classes prédites : {list(unique_classes)}")

# Prédiction avec métriques détaillées
if predict_metrics_button:
    with st.spinner("Prédiction et calcul des métriques en cours..."):
        # Récupérer l'image
        img_response = safe_request('get', f"{API_URL}/images/{image_id}")
        if img_response:
            img_bytes = img_response.content
            files = {"file": (f"{image_id}.png", img_bytes, "image/png")}

            # Appeler l'endpoint avec métriques
            metrics_response = safe_request('post', f"{API_URL}/predict_with_metrics", files=files)

            if metrics_response:
                results = metrics_response.json()

                # Afficher les résultats dans des colonnes
                col1, col2, col3 = st.columns([1, 1, 1])

                with col1:
                    st.metric("Modèle utilisé", results['model_used'])
                    st.metric("Temps de prédiction", f"{results['prediction_time']:.3f}s")

                with col2:
                    if 'metrics' in results and 'accuracy' in results['metrics']:
                        st.metric("Accuracy", f"{results['metrics']['accuracy']:.2%}")
                        st.metric("IoU moyen", f"{results['metrics']['mean_iou']:.2%}")

                with col3:
                    if 'debug_info' in results:
                        st.write("**Distribution des classes prédites :**")
                        for cls_id, info in results['debug_info']['class_distribution'].items():
                            st.write(f"- {info['name']}: {info['percentage']:.1f}%")

                # Afficher les IoU par classe si disponibles
                if 'metrics' in results and 'iou_per_class' in results['metrics']:
                    st.subheader("IoU par classe")
                    iou_data = results['metrics']['iou_per_class']

                    # Créer un tableau pour l'affichage
                    col1, col2 = st.columns(2)
                    items = list(iou_data.items())
                    half = len(items) // 2

                    with col1:
                        for class_name, iou in items[:half]:
                            st.write(f"**{class_name}:** {iou:.3f}")

                    with col2:
                        for class_name, iou in items[half:]:
                            st.write(f"**{class_name}:** {iou:.3f}")

# Section d'upload personnalisé
st.divider()
st.header("4. Tester avec votre propre image")

uploaded_file = st.file_uploader("Choisir une image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image uploadée")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        st.caption(f"Dimensions : {image.size[0]}×{image.size[1]}")

    with col2:
        if st.button("Prédire sur l'image uploadée", use_container_width=True):
            with st.spinner("Prédiction en cours..."):
                # Réinitialiser la position du fichier
                uploaded_file.seek(0)

                # Faire la prédiction avec debug
                files = {"file": (uploaded_file.name, uploaded_file, "image/png")}
                debug_response = safe_request('post', f"{API_URL}/predict_with_debug", files=files)

                if debug_response:
                    debug_results = debug_response.json()

                    # Décoder et afficher le masque
                    import base64

                    mask_bytes = base64.b64decode(debug_results['mask_base64'])
                    st.subheader("Masque prédit")
                    st.image(mask_bytes, use_container_width=True)

                    # Afficher les infos de debug
                    with st.expander("Informations détaillées"):
                        st.write("**Temps de prédiction :**", f"{debug_results['prediction_time']:.3f}s")
                        st.write("**Distribution des classes :**")
                        for cls_id, info in debug_results['prediction_info']['class_distribution'].items():
                            st.write(f"- {info['name']}: {info['percentage']:.1f}%")

# Footer avec informations système
st.divider()
st.caption(f"API : {API_URL} | Système de segmentation avec 8 macro-classes")