import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# ── Model paths ──────────────────────────────────────────────────────────────
MODEL_OCT2017_PATH = "eye_disease_model.keras"
MODEL_OCTC8_PATH   = "model_oct_c8"
MODEL_FUNDUS_PATH  = "fundus_final"

# ── Class definitions ─────────────────────────────────────────────────────────
CLASSES_OCT2017 = ["CNV", "DME", "DRUSEN", "NORMAL"]
CLASSES_OCTC8   = ["AMD", "CSR", "MH", "DR"]
CLASSES_FUNDUS  = ["Glaucoma", "Cataract", "Normal", "Hypertension", "Myopia"]

CLASS_INFO = {
    # OCT
    "CNV":         "Choroidal Neovascularization — abnormal blood vessel growth beneath the retina.",
    "DME":         "Diabetic Macular Edema — fluid buildup in the macula due to diabetes.",
    "DRUSEN":      "Drusen — small yellow deposits under the retina, an early sign of AMD.",
    "NORMAL":      "No disease detected — healthy retina.",
    "AMD":         "Age-related Macular Degeneration — progressive damage to the central retina.",
    "CSR":         "Central Serous Retinopathy — fluid under the retina causing vision distortion.",
    "MH":          "Macular Hole — a small break in the macula affecting central vision.",
    "DR":          "Diabetic Retinopathy — damage to retinal blood vessels caused by diabetes.",
    # Fundus
    "Glaucoma":    "Glaucoma — optic nerve damage usually caused by elevated eye pressure.",
    "Cataract":    "Cataract — clouding of the eye lens causing blurred vision.",
    "Normal":      "No disease detected — healthy eye.",
    "Hypertension":"Hypertensive Retinopathy — retinal damage caused by high blood pressure.",
    "Myopia":      "Myopia (Nearsightedness) — light focuses in front of the retina instead of on it.",
}

CONFIDENCE_THRESHOLD = 0.75

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Eye Disease Detector", page_icon="👁️", layout="centered")
st.title("👁️ Eye Disease Detection")
st.caption("AI-powered detection of retinal diseases from OCT scans and Fundus photographs.")


# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    oct2017 = tf.keras.models.load_model(MODEL_OCT2017_PATH)
    octc8   = tf.keras.models.load_model(MODEL_OCTC8_PATH)
    fundus  = tf.keras.models.load_model(MODEL_FUNDUS_PATH)
    return oct2017, octc8, fundus


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img: Image.Image, size: int = 224, efficientnet: bool = False) -> np.ndarray:
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32)
    if efficientnet:
        arr = efficientnet_preprocess(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def get_last_conv_layer(model):
    # Search top-level layers first
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    # Search inside nested models (e.g. EfficientNet backbone)
    for layer in reversed(model.layers):
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, tf.keras.layers.Conv2D):
                    return sublayer.name
    return None


def get_layer_recursive(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        if hasattr(layer, "layers"):
            for sublayer in layer.layers:
                if sublayer.name == layer_name:
                    return sublayer
    return None


def _compute_heatmap(conv_outputs, grads):
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    img_tensor = tf.cast(img_array, tf.float32)

    # Strategy 1: nested backbone sub-model (OCT-C8 — EfficientNetB0 as sub-model)
    backbone = None
    post_backbone = []
    for i, layer in enumerate(model.layers):
        if hasattr(layer, "layers") and len(layer.layers) > 10:
            backbone = layer
            post_backbone = [l for l in model.layers[i + 1:]
                             if not isinstance(l, tf.keras.layers.InputLayer)]
            break

    if backbone is not None and post_backbone:
        try:
            with tf.GradientTape() as tape:
                conv_outputs = backbone(img_tensor, training=False)
                tape.watch(conv_outputs)
                x = conv_outputs
                for layer in post_backbone:
                    x = layer(x)
                pred_index = tf.argmax(x[0])
                class_channel = x[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is not None:
                return _compute_heatmap(conv_outputs, grads)
        except Exception:
            pass

    # Strategy 2: inlined Functional model (Fundus — EfficientNetB3 with input_tensor)
    try:
        target_name = None
        for layer in model.layers:
            if layer.name == "top_conv":
                target_name = "top_conv"
                break
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_name = layer.name
        if target_name:
            feat_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(target_name).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = feat_model(img_tensor, training=False)
                tape.watch(conv_outputs)
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is not None:
                return _compute_heatmap(conv_outputs, grads)
    except Exception:
        pass

    # Strategy 3: simple Sequential layer-by-layer (OCT2017)
    try:
        pre_layers, post_layers = [], []
        found = False
        for layer in model.layers:
            if not found:
                pre_layers.append(layer)
                if layer.name == last_conv_layer_name:
                    found = True
            else:
                post_layers.append(layer)
        if found:
            with tf.GradientTape() as tape:
                x = img_tensor
                for layer in pre_layers:
                    x = layer(x)
                conv_outputs = x
                tape.watch(conv_outputs)
                for layer in post_layers:
                    x = layer(x)
                pred_index = tf.argmax(x[0])
                class_channel = x[:, pred_index]
            grads = tape.gradient(class_channel, conv_outputs)
            if grads is not None:
                return _compute_heatmap(conv_outputs, grads)
    except Exception:
        pass

    return None


def overlay_gradcam(original_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(original_img.size, Image.LANCZOS)
    )
    colormap = cm.get_cmap("jet")
    heatmap_colored = np.uint8(255 * colormap(heatmap_resized / 255.0)[:, :, :3])
    original_arr = np.array(original_img.convert("RGB"), dtype=np.float32)
    superimposed = heatmap_colored * alpha + original_arr * (1 - alpha)
    return Image.fromarray(np.uint8(superimposed))


# ── Results display ───────────────────────────────────────────────────────────
def show_results(img, preds, class_names, model, last_conv, size=224, efficientnet=False):
    predicted_idx = int(np.argmax(preds))
    predicted_class = class_names[predicted_idx]
    confidence = float(preds[predicted_idx])

    try:
        arr = preprocess(img, size=size, efficientnet=efficientnet)
        heatmap = make_gradcam_heatmap(arr, model, last_conv) if last_conv else None
    except Exception:
        heatmap = None

    if heatmap is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlay_gradcam(img, heatmap), caption="Grad-CAM Heatmap", use_container_width=True)
        st.caption("Red/yellow = regions the model focused on most.")
    else:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    st.divider()

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(f"⚠️ Low confidence ({confidence*100:.1f}%) — result may be unreliable. Please consult a specialist.")
    else:
        if predicted_class in ("NORMAL", "Normal"):
            st.success(f"✅ Prediction: **{predicted_class}** ({confidence*100:.1f}% confidence)")
        else:
            st.error(f"🔴 Prediction: **{predicted_class}** ({confidence*100:.1f}% confidence)")

    st.markdown(f"_{CLASS_INFO[predicted_class]}_")

    st.divider()
    st.subheader("Confidence Scores")
    for i in np.argsort(preds)[::-1]:
        st.markdown(f"**{class_names[i]}**")
        st.progress(float(preds[i]), text=f"{preds[i]*100:.1f}%")


# ── Load models ───────────────────────────────────────────────────────────────
model_oct2017, model_octc8, model_fundus = load_all_models()
last_conv_oct2017 = get_last_conv_layer(model_oct2017)
last_conv_octc8   = get_last_conv_layer(model_octc8)
last_conv_fundus  = get_last_conv_layer(model_fundus)

# ── Step 1: Imaging type ──────────────────────────────────────────────────────
st.divider()
st.subheader("Step 1 — Select Imaging Type")
col1, col2 = st.columns(2)
with col1:
    btn_oct = st.button("🔬 OCT Scan", use_container_width=True,
                        help="Optical Coherence Tomography — cross-sectional retinal scans")
with col2:
    btn_fundus = st.button("📷 Fundus Photo", use_container_width=True,
                           help="Fundus photography — images of the back of the eye")

if btn_oct:
    st.session_state.imaging_type = "oct"
    st.session_state.selected_group = None
if btn_fundus:
    st.session_state.imaging_type = "fundus"
    st.session_state.selected_group = None

if "imaging_type" not in st.session_state:
    st.session_state.imaging_type = None
if "selected_group" not in st.session_state:
    st.session_state.selected_group = None

# ── Step 2: OCT subgroup or Fundus upload ────────────────────────────────────
if st.session_state.imaging_type == "oct":
    st.info("Selected: **OCT Scan**")
    st.subheader("Step 2 — Select Disease Group")
    col1, col2 = st.columns(2)
    with col1:
        g1 = st.button("🔵 CNV · DME · DRUSEN · NORMAL", use_container_width=True)
    with col2:
        g2 = st.button("🟣 AMD · CSR · MH · DR", use_container_width=True)

    if g1:
        st.session_state.selected_group = "oct2017"
    if g2:
        st.session_state.selected_group = "octc8"

    if st.session_state.selected_group:
        label = "CNV · DME · DRUSEN · NORMAL" if st.session_state.selected_group == "oct2017" else "AMD · CSR · MH · DR"
        st.info(f"Selected: **{label}**")
        uploaded = st.file_uploader("Upload a Retinal OCT image", type=["jpg", "jpeg", "png"])
        if uploaded:
            img = Image.open(uploaded)
            with st.spinner("Analyzing..."):
                if st.session_state.selected_group == "oct2017":
                    arr = preprocess(img, size=224, efficientnet=False)
                    preds = model_oct2017.predict(arr, verbose=0)[0]
                    show_results(img, preds, CLASSES_OCT2017, model_oct2017, last_conv_oct2017, size=224, efficientnet=False)
                else:
                    arr = preprocess(img, size=224, efficientnet=True)
                    preds = model_octc8.predict(arr, verbose=0)[0]
                    show_results(img, preds, CLASSES_OCTC8, model_octc8, last_conv_octc8, size=224, efficientnet=True)

elif st.session_state.imaging_type == "fundus":
    st.info("Selected: **Fundus Photo**")
    st.subheader("Step 2 — Upload Image")
    uploaded = st.file_uploader("Upload a Fundus photograph", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded)
        with st.spinner("Analyzing..."):
            arr = preprocess(img, size=300, efficientnet=True)
            preds = model_fundus.predict(arr, verbose=0)[0]
            show_results(img, preds, CLASSES_FUNDUS, model_fundus, last_conv_fundus, size=300, efficientnet=True)

else:
    st.info("👆 Select an imaging type above to begin.")

st.divider()
st.caption("⚠️ This tool is for educational purposes only and is not a substitute for professional medical advice.")
