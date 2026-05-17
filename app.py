import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io
import base64
import matplotlib.cm as cm

st.set_page_config(page_title="RetinalAI", page_icon="👁", layout="wide", initial_sidebar_state="collapsed")

C = {
    "bg": "#E8F2ED", "surface": "#F3FAF7", "card": "#FFFFFF", "border": "#A8D5C2",
    "accent": "#127A50", "indigo": "#0C6645", "success": "#127A50",
    "warning": "#B86010", "danger": "#A82C2C", "text": "#081812",
    "muted": "#4A6E5C", "mutedMid": "#1E4A35",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; background-color: {C["bg"]} !important; color: {C["text"]}; }}
.stApp {{ background-color: {C["bg"]} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none; }}
.stButton > button {{ background: {C["accent"]} !important; color: white !important; border: none !important; border-radius: 10px !important; font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; width: 100%; padding: 10px 28px !important; }}
.stButton > button:hover {{ opacity: .88 !important; }}
.stButton > button {{ min-width: 200px !important; }}
.main .block-container {{ max-width: 860px !important; margin: 0 auto !important; padding-left: 24px !important; padding-right: 24px !important; }}
[data-testid="stFileUploader"] {{ background: {C["surface"]}; border: 2px dashed {C["border"]}; border-radius: 16px; padding: 12px; }}
.stProgress > div > div > div {{ background: linear-gradient(90deg, {C["indigo"]}, {C["accent"]}) !important; border-radius: 4px !important; }}
.stProgress > div > div {{ background: #D4EDE4 !important; border-radius: 4px !important; }}
[data-testid="stImage"] img {{ border-radius: 14px; border: 1px solid {C["border"]}; }}
[data-testid="stExpander"] {{ max-width: 860px !important; margin: 0 auto !important; }}
[data-testid="stExpander"] summary {{ background: #0A1F16 !important; color: #FFFFFF !important; border-radius: 12px !important; font-weight: 600 !important; }}
[data-testid="stExpander"] summary:hover {{ background: #0A1F16 !important; color: #FFFFFF !important; }}
[data-testid="stExpander"] summary p {{ color: #FFFFFF !important; }}
[data-testid="stExpander"] summary svg {{ stroke: #FFFFFF !important; fill: #FFFFFF !important; }}
[data-testid="stExpander"] summary:hover svg {{ stroke: #FFFFFF !important; }}
</style>
""", unsafe_allow_html=True)

DISEASE_INFO = {
    "CNV":         {"full": "Choroidal Neovascularization",     "sev": "high",   "desc": "Abnormal blood vessel growth beneath the retina causing fluid leakage and rapid vision loss."},
    "DME":         {"full": "Diabetic Macular Edema",           "sev": "high",   "desc": "Fluid accumulation in the macula due to leaking diabetic retinal blood vessels."},
    "DRUSEN":      {"full": "Drusen Deposits",                  "sev": "medium", "desc": "Tiny yellow lipid deposits under the retina; an early marker of AMD progression."},
    "NORMAL":      {"full": "Healthy Retina",                   "sev": "none",   "desc": "No pathological findings detected."},
    "AMD":         {"full": "Age-related Macular Degeneration", "sev": "high",   "desc": "Progressive degeneration of the central retina, leading to loss of fine-detail vision."},
    "CSR":         {"full": "Central Serous Retinopathy",       "sev": "medium", "desc": "Subretinal fluid accumulation causing blurred or distorted central vision."},
    "MH":          {"full": "Macular Hole",                     "sev": "high",   "desc": "A full-thickness defect at the foveal center severely impacting sharp central vision."},
    "DR":          {"full": "Diabetic Retinopathy",             "sev": "high",   "desc": "Diabetic microvascular damage to retinal capillaries — a leading cause of blindness."},
    "Glaucoma":    {"full": "Glaucoma",                         "sev": "high",   "desc": "Optic nerve damage from elevated intraocular pressure causing irreversible visual field loss."},
    "Cataract":    {"full": "Cataract",                         "sev": "medium", "desc": "Crystalline lens opacity causing progressive blurring, glare, and color desaturation."},
    "Normal":      {"full": "Healthy Eye",                      "sev": "none",   "desc": "No disease detected. The fundus photograph reveals a normal, healthy ocular fundus."},
    "Hypertension":{"full": "Hypertensive Retinopathy",         "sev": "high",   "desc": "Arteriolar changes and retinal hemorrhages from sustained systemic hypertension."},
    "Myopia":      {"full": "Pathologic Myopia",                "sev": "low",    "desc": "Axial elongation causing the focal point to fall anterior to the retina."},
}
CLASSES = {"oct2017": ["CNV","DME","DRUSEN","NORMAL"], "octc8": ["AMD","CSR","MH","DR"], "fundus": ["Glaucoma","Cataract","Normal","Hypertension","Myopia"]}
SEV_COLOR = {"high": C["danger"], "medium": C["warning"], "low": C["accent"], "none": C["success"]}

@st.cache_resource
def load_models():
    models = {}
    try: models["oct2017"] = tf.keras.models.load_model("eye_disease_model.keras")
    except Exception as e: st.warning(f"OCT2017 not loaded: {e}")
    try: models["octc8"] = tf.keras.models.load_model("model_oct_c8")
    except Exception as e: st.warning(f"OCT-C8 not loaded: {e}")
    try: models["fundus"] = tf.keras.models.load_model("fundus_final")
    except Exception as e: st.warning(f"Fundus not loaded: {e}")
    return models

def preprocess_image(img, size, efficientnet=False):
    img = img.convert("RGB").resize((size, size))
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr) if efficientnet else arr / 255.0
    return np.expand_dims(arr, 0)

def make_gradcam(img_array, model):
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name; break
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, tf.keras.layers.Conv2D):
                        last_conv = sub.name; break
            if last_conv: break
        if not last_conv: return None
        feat_model = tf.keras.Model(inputs=model.inputs, outputs=[model.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = feat_model(tf.cast(img_array, tf.float32), training=False)
            tape.watch(conv_out)
            class_channel = preds[:, tf.argmax(preds[0])]
        grads = tape.gradient(class_channel, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(tf.maximum(heatmap, 0))
        mx = tf.math.reduce_max(heatmap)
        if mx > 0: heatmap = heatmap / mx
        return heatmap.numpy()
    except: return None

def overlay_heatmap(orig_img, heatmap):
    h_resized = np.array(Image.fromarray(np.uint8(255 * heatmap)).resize(orig_img.size, Image.LANCZOS))
    colored = np.uint8(255 * cm.get_cmap("jet")(h_resized / 255.0)[:,:,:3])
    orig = np.array(orig_img.convert("RGB"), dtype=np.float32)
    return Image.fromarray(np.uint8(colored * 0.45 + orig * 0.55))

def badge(text, color):
    return f'<span style="display:inline-block;font-family:DM Mono,monospace;font-size:10px;font-weight:600;background:{color}18;border:1px solid {color}40;color:{color};border-radius:5px;padding:3px 8px;margin:2px;">{text}</span>'

def conf_bar(label, value, is_top, color):
    bc = color if is_top else C["border"]
    tc = C["text"] if is_top else C["muted"]
    fw = "600" if is_top else "400"
    return f'''<div style="margin-bottom:12px;"><div style="display:flex;justify-content:space-between;margin-bottom:5px;"><span style="font-size:13px;color:{tc};font-weight:{fw};">{label}</span><span style="font-family:DM Mono,monospace;font-size:12px;color:{color if is_top else C["muted"]};">{value*100:.1f}%</span></div><div style="height:8px;background:#D4EDE4;border-radius:4px;overflow:hidden;"><div style="height:100%;width:{value*100:.1f}%;background:{bc};border-radius:4px;"></div></div></div>'''

def render_navbar(show_reset=False):
    st.markdown(f'''<div style="background:#fff;border-bottom:1px solid {C["border"]};
        box-shadow:0 1px 8px rgba(0,0,0,.09);padding:14px 0;">
        <div style="max-width:860px;margin:0 auto;padding:0 24px;
            display:flex;align-items:center;justify-content:space-between;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="width:32px;height:32px;border-radius:50%;background:#127A50;
                    border:1px solid #0C6645;display:flex;align-items:center;
                    justify-content:center;font-size:16px;">👁</div>
                <span style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;
                    color:{C["text"]};">Retinal<span style="color:{C["accent"]};">AI</span></span>
                <span style="font-family:DM Mono,monospace;font-size:10px;color:{C["accent"]};
                    background:rgba(18,122,80,.1);border:1px solid rgba(18,122,80,.3);
                    border-radius:6px;padding:2px 7px;margin-left:4px;">BETA</span>
            </div>
        </div>
    </div>''', unsafe_allow_html=True)


def render_steps(steps, current):
    html = '<div style="display:flex;align-items:center;margin-bottom:36px;flex-wrap:wrap;gap:4px;">'
    for i, label in enumerate(steps):
        done, active = i < current, i == current
        bg = C["success"] if done else (C["accent"] if active else "#fff")
        bc = C["success"] if done else (C["accent"] if active else C["border"])
        tc = "white" if (done or active) else C["muted"]
        icon = "✓" if done else str(i+1)
        lc = C["text"] if active else C["muted"]
        lw = "500" if active else "400"
        html += f'<div style="display:flex;align-items:center;gap:8px;"><div style="width:32px;height:32px;border-radius:50%;background:{bg};border:2px solid {bc};display:flex;align-items:center;justify-content:center;font-size:13px;font-weight:600;color:{tc};">{icon}</div><span style="font-size:13px;color:{lc};font-weight:{lw};">{label}</span></div>'
        if i < len(steps)-1: html += f'<div style="width:28px;height:1px;background:{C["border"]};margin:0 4px;"></div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

for key, val in [("view","hero"),("mode",None),("group",None),("results",None),("preview",None),("history",[]),("show_history",False),("last_saved_time","")]:
    if key not in st.session_state: st.session_state[key] = val

models = load_models()

# History toggle button in navbar area


# Show history panel
if st.session_state.get("show_history", False):
    _, hpanel, _ = st.columns([1,3,1])
    with hpanel:
        st.markdown(f'''<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;
            padding:20px 24px;box-shadow:0 4px 16px rgba(0,0,0,.08);margin:20px 0;">
            <div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:{C["text"]};
                margin-bottom:16px;border-bottom:1px solid {C["border"]};padding-bottom:12px;">
                🕐 Recent Scans
            </div>
        </div>''', unsafe_allow_html=True)
        if not st.session_state.history:
            st.markdown(f'<div style="text-align:center;color:{C["muted"]};padding:20px;font-size:14px;">No scans yet. Run an analysis first.</div>', unsafe_allow_html=True)
        else:
            for entry in st.session_state.history:
                sev_color = C["success"] if entry["predicted"] in ("NORMAL","Normal") else C["danger"]
                col_img, col_info = st.columns([1,2])
                with col_img:
                    st.image(entry["img"], width=100)
                with col_info:
                    st.markdown(f'''<div style="padding:8px 0;">
                        <div style="font-family:Syne,sans-serif;font-size:16px;font-weight:700;
                            color:{sev_color};margin-bottom:4px;">{entry["predicted"]}</div>
                        <div style="font-size:13px;color:{C["mutedMid"]};margin-bottom:4px;">
                            {entry.get("group","").upper()} Model</div>
                        <div style="font-family:DM Mono,monospace;font-size:12px;color:{C["muted"]};">
                            Confidence: {entry["confidence"]*100:.1f}%</div>
                        <div style="font-family:DM Mono,monospace;font-size:11px;color:{C["muted"]};">
                            {entry["time"]}</div>
                    </div>''', unsafe_allow_html=True)
                st.markdown(f'<hr style="border:none;border-top:1px solid {C["border"]};margin:8px 0;">', unsafe_allow_html=True)
        bc1, bc2, _ = st.columns([1,1,2])
        with bc1:
            if st.button("← Back", use_container_width=True):
                st.session_state.show_history = False
                st.rerun()
        with bc2:
            if st.button("🗑 Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.show_history = False
                st.rerun()
    st.stop()

if st.session_state.view == "hero":
    render_navbar()
    st.markdown(f'''<div style="max-width:960px;margin:0 auto;padding:64px 32px;">
    <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(18,122,80,.1);border:1px solid rgba(18,122,80,.3);border-radius:8px;padding:6px 14px;margin-bottom:28px;">
        <span style="font-family:DM Mono,monospace;font-size:11px;color:{C["accent"]};letter-spacing:.08em;">⚡ AI-POWERED RETINAL ANALYSIS</span>
    </div>
    <h1 style="font-family:sans-serif;font-size:clamp(26px,4vw,42px);font-weight:800;line-height:1.1;margin-bottom:20px;color:{C["text"]};">Detect Eye Diseases<br><span style="color:{C["accent"]};">Instantly</span></h1>
    <p style="font-size:16px;color:{C["mutedMid"]};line-height:1.7;margin-bottom:36px;max-width:520px;">Upload an OCT scan or fundus photograph. Our EfficientNet deep learning model classifies 13 retinal conditions with Grad-CAM visual explanations.</p>
    <div style="display:flex;gap:28px;flex-wrap:wrap;margin-top:8px;">
        <div><div style="font-family:Syne,sans-serif;font-size:28px;font-weight:700;color:{C["accent"]};">97%</div><div style="font-size:11px;color:{C["muted"]};text-transform:uppercase;letter-spacing:.06em;">Avg. accuracy</div></div>
        <div><div style="font-family:Syne,sans-serif;font-size:28px;font-weight:700;color:{C["accent"]};">&lt;2s</div><div style="font-size:11px;color:{C["muted"]};text-transform:uppercase;letter-spacing:.06em;">Analysis time</div></div>
        <div><div style="font-family:Syne,sans-serif;font-size:28px;font-weight:700;color:{C["accent"]};">Free</div><div style="font-size:11px;color:{C["muted"]};text-transform:uppercase;letter-spacing:.06em;">Open source</div></div>
    </div></div>''', unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;margin-top:16px;'>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([0.82, 1, 2])
    with btn_col:
        if st.button("🔍  Begin Analysis"): st.session_state.view = "mode"; st.rerun()
        if st.button("🕐 History", key="hist_nav"):
            st.session_state.show_history = not st.session_state.get("show_history", False)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.view == "mode":
    render_navbar(True)
    st.markdown(f'<div style="max-width:780px;margin:0 auto;padding:48px 32px;">', unsafe_allow_html=True)
    cl2, cm2, cr2 = st.columns([1,3,1])
    with cm2:
        render_steps(["Imaging type","Configuration","Upload & Analyze"], 0)
        st.markdown(f'<h2 style="font-family:Syne,sans-serif;font-size:22px;font-weight:700;margin-bottom:6px;color:{C["text"]};">Select imaging type</h2><p style="color:{C["muted"]};font-size:13px;margin-bottom:20px;">Choose the type of retinal image you want to analyze.</p>', unsafe_allow_html=True)
        oct_tags = "".join([badge(t,C["accent"]) for t in ["CNV","DME","DRUSEN","AMD","CSR","MH","DR","NORMAL"]])
    fund_tags = "".join([badge(t,C["indigo"]) for t in ["Glaucoma","Cataract","Hypertension","Myopia","Normal"]])
    cl, cm, cr = st.columns([1,3,1])
    with cm:
        st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.07);margin-bottom:16px;"><div style="font-size:20px;margin-bottom:12px;">🔬</div><div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:{C["text"]};margin-bottom:6px;">OCT Scan</div><div style="font-size:13px;color:{C["muted"]};line-height:1.6;margin-bottom:12px;">Optical Coherence Tomography — cross-sectional imaging of retinal layers.</div><div>{oct_tags}</div></div>', unsafe_allow_html=True)
        if st.button("Select OCT Scan →"): st.session_state.mode="oct"; st.session_state.view="subgroup"; st.rerun()
        st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.07);margin-top:16px;margin-bottom:16px;"><div style="font-size:20px;margin-bottom:12px;">📷</div><div style="font-family:Syne,sans-serif;font-size:18px;font-weight:700;color:{C["text"]};margin-bottom:6px;">Fundus Photo</div><div style="font-size:13px;color:{C["muted"]};line-height:1.6;margin-bottom:12px;">Photography of the interior surface of the eye, including the optic disc and macula.</div><div>{fund_tags}</div></div>', unsafe_allow_html=True)
        if st.button("Select Fundus Photo →"): st.session_state.mode="fundus"; st.session_state.group="fundus"; st.session_state.view="upload"; st.rerun()
        if st.button("← Back to Home"): st.session_state.view="hero"; st.rerun()

elif st.session_state.view == "subgroup":
    render_navbar(True)
    st.markdown(f'<div style="max-width:780px;margin:0 auto;padding:48px 32px;">', unsafe_allow_html=True)
    cl3,cm3,cr3 = st.columns([1,3,1])
    with cm3:
        render_steps(["Imaging type","Configuration","Upload & Analyze"], 1)
        st.markdown(f'<h2 style="font-family:Syne,sans-serif;font-size:22px;font-weight:700;margin-bottom:6px;color:{C["text"]};">🔬 Select disease group</h2><p style="color:{C["muted"]};font-size:13px;margin-bottom:20px;">Two separate OCT models cover different disease sets.</p>', unsafe_allow_html=True)
    _, cm_sub, _ = st.columns([1,3,1])
    with cm_sub:
        t1 = "".join([badge(c,C["accent"]) for c in ["CNV","DME","DRUSEN","NORMAL"]])
        st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.07);margin-bottom:16px;"><div style="font-size:11px;color:{C["muted"]};letter-spacing:.07em;margin-bottom:6px;">MODEL 1 · OCT2017</div><div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;color:{C["text"]};margin-bottom:12px;">Kermy Dataset</div><div>{t1}</div></div>', unsafe_allow_html=True)
        if st.button("Use OCT2017 Model →"): st.session_state.group="oct2017"; st.session_state.view="upload"; st.rerun()
        t2 = "".join([badge(c,C["indigo"]) for c in ["AMD","CSR","MH","DR"]])
        st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;padding:24px;box-shadow:0 2px 8px rgba(0,0,0,.07);margin-top:16px;margin-bottom:16px;"><div style="font-size:11px;color:{C["muted"]};letter-spacing:.07em;margin-bottom:6px;">MODEL 2 · OCT-C8</div><div style="font-family:Syne,sans-serif;font-size:17px;font-weight:700;color:{C["text"]};margin-bottom:12px;">Extended OCT</div><div>{t2}</div></div>', unsafe_allow_html=True)
        if st.button("Use OCT-C8 Model →"): st.session_state.group="octc8"; st.session_state.view="upload"; st.rerun()
        if st.button("← Back"): st.session_state.view="mode"; st.rerun()

elif st.session_state.view == "upload":
    render_navbar(True)
    st.markdown(f'<div style="max-width:780px;margin:0 auto;padding:48px 32px;">', unsafe_allow_html=True)
    _, cup, _ = st.columns([1,3,1])
    with cup:
        group = st.session_state.group
        classes = CLASSES[group]
        mode_label = "OCT Scan" if st.session_state.mode=="oct" else "Fundus Photo"
        st.markdown(f'<h2 style="font-family:Syne,sans-serif;font-size:22px;font-weight:700;margin-bottom:6px;color:{C["text"]};">📤 Upload retinal image</h2><p style="color:{C["muted"]};font-size:13px;margin-bottom:12px;">{mode_label} — detecting: {" · ".join(classes)}</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        if st.button("← Back"): st.session_state.view="subgroup" if st.session_state.mode=="oct" else "mode"; st.rerun()
    if uploaded:
        img = Image.open(uploaded)
        _, col1, col2, _ = st.columns([1,0.8,1,0.8])
        with col1: st.image(img, width=240, caption="Uploaded scan")
        with col2:
            tags = "".join([badge(c,C["accent"]) for c in classes])
            st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:14px;padding:16px;margin-bottom:12px;"><div style="font-size:11px;color:{C["muted"]};margin-bottom:3px;">Scan type</div><div style="font-size:13px;font-weight:500;color:{C["text"]};">{mode_label}</div></div><div style="background:#fff;border:1px solid {C["border"]};border-radius:12px;padding:12px 14px;margin-top:10px;"><div style="font-size:10px;color:{C["muted"]};margin-bottom:6px;">Detecting conditions</div><div>{tags}</div></div>', unsafe_allow_html=True)
        _, ba, bb, _ = st.columns([1,0.8,0.8,1])
        with ba:
            if st.button("🧠 Run Analysis", use_container_width=True): st.session_state.preview=img; st.session_state.view="results"; st.rerun()
        with bb:
            if st.button("← Back", use_container_width=True): st.session_state.view="subgroup" if st.session_state.mode=="oct" else "mode"; st.rerun()

elif st.session_state.view == "results":
    render_navbar(True)
    st.markdown('<div style="max-width:960px;margin:0 auto;padding:48px 32px;">', unsafe_allow_html=True)
    group = st.session_state.group
    classes = CLASSES[group]
    img = st.session_state.preview
    # Save to history
    import copy, datetime
    if len(st.session_state.history) == 0 or st.session_state.history[-1].get("img") is not st.session_state.preview:
        pass  # will save after prediction

    with st.spinner("Running model inference + Grad-CAM…"):
        model = models.get(group)
        if model is None: st.error(f"Model not loaded."); st.stop()
        arr = preprocess_image(img, 224, efficientnet=(group!="oct2017")) if group!="fundus" else preprocess_image(img, 300, efficientnet=True)
        preds = model.predict(arr, verbose=0)[0]
        predicted = classes[int(np.argmax(preds))]
        scores = {c: float(preds[i]) for i,c in enumerate(classes)}
        confidence = scores[predicted]
        heatmap = make_gradcam(arr, model)
        gradcam_img = overlay_heatmap(img, heatmap) if heatmap is not None else None

    # Add to history
    import datetime
    history_entry = {
        "img": img,
        "predicted": classes[int(np.argmax(preds))],
        "confidence": float(np.max(preds)),
        "group": group,
        "time": datetime.datetime.now().strftime("%H:%M")
    }
    if len(st.session_state.history) == 0 or st.session_state.history[-1]["time"] != history_entry["time"]:
        st.session_state.history.insert(0, history_entry)
        if len(st.session_state.history) > 5:
            st.session_state.history = st.session_state.history[:5]
    is_healthy = predicted in ("NORMAL","Normal")
    is_low_conf = confidence < 0.75
    is_unrecognized = confidence < 0.45
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2_diff = sorted_scores[0][1] - sorted_scores[1][1]
    is_ambiguous = top2_diff < 0.15 and confidence < 0.85
    ambiguous_cands = [(c,v) for c,v in sorted_scores if v > 0.15]
    info = DISEASE_INFO.get(predicted, {"full":predicted,"sev":"high","desc":""})
    pred_color = C["success"] if is_healthy else (C["warning"] if is_low_conf else C["danger"])
    _, hcol, _ = st.columns([0.3,3,0.3])
    with hcol:
        st.markdown(f'<div style="margin-bottom:24px;"><div style="font-family:DM Mono,monospace;font-size:11px;color:{C["accent"]};letter-spacing:.1em;margin-bottom:6px;">ANALYSIS COMPLETE</div><h2 style="font-family:Syne,sans-serif;font-size:24px;font-weight:700;color:{C["text"]};">Diagnostic Result</h2></div>', unsafe_allow_html=True)
    _, col_left, col_right, _ = st.columns([0.3,1,1.4,0.3])
    with col_left:
        st.image(img, use_container_width=True, caption="Uploaded scan")
        if gradcam_img:
            st.markdown(f'<div style="text-align:center;margin-top:8px;"><div style="font-family:DM Mono,monospace;font-size:11px;color:{C["muted"]};letter-spacing:.06em;margin-bottom:6px;">GRAD-CAM HEATMAP</div></div>', unsafe_allow_html=True)
            st.image(gradcam_img, use_container_width=True)
            st.markdown(f'<div style="font-size:11px;color:{C["muted"]};text-align:center;margin-top:4px;">Red/yellow = regions the model focused on</div>', unsafe_allow_html=True)
    with col_right:
        icon = "✅" if is_healthy else "⚠️"
        status = "HEALTHY · NO DISEASE DETECTED" if is_healthy else ("LOW CONFIDENCE · UNCERTAIN" if is_low_conf else "CONDITION DETECTED")
        desc_html = f'<div style="margin-top:12px;padding-top:12px;border-top:1px solid {pred_color}25;font-size:13px;color:{C["mutedMid"]};line-height:1.6;">{info["desc"]}</div>' if info["sev"]!="none" else ""
        st.markdown(f'<div style="background:{pred_color}12;border:1px solid {pred_color}50;border-radius:14px;padding:20px 22px;margin-bottom:14px;"><div style="display:flex;align-items:flex-start;justify-content:space-between;gap:12px;"><div><div style="font-size:11px;color:{pred_color};letter-spacing:.08em;margin-bottom:6px;">{icon} {status}</div><div style="font-family:Syne,sans-serif;font-size:24px;font-weight:700;color:{C["text"]};margin-bottom:4px;">{predicted}</div><div style="font-size:13px;color:{C["mutedMid"]};">{info["full"]}</div></div><div style="text-align:right;flex-shrink:0;"><div style="font-family:Syne,sans-serif;font-size:32px;font-weight:800;color:{pred_color};">{confidence*100:.1f}%</div><div style="font-size:11px;color:{C["muted"]};">confidence</div></div></div>{desc_html}</div>', unsafe_allow_html=True)
        if is_unrecognized:
            st.markdown(f'<div style="background:rgba(168,44,44,.06);border:1px solid rgba(168,44,44,.25);border-radius:10px;padding:12px 16px;margin-bottom:14px;"><span style="font-size:13px;color:{C["danger"]};line-height:1.5;">⚠️ <strong>Image not recognized.</strong> This image does not clearly match any known category. Please consult a specialist.</span></div>', unsafe_allow_html=True)
        elif is_ambiguous:
            cand_html = "".join([f'<span style="font-size:11px;background:rgba(184,96,16,.08);border:1px solid rgba(184,96,16,.2);color:{C["warning"]};border-radius:5px;padding:3px 8px;margin:2px;display:inline-block;">{c} — {v*100:.1f}%</span>' for c,v in ambiguous_cands])
            st.markdown(f'<div style="background:rgba(184,96,16,.06);border:1px solid rgba(184,96,16,.25);border-radius:10px;padding:12px 16px;margin-bottom:14px;"><div style="font-size:12px;color:{C["warning"]};font-weight:600;margin-bottom:8px;">🔀 Multiple possible conditions detected</div><div style="margin-bottom:8px;">{cand_html}</div><div style="font-size:11px;color:{C["muted"]};">Professional evaluation strongly recommended.</div></div>', unsafe_allow_html=True)
        elif is_low_conf:
            st.markdown(f'<div style="background:rgba(184,96,16,.06);border:1px solid rgba(184,96,16,.25);border-radius:10px;padding:12px 16px;margin-bottom:14px;"><span style="font-size:12px;color:{C["warning"]};">⚠️ Confidence below 75%. Please consult a qualified ophthalmologist.</span></div>', unsafe_allow_html=True)
        bars = "".join([conf_bar(c, scores[c], c==predicted, pred_color) for c,_ in sorted_scores])
        st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:16px;padding:18px 20px;box-shadow:0 2px 8px rgba(0,0,0,.07);"><div style="font-family:DM Mono,monospace;font-size:12px;color:{C["muted"]};letter-spacing:.06em;margin-bottom:16px;">CONFIDENCE SCORES</div>{bars}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="margin-top:24px;background:rgba(168,44,44,.05);border:1px solid rgba(168,44,44,.15);border-radius:10px;padding:14px 20px;text-align:center;"><p style="font-size:13px;color:{C["danger"]};line-height:1.5;margin:0;font-weight:700;">⚠️ This tool is for educational purposes only. Always consult a licensed ophthalmologist.</p></div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    _, nb, _ = st.columns([0.3,1,0.3])
    with nb:
        if st.button("← New Analysis", use_container_width=True):
            for k in ["view","mode","group","results","preview"]: st.session_state[k] = "hero" if k=="view" else None
        st.rerun()

if st.session_state.view != "results":
    st.markdown(f'<div style="max-width:900px;margin:0 auto;padding:0 32px 64px;">', unsafe_allow_html=True)
    st.markdown('<div style="max-width:860px;margin:0 auto;">', unsafe_allow_html=True)

    with st.expander("📋  Disease reference — 13 conditions"):
        for label, keys in [("OCT SCAN", list(DISEASE_INFO.keys())[:8]), ("FUNDUS PHOTO", list(DISEASE_INFO.keys())[8:])]:
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{C["accent"]};letter-spacing:.08em;margin:12px 0;">{label}</div>', unsafe_allow_html=True)
            cols = st.columns(3)
            for i,k in enumerate(keys):
                info = DISEASE_INFO[k]; color = SEV_COLOR[info["sev"]]
                with cols[i%3]:
                    st.markdown(f'<div style="background:#fff;border:1px solid {C["border"]};border-radius:14px;padding:14px 16px;margin-bottom:12px;"><div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;"><div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div><span style="font-family:DM Mono,monospace;font-size:14px;font-weight:600;color:{color};">{k}</span></div><div style="font-size:12px;color:{C["mutedMid"]};line-height:1.6;">{info["desc"]}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
