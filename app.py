import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import json
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="centered"
)

# ── Class Names ───────────────────────────────────────────────
CLASS_NAMES_FALLBACK = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "PlantVillage",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

def load_class_names():
    path = Path("models/class_names.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return CLASS_NAMES_FALLBACK

CLASS_NAMES = load_class_names()
NUM_CLASSES = len(CLASS_NAMES)

# ── Disease Info ──────────────────────────────────────────────
DISEASE_INFO = {
    "Pepper__bell___Bacterial_spot": {
        "crop": "Bell Pepper", "disease": "Bacterial Spot", "severity": "Moderate",
        "description": "Caused by Xanthomonas bacteria. Creates water-soaked lesions on leaves and fruit.",
        "treatment": "Apply copper-based bactericides. Remove infected plant material. Avoid overhead watering.",
        "prevention": "Use disease-free seeds. Practice crop rotation. Space plants for good air circulation."
    },
    "Pepper__bell___healthy": {
        "crop": "Bell Pepper", "disease": "Healthy ✅", "severity": "None",
        "description": "Your bell pepper plant looks healthy!",
        "treatment": "No treatment needed.",
        "prevention": "Continue regular watering and fertilization. Monitor for pests."
    },
    "PlantVillage": {
        "crop": "Unknown", "disease": "Dataset Label", "severity": "Unknown",
        "description": "This was classified as a dataset category. Please upload a clear leaf image.",
        "treatment": "N/A", "prevention": "Ensure the image shows a single leaf clearly."
    },
    "Potato___Early_blight": {
        "crop": "Potato", "disease": "Early Blight", "severity": "Moderate",
        "description": "Caused by Alternaria solani. Dark brown spots with concentric rings on older leaves.",
        "treatment": "Apply fungicides like chlorothalonil or mancozeb. Remove infected leaves.",
        "prevention": "Rotate crops every 2–3 years. Use certified disease-free seed potatoes."
    },
    "Potato___Late_blight": {
        "crop": "Potato", "disease": "Late Blight", "severity": "Severe",
        "description": "Caused by Phytophthora infestans. Spreads rapidly in cool, wet conditions.",
        "treatment": "Apply copper-based or systemic fungicides immediately. Destroy infected plants.",
        "prevention": "Plant resistant varieties. Avoid overhead irrigation. Ensure good drainage."
    },
    "Potato___healthy": {
        "crop": "Potato", "disease": "Healthy ✅", "severity": "None",
        "description": "Your potato plant looks healthy!",
        "treatment": "No treatment needed.",
        "prevention": "Continue proper irrigation and hilling. Watch for Colorado potato beetle."
    },
    "Tomato_Bacterial_spot": {
        "crop": "Tomato", "disease": "Bacterial Spot", "severity": "Moderate",
        "description": "Caused by Xanthomonas species. Small dark water-soaked spots on leaves and fruit.",
        "treatment": "Use copper-based sprays. Remove infected leaves. Avoid working with wet plants.",
        "prevention": "Use resistant varieties. Practice crop rotation. Apply mulch to reduce soil splash."
    },
    "Tomato_Early_blight": {
        "crop": "Tomato", "disease": "Early Blight", "severity": "Moderate",
        "description": "Caused by Alternaria solani. Target-like brown spots starting on lower leaves.",
        "treatment": "Apply chlorothalonil or copper-based fungicides. Remove affected lower leaves.",
        "prevention": "Stake plants for air circulation. Avoid wetting leaves during irrigation."
    },
    "Tomato_Late_blight": {
        "crop": "Tomato", "disease": "Late Blight", "severity": "Severe",
        "description": "Caused by Phytophthora infestans. Dark greasy lesions, spreads extremely fast.",
        "treatment": "Apply fungicides immediately. Destroy infected plants. Do not compost infected material.",
        "prevention": "Plant resistant varieties. Improve air circulation. Avoid wet foliage."
    },
    "Tomato_Leaf_Mold": {
        "crop": "Tomato", "disease": "Leaf Mold", "severity": "Moderate",
        "description": "Caused by Passalora fulva. Yellow patches on upper leaf, olive-grey mold on underside.",
        "treatment": "Apply fungicides. Reduce humidity. Remove infected leaves.",
        "prevention": "Use resistant cultivars. Ensure proper ventilation. Avoid leaf wetness."
    },
    "Tomato_Septoria_leaf_spot": {
        "crop": "Tomato", "disease": "Septoria Leaf Spot", "severity": "Moderate",
        "description": "Caused by Septoria lycopersici. Small circular spots with dark borders on lower leaves.",
        "treatment": "Apply fungicides (chlorothalonil, mancozeb). Remove infected leaves immediately.",
        "prevention": "Rotate crops. Mulch around plants. Use drip irrigation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "crop": "Tomato", "disease": "Spider Mites", "severity": "Moderate",
        "description": "Caused by Tetranychus urticae. Tiny mites cause stippled, bronze or yellow leaves.",
        "treatment": "Apply miticides or insecticidal soap. Use neem oil. Introduce predatory mites.",
        "prevention": "Keep plants well-watered. Avoid dusty conditions."
    },
    "Tomato__Target_Spot": {
        "crop": "Tomato", "disease": "Target Spot", "severity": "Moderate",
        "description": "Caused by Corynespora cassiicola. Concentric ring lesions resembling a target on leaves.",
        "treatment": "Apply fungicides. Remove and destroy infected plant debris.",
        "prevention": "Ensure good air circulation. Avoid excessive nitrogen fertilization."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "crop": "Tomato", "disease": "Yellow Leaf Curl Virus", "severity": "Severe",
        "description": "Caused by TYLCV, spread by whiteflies. Causes leaf curling, yellowing, severe yield loss.",
        "treatment": "No cure. Remove and destroy infected plants. Control whitefly populations.",
        "prevention": "Use resistant varieties. Apply reflective mulches. Control whiteflies."
    },
    "Tomato__Tomato_mosaic_virus": {
        "crop": "Tomato", "disease": "Mosaic Virus", "severity": "Severe",
        "description": "Caused by ToMV. Mosaic patterns of light and dark green on leaves, stunted growth.",
        "treatment": "No cure. Remove infected plants. Disinfect tools with bleach solution.",
        "prevention": "Use resistant varieties. Wash hands before handling plants. Control aphids."
    },
    "Tomato_healthy": {
        "crop": "Tomato", "disease": "Healthy ✅", "severity": "None",
        "description": "Your tomato plant looks healthy!",
        "treatment": "No treatment needed.",
        "prevention": "Continue proper care: consistent watering, balanced fertilization, and regular monitoring."
    }
}

SEVERITY_COLORS = {
    "None": "#2e7d32", "Moderate": "#e65100",
    "Severe": "#b71c1c", "Unknown": "#616161"
}

# ── Styling ───────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #f0f7f0 0%, #e8f5e9 50%, #f1f8e9 100%); }
    .hero-header { text-align: center; padding: 2rem 0 1rem 0; }
    .hero-title { font-family:'Nunito',sans-serif; font-size:2.6rem; font-weight:800; color:#1b5e20; letter-spacing:-0.5px; }
    .hero-subtitle { font-size:1.05rem; color:#558b2f; }
    .stat-box { background:white; border-radius:12px; padding:1rem; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); }
    .stat-number { font-family:'Nunito',sans-serif; font-size:1.8rem; font-weight:800; color:#2e7d32; }
    .stat-label  { font-size:0.78rem; color:#757575; font-weight:500; }
    [data-testid="stFileUploader"] { border:2px dashed #81c784; border-radius:16px; padding:1rem; background:rgba(255,255,255,0.6); }
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #43a047) !important;
        color:white !important; border:none !important; border-radius:12px !important;
        padding:0.7em 2.5em !important; font-size:1.05rem !important; font-weight:600 !important;
        width:100% !important; box-shadow:0 4px 15px rgba(46,125,50,0.3) !important;
    }
    .stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 6px 20px rgba(46,125,50,0.4) !important; }
    .result-card { background:white; border-radius:16px; padding:1.5rem; box-shadow:0 4px 20px rgba(0,0,0,0.08); margin-top:0.5rem; border-top:5px solid #2e7d32; }
    .result-crop { font-family:'Nunito',sans-serif; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; color:#757575; margin-bottom:0.2rem; }
    .result-disease { font-family:'Nunito',sans-serif; font-size:1.6rem; font-weight:800; margin-bottom:0.6rem; }
    .severity-badge { display:inline-block; padding:0.25em 0.9em; border-radius:20px; font-size:0.8rem; font-weight:700; color:white; margin-bottom:1rem; }
    .info-section { background:#f9fbe7; border-radius:10px; padding:0.8rem 1rem; margin-bottom:0.7rem; border-left:4px solid #aed581; }
    .info-label { font-size:0.72rem; font-weight:700; text-transform:uppercase; letter-spacing:1px; color:#558b2f; margin-bottom:0.25rem; }
    .info-text { font-size:0.9rem; color:#424242; line-height:1.5; }
    .placeholder-box { background:white; border-radius:16px; padding:2.5rem; text-align:center; box-shadow:0 2px 10px rgba(0,0,0,0.06); color:#9e9e9e; margin-top:0.5rem; }
    hr { border:none; border-top:1px solid #e0e0e0; margin:1.5rem 0; }
    .footer { text-align:center; padding:1.5rem 0 0.5rem; color:#9e9e9e; font-size:0.82rem; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl = models.efficientnet_b2(weights=None)
    in_features = mdl.classifier[1].in_features
    mdl.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load("models/best_model.pth", map_location=device, weights_only=True)
    mdl.load_state_dict(checkpoint["model_state_dict"])
    mdl.to(device)
    mdl.eval()
    return mdl, device

# ── Preprocessing (matches training) ─────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class='hero-header'>
    <div class='hero-title'>🌿 Crop Disease Detector</div>
    <div class='hero-subtitle'>AI-powered plant disease detection · EfficientNet-B2 · PlantVillage Dataset</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("<div class='stat-box'><div class='stat-number'>3</div><div class='stat-label'>Crops Supported</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='stat-box'><div class='stat-number'>16</div><div class='stat-label'>Disease Classes</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='stat-box'><div class='stat-number'>224px</div><div class='stat-label'>Input Resolution</div></div>", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ── Load Model with error handling ───────────────────────────
try:
    model, device = load_model()
    model_loaded = True
except FileNotFoundError:
    st.error("❌ `models/best_model.pth` not found. Make sure it's in the `models/` folder.")
    model_loaded = False
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    model_loaded = False

# ── Main Layout ───────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("#### 📁 Upload Leaf Image")
    uploaded_file = st.file_uploader("Drag & drop or browse", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        detect_clicked = st.button("🔍 Detect Disease", disabled=not model_loaded)
    else:
        st.info("👆 Upload a clear photo of a single leaf to get started.")
        detect_clicked = False

with right_col:
    st.markdown("#### 🔬 Detection Results")

    if uploaded_file and detect_clicked and model_loaded:
        with st.spinner("Analyzing leaf..."):
            tensor = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(tensor)
                probs   = torch.softmax(outputs, dim=1)[0]
                conf    = float(probs.max().item()) * 100
                idx     = int(probs.argmax().item())
                label   = CLASS_NAMES[idx]

        info = DISEASE_INFO.get(label, {
            "crop": "Unknown", "disease": label, "severity": "Unknown",
            "description": "No info available.", "treatment": "Consult an agronomist.",
            "prevention": "Monitor your crops regularly."
        })
        sev_color = SEVERITY_COLORS.get(info["severity"], "#616161")
        sev_icon  = {"None": "✅", "Moderate": "⚠️", "Severe": "🚨"}.get(info["severity"], "ℹ️")

        st.markdown(f"""
        <div class='result-card' style='border-top-color:{sev_color};'>
            <div class='result-crop'>{info["crop"]}</div>
            <div class='result-disease' style='color:{sev_color};'>{info["disease"]}</div>
            <span class='severity-badge' style='background:{sev_color};'>{sev_icon} Severity: {info["severity"]}</span>
            <div class='info-section'>
                <div class='info-label'>📋 Description</div>
                <div class='info-text'>{info["description"]}</div>
            </div>
            <div class='info-section'>
                <div class='info-label'>💊 Treatment</div>
                <div class='info-text'>{info["treatment"]}</div>
            </div>
            <div class='info-section'>
                <div class='info-label'>🛡️ Prevention</div>
                <div class='info-text'>{info["prevention"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"**Model Confidence: {conf:.1f}%**")
        st.progress(conf / 100)

        with st.expander("📊 Top 3 Predictions"):
            top3 = torch.topk(probs, 3)
            for i in range(3):
                name = CLASS_NAMES[top3.indices[i].item()].replace("___", " → ").replace("_", " ")
                pct  = float(top3.values[i].item()) * 100
                st.markdown(f"**{i+1}. {name}** — {pct:.1f}%")
                st.progress(pct / 100)

    elif not uploaded_file:
        st.markdown("<div class='placeholder-box'><div style='font-size:3rem;margin-bottom:0.5rem;'>🌱</div><div>Upload an image and click<br><strong>Detect Disease</strong> to see results.</div></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='placeholder-box'><div style='font-size:3rem;margin-bottom:0.5rem;'>👈</div><div>Click <strong>Detect Disease</strong><br>to analyze your leaf.</div></div>", unsafe_allow_html=True)

# ── Supported Crops ───────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("#### 🌾 Supported Crops & Diseases")
tab1, tab2, tab3 = st.tabs(["🫑 Bell Pepper", "🥔 Potato", "🍅 Tomato"])
with tab1:
    st.markdown("- Bacterial Spot\n- ✅ Healthy")
with tab2:
    st.markdown("- Early Blight\n- Late Blight\n- ✅ Healthy")
with tab3:
    st.markdown("- Bacterial Spot\n- Early Blight\n- Late Blight\n- Leaf Mold\n- Septoria Leaf Spot\n- Spider Mites\n- Target Spot\n- Yellow Leaf Curl Virus\n- Mosaic Virus\n- ✅ Healthy")

st.markdown("<div class='footer'>🌿 Crop Disease Detection System &nbsp;|&nbsp; EfficientNet-B2 &nbsp;|&nbsp; PlantVillage Dataset</div>", unsafe_allow_html=True)
