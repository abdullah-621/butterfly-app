"""
🦋 Butterfly Species Classifier
Streamlit single-file app — no FastAPI needed.
Place butterfly_model.keras (or .h5) and class_names.json
in the same folder as this file before running.
"""

import streamlit as st
import numpy as np
import json
import io
import os
from PIL import Image

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🦋 Butterfly Classifier",
    page_icon="🦋",
    layout="centered",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    background: linear-gradient(135deg, #2d6a4f, #95d5b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 0.4rem;
}
.hero-sub {
    color: #6b7280;
    font-size: 1rem;
    margin-bottom: 0;
}
.result-card {
    background: linear-gradient(135deg, #d8f3dc33, #b7e4c733);
    border: 1.5px solid #74c69d88;
    border-radius: 20px;
    padding: 1.8rem 1.5rem;
    text-align: center;
    margin: 1.2rem 0;
}
.species {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #d8f3dc;
    letter-spacing: 0.02em;
}
.conf-label {
    color: #95d5b2;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0.3rem;
}
.top5-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.35rem 0;
    border-bottom: 1px solid #e5e7eb;
    font-size: 0.92rem;
}
.top5-row:last-child { border-bottom: none; }
.rank { color: #9ca3af; width: 1.4rem; }
.bar-wrap {
    flex: 1;
    height: 6px;
    background: #e5e7eb;
    border-radius: 99px;
    margin: 0 0.75rem;
    overflow: hidden;
}
.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #52b788, #2d6a4f);
    border-radius: 99px;
}
.pct { color: #374151; font-weight: 600; width: 3.2rem; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🦋 Butterfly Classifier</div>
    <p class="hero-sub">Upload a butterfly photo — discover its species instantly</p>
</div>
""", unsafe_allow_html=True)

# ── Model loader (cached so it only loads once) ────────────────────────────────
@st.cache_resource(show_spinner="Loading model… this takes ~30 seconds on first run")
def load_model():
    # Import inside function so Streamlit Cloud only loads TF when needed
    try:
        import keras
        model_path = "butterfly_model.onnx"
        names_path = "class_names.json"

        if not os.path.exists(model_path):
            # fallback to .h5
            model_path = "butterfly_model.h5"

        if not os.path.exists(model_path):
            return None, {}, "❌ Model file not found. Add `butterfly_model.onnx` to the repo."

        if not os.path.exists(names_path):
            return None, {}, "❌ `class_names.json` not found. Add it to the repo."

        model = keras.models.load_model(model_path)

        with open(names_path) as f:
            raw = json.load(f)
        class_names = {str(k): v for k, v in raw.items()}

        return model, class_names, None

    except Exception as e:
        return None, {}, f"❌ Model load error: {e}"


model, class_names, load_error = load_model()

if load_error:
    st.error(load_error)
    st.info("📂 Make sure `butterfly_model.keras` and `class_names.json` are in the **root of your GitHub repo**.")
    st.stop()

st.success(f"✅ Model ready — {len(class_names)} species loaded", icon="🦋")

# ── Image Upload ───────────────────────────────────────────────────────────────
st.divider()
uploaded = st.file_uploader(
    "Upload a butterfly image",
    type=["jpg", "jpeg", "png", "webp"],
    help="A clear, well-lit photo gives the best results"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    fmt = img.format or uploaded.name.rsplit(".", 1)[-1].upper()

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, use_container_width=True, caption=f"{uploaded.name} · {img.size[0]}×{img.size[1]} · {fmt}")

    with col2:
        if st.button("🔍 Identify Species", use_container_width=True, type="primary"):
            with st.spinner("Analysing…"):
                # Preprocess
                img_resized = img.resize((150, 150))
                arr = np.array(img_resized, dtype=np.float32) / 255.0
                arr = np.expand_dims(arr, axis=0)   # (1, 150, 150, 3)

                preds = model.predict(arr, verbose=0)[0]
                top5  = np.argsort(preds)[::-1][:5]

                best_label = class_names.get(str(top5[0]), f"Class {top5[0]}")
                best_conf  = round(float(preds[top5[0]]) * 100, 1)

            # ── Result card ──
            st.markdown(f"""
            <div class="result-card">
                <div class="species">{best_label.title()}</div>
                <div class="conf-label">Confidence: {best_conf}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(best_conf / 100)

            # ── Top 5 ──
            st.markdown("#### Top 5 Predictions")
            rows_html = ""
            for idx in top5:
                label = class_names.get(str(idx), f"Class {idx}").title()
                conf  = round(float(preds[idx]) * 100, 1)
                rows_html += f"""
                <div class="top5-row">
                    <span class="rank">#{top5.tolist().index(idx)+1}</span>
                    <span style="flex:2">{label}</span>
                    <div class="bar-wrap"><div class="bar-fill" style="width:{conf}%"></div></div>
                    <span class="pct">{conf}%</span>
                </div>"""
            st.markdown(rows_html, unsafe_allow_html=True)

        else:
            st.info("👆 Click **Identify Species** to classify")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<p style='text-align:center;color:#9ca3af;font-size:0.78rem;'>Butterfly Classifier · CNN · 75 Species</p>",
    unsafe_allow_html=True
)
