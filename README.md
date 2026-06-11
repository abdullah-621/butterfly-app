# 🦋 Butterfly Species Classifier

A deep learning-powered web app that identifies **75 butterfly species** from a single photo — built with CNN, ONNX Runtime, and Streamlit.

🌐 **Live App:** [butterfly-app-2504.streamlit.app](https://butterfly-app-2504.streamlit.app/)

---

## 📸 Demo

Upload any butterfly photo and get:
- ✅ Predicted species name
- ✅ Confidence score
- ✅ Top 5 predictions with probability bars

---

## 🚀 Features

- 🧠 CNN model trained on 75 butterfly species
- ⚡ Fast inference using ONNX Runtime
- 🎨 Clean, responsive UI built with Streamlit
- ☁️ Deployed on Streamlit Community Cloud
- 📦 Large model file managed with Git LFS

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Model Training | TensorFlow / Keras |
| Model Format | ONNX |
| Inference | ONNX Runtime |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |
| Version Control | Git + Git LFS |

---

## 📁 Project Structure

```
butterfly-app/
├── app.py                   # Streamlit frontend + inference
├── requirements.txt         # Python dependencies
├── butterfly_model.onnx     # Trained CNN model (ONNX format)
└── class_names.json         # 75 butterfly species labels
```

---

## ⚙️ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/abdullah-621/butterfly-app.git
cd butterfly-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` ✅

---

## 🧠 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input Size:** 150 × 150 × 3 (RGB)
- **Output:** 75 species probability scores
- **Training Framework:** TensorFlow / Keras
- **Inference Format:** ONNX (converted for Python 3.14 compatibility)

### 🦋 Sample Species
| | | |
|---|---|---|
| Adonis | African Giant Swallowtail | American Snoot |
| Atala | Banded Orange Heliconian | Black Hairstreak |
| Cabbage White | Cairns Birdwing | Chestnut |
| Clodius Parnassian | Clouded Sulphur | Common Wood-Nymph |
| *...and 63 more* | | |

---

## 🚧 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Python 3.14 incompatibility with TensorFlow | Converted model to ONNX format |
| 218MB model too large for GitHub | Converted to ONNX (72MB) + Git LFS |
| `tf2onnx` conversion errors | Used `model.export()` → SavedModel → ONNX CLI |
| Streamlit Cloud dependency errors | Switched from `tensorflow` to `onnxruntime` |

---

## 📦 Dependencies

```
streamlit>=1.35.0
onnxruntime>=1.18.0
pillow>=10.0.0
numpy>=1.24.0
```

---

## 🌐 Deployment

This app is deployed on **Streamlit Community Cloud**.

To deploy your own version:
1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set main file to `app.py`
4. Click **Deploy** ✅

---

## 👨‍💻 Author

**Abdullah Al Masum**
- GitHub: [@abdullah-621](https://github.com/abdullah-621)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
