# 🦋 Butterfly Classifier

A Streamlit app that classifies butterfly species using a CNN model trained on 75 species.

## Files needed in this repo

```
├── app.py
├── requirements.txt
├── butterfly_model.keras   ← your trained model
└── class_names.json        ← your class name mapping
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push all files (including `butterfly_model.keras` and `class_names.json`) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo → set **Main file path** to `app.py`
4. Click **Deploy**
