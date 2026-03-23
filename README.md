# VibeMatch 🎵 — Music Recommendation System

A content-based music recommendation system built with **PySpark MLlib**, **scikit-learn**, and **Streamlit**.

## How It Works
- **PySpark backend**: Processes song lyrics using TF-IDF + LSH to pre-compute a similarity matrix for fast recommendations.
- **scikit-learn fallback**: Builds 100-dim SVD vectors for all 57k songs so every song in the dataset gets a live recommendation if it's not in the pre-computed matrix.
- **Streamlit frontend**: Serves recommendations instantly from the pre-computed matrix (✨) or computes them live on-the-fly (⚡).

## Setup

### 1. Dataset
Download `spotify_millsongdata.csv` from Kaggle and place it in the project root:
> 🔗 https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset

### 2. Install training dependencies
_(Only needed to regenerate model files — requires Java 11)_
```bash
pip install -r requirements-train.txt
```

### 3. Run the training pipeline
```bash
python src/backend/train.py
```
Generates pre-computed model files in `models/`. Takes ~5–15 min locally.

> **Note:** Requires Java 11 (e.g. [Eclipse Adoptium JDK 11](https://adoptium.net/)). Set `JAVA_HOME` to your JDK path if PySpark can't find it.

### 4. Run the Streamlit app
```bash
pip install -r requirements.txt
streamlit run src/frontend/app.py
```

The `models/` folder is already committed — you can skip steps 1–3 and run the app directly if you just want to use it.
