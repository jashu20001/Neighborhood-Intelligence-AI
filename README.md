# Neighborhood Intelligence AI

A California-focused real estate analysis system built with Streamlit. It combines a trained Random Forest machine learning model with structured reasoning agents and a local LLM to deliver interpretable investment guidance on neighborhood-level housing data.

---

## Live Demo

> Deploy link will appear here after Streamlit Cloud deployment.

---

## Overview

This app takes neighborhood-level inputs based on the California Housing Dataset and returns a price estimate, investment score, confidence rating, and a multi-agent analysis covering market conditions, risk factors, and investment outlook. A built-in AI chat interface lets you ask follow-up questions about any active analysis.

**Model performance on held-out test data:**

- RMSE: ~$50,938
- R2 Score: 0.80

---

## Features

- Predicted median house value for a California neighborhood block group
- Investment score out of 100 with a verdict label (Strong / Moderate / Cautious)
- Confidence score based on input quality validation
- Market analyst, risk agent, and investment agent outputs with structured reasoning
- Strengths and risks breakdown with labeled factors
- Interactive map showing the analyzed location
- AI explanation of how the decision was reached (powered by Ollama/Mistral locally)
- Follow-up chat grounded in the current analysis
- Recent analysis history panel
- Feedback saving per session

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend and UI | Streamlit |
| ML Model | scikit-learn Random Forest Regressor |
| Data | California Housing Dataset (housing.csv) |
| LLM (local) | Ollama with Mistral |
| Data processing | pandas, numpy |
| Model persistence | joblib |

---

## Project Structure

```
real-estate-ai-analyst/
├── app/
│   └── app.py               # Main Streamlit application
├── data/
│   └── housing.csv          # California housing training data
├── models/
│   └── real_estate_rf.pkl   # Trained model (excluded from repo, see note below)
├── Untitled.ipynb            # Model training and exploration notebook
├── requirements.txt
└── .gitignore
```

---

## Local Setup

**Prerequisites:** Python 3.10 or later, and [Ollama](https://ollama.com) installed locally for LLM features.

**1. Clone the repository**

```bash
git clone https://github.com/jashu20001/Neighborhood-Intelligence-AI.git
cd Neighborhood-Intelligence-AI
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Obtain the trained model**

The model file (`models/real_estate_rf.pkl`) is not stored in this repository due to its size (138 MB). You can recreate it by running the training notebook:

```bash
jupyter notebook Untitled.ipynb
```

Run all cells in order. The final cell saves the model to `models/real_estate_rf.pkl` automatically.

**5. Pull the local LLM (optional)**

LLM features (final recommendation, AI reasoning, chat) require Ollama running locally with the Mistral model. The app will still run without it, but those sections will show an error message instead.

```bash
ollama pull mistral
```

**6. Run the app**

```bash
streamlit run app/app.py
```

The app opens at `http://localhost:8501`.

---

## Streamlit Cloud Deployment

To make the app publicly accessible, deploy it on [Streamlit Community Cloud](https://streamlit.io/cloud) (free).

**Step 1 - Push your repo to GitHub** (if not done already)

```bash
git push --force origin main
```

**Step 2 - Host the trained model externally**

Because the `.pkl` model file (138 MB) is excluded from the repo, you need to host it somewhere and have the app download it on startup. The recommended option is Hugging Face Hub (free, no account tier required for public repos).

- Go to [huggingface.co](https://huggingface.co) and create a free account
- Create a new model repository
- Upload `models/real_estate_rf.pkl` via the web interface
- Copy the download URL (it will look like: `https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/real_estate_rf.pkl`)

Then add this block near the top of `app/app.py`, just after the imports:

```python
import urllib.request

MODEL_URL = "https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/real_estate_rf.pkl"

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
```

Replace `MODEL_URL` with your actual Hugging Face file URL.

**Step 3 - Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
2. Click **New app**
3. Select your repository: `jashu20001/Neighborhood-Intelligence-AI`
4. Set the main file path to: `app/app.py`
5. Click **Deploy**

Streamlit Cloud will install your `requirements.txt` automatically. The first launch will download the model from Hugging Face before the app loads.

**Note on LLM features in the cloud:** Ollama runs locally and is not available on Streamlit Cloud. The app handles this gracefully: the final recommendation, AI reasoning, and chat sections will show a local LLM error message instead of a response. All other features (price prediction, investment scoring, market analysis, risk breakdown, map) will work normally.

---

## Dataset

This project uses the **California Housing Dataset**, which contains block-group-level census data for California. Each row represents a neighborhood block group, not a single property. Inputs like total rooms, total bedrooms, and population are neighborhood-level aggregates.

The model is trained on this data and performs most reliably for California latitude (32.54 to 41.95) and longitude (-124.35 to -114.31) ranges.

---

## Notes

- This tool is for educational and exploratory purposes only. It is not financial or investment advice.
- Predictions are based on historical California census data and may not reflect current market conditions.
- The LLM component (Ollama/Mistral) runs locally and is not required for the core ML features to function.

---

## Author

Built by [jas](https://github.com/jashu20001)
