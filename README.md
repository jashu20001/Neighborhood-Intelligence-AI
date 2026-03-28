# Neighborhood Intelligence AI

An AI-powered real estate analysis platform that combines machine learning, multi-agent reasoning, and natural language processing to evaluate California neighborhoods and deliver interpretable investment guidance.

**[View Live Demo](https://neighborhood-intelligence-ai-zlxawhn2tonbirgw4fopft.streamlit.app)**

---

## What It Does

Most real estate tools give you a number. This one gives you a decision.

Enter neighborhood-level data and the system runs it through a trained ML model, three specialized AI agents, and a local language model to produce a full investment analysis with a price estimate, risk breakdown, confidence rating, and a plain-English recommendation you can actually act on.

---

## How It Works

The app is built around a multi-agent architecture where each agent has a distinct role:

**Market Analyst Agent** evaluates income levels, space availability, and location relative to the California housing market to assess pricing strength.

**Risk Agent** examines population density, room composition, and relative pricing to surface crowding risks, infrastructure pressure, and overvaluation signals.

**Investment Agent** synthesizes both views into a scored verdict (0 to 100) with a label of Strong, Moderate, or Cautious opportunity.

**LLM Layer** takes the agent outputs and generates a final recommendation and reasoning breakdown in plain language using a local Mistral model.

---

## Features

- Predicted neighborhood median house value powered by a trained Random Forest model
- Investment score out of 100 with a labeled verdict
- Confidence score derived from input quality validation
- Strengths and risks breakdown with individual factor cards
- Interactive location map for spatial context
- AI-generated final recommendation and reasoning explanation
- Follow-up chat grounded in the active analysis context
- Recent analysis history panel
- Per-session feedback collection

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Machine Learning | scikit-learn Random Forest Regressor |
| Reasoning Layer | Custom multi-agent system |
| LLM (local) | Ollama with Mistral |
| Data | California Housing Dataset |
| Data processing | pandas, numpy |
| Model hosting | Hugging Face Hub |

---

## Project Structure

```
real-estate-ai-analyst/
├── app/
│   └── app.py               # Main Streamlit application
├── data/
│   └── housing.csv          # California housing training data
├── models/
│   └── real_estate_rf.pkl   # Trained model (hosted on Hugging Face)
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

**4. Train the model**

The model file is not stored in this repository. Recreate it by running the training notebook:

```bash
jupyter notebook Untitled.ipynb
```

Run all cells in order. The final cell saves the model to `models/real_estate_rf.pkl` automatically.

**5. Pull the local LLM (optional)**

LLM features require Ollama running locally with the Mistral model. The app runs without it, but the recommendation and chat sections will not be available.

```bash
ollama pull mistral
```

**6. Run the app**

```bash
streamlit run app/app.py
```

Opens at `http://localhost:8501`.

---

## Notes

- Built for educational and portfolio purposes. Not financial or investment advice.
- Predictions are based on historical California census data and may not reflect current market conditions.
- The LLM component is optional. All core ML features work independently of it.

---

## Author

Built by [jas](https://github.com/jashu20001)
