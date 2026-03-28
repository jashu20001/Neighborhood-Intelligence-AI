import json
import subprocess
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# PATHS
# =========================================================
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "real_estate_rf.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
HISTORY_PATH = PROJECT_ROOT / "history.json"
MEMORY_PATH = PROJECT_ROOT / "memory.json"

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Neighborhood Intelligence AI",
    page_icon="🏠",
    layout="wide",
)

# =========================================================
# MODEL DOWNLOAD (for cloud deployment)
# =========================================================
MODEL_URL = "https://huggingface.co/Jaswanth737/neighborhood-intelligence-ai/resolve/main/real_estate_rf.pkl?download=true"

if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with st.spinner("Downloading model... this may take a minute."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #090d16 0%, #0f1724 100%);
        }
        .block-container {
            max-width: 1280px;
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .hero-box {
            padding: 1.35rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(20,20,20,0.96), rgba(28,20,10,0.96));
            border: 1px solid rgba(255,140,0,0.18);
            box-shadow: 0 14px 34px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2.15rem;
            font-weight: 800;
            color: #ffa726;
            margin-bottom: 0.2rem;
        }
        .hero-sub {
            color: #e5e7eb;
            font-size: 1rem;
            line-height: 1.55;
        }
        .soft-card {
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(18, 18, 18, 0.88);
            border: 1px solid rgba(255,140,0,0.14);
            box-shadow: 0 10px 24px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }
        .soft-title {
            font-size: 1.04rem;
            font-weight: 700;
            color: #ffb74d;
            margin-bottom: 0.55rem;
        }
        .warning-box {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            background: rgba(120,53,15,0.18);
            border: 1px solid rgba(255,167,38,0.30);
            color: #ffd699;
            margin-bottom: 1rem;
        }
        .info-box {
            padding: 0.95rem 1rem;
            border-radius: 14px;
            background: rgba(30, 30, 30, 0.72);
            border: 1px solid rgba(255,140,0,0.15);
            color: #f3f4f6;
            margin-bottom: 1rem;
        }
        .factor-good {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: rgba(38, 50, 24, 0.7);
            border: 1px solid rgba(255,140,0,0.12);
            margin-bottom: 0.7rem;
            color: #f3f4f6;
        }
        .factor-risk {
            padding: 0.85rem 1rem;
            border-radius: 14px;
            background: rgba(60, 26, 18, 0.7);
            border: 1px solid rgba(255,140,0,0.12);
            margin-bottom: 0.7rem;
            color: #f3f4f6;
        }
        .factor-title {
            font-weight: 700;
            color: #ffa726;
            margin-bottom: 0.2rem;
        }
        .small-note {
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        div[data-testid="stMetric"] {
            background: rgba(18, 18, 18, 0.88);
            border: 1px solid rgba(255,140,0,0.14);
            padding: 14px;
            border-radius: 16px;
            box-shadow: 0 10px 22px rgba(0,0,0,0.20);
        }
        div[data-testid="stMetricLabel"] {
            color: #f3f4f6 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ffa726 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1118 0%, #131820 100%);
        }
        [data-testid="stSidebar"] * {
            color: #f3f4f6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOADERS
# =========================================================
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset_bundle():
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)

    # Reference frame for medians and range checks
    X_ref = df.drop("median_house_value", axis=1).drop("ocean_proximity", axis=1)
    X_ref = X_ref.fillna(X_ref.median())

    X_ref["rooms_per_household"] = X_ref["total_rooms"] / X_ref["households"]
    X_ref["bedrooms_per_room"] = X_ref["total_bedrooms"] / X_ref["total_rooms"]
    X_ref["population_per_household"] = X_ref["population"] / X_ref["households"]

    stats = {
        "lat_min": float(df["latitude"].min()),
        "lat_max": float(df["latitude"].max()),
        "lon_min": float(df["longitude"].min()),
        "lon_max": float(df["longitude"].max()),
        "median_income_median": float(X_ref["median_income"].median()),
        "rooms_per_household_median": float(X_ref["rooms_per_household"].median()),
        "bedrooms_per_room_median": float(X_ref["bedrooms_per_room"].median()),
        "population_per_household_median": float(X_ref["population_per_household"].median()),
        "price_mean": float(df["median_house_value"].mean()),
    }

    return df, X_ref, stats


model = load_model()
dataset_bundle = load_dataset_bundle()

# =========================================================
# JSON HELPERS
# =========================================================
def load_json_list(path: Path):
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_json_list(path: Path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_history(entry: dict):
    history = load_json_list(HISTORY_PATH)
    history.append(entry)
    save_json_list(HISTORY_PATH, history)


def save_feedback(entry: dict):
    memory = load_json_list(MEMORY_PATH)
    memory.append(entry)
    save_json_list(MEMORY_PATH, memory)

# =========================================================
# LLM
# =========================================================
def ask_llm(prompt: str) -> str:
    try:
        result = subprocess.run(
            ["ollama", "run", "mistral"],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120,
        )
        output = result.stdout.strip()
        if output:
            return output
        err = result.stderr.strip()
        return f"Local LLM error: {err}" if err else "No response returned from the local model."
    except Exception as e:
        return f"Local LLM error: {e}"


def stream_words(text: str):
    for word in text.split():
        yield word + " "

# =========================================================
# ANALYSIS HELPERS
# =========================================================
def validate_inputs(
    population: int,
    households: int,
    total_rooms: int,
    total_bedrooms: int,
    latitude: float,
    longitude: float,
    stats: dict,
):
    warnings = []
    confidence = 100

    population_per_household = population / households if households else 0
    bedrooms_per_room = total_bedrooms / total_rooms if total_rooms else 0

    if total_bedrooms > total_rooms:
        warnings.append("Total bedrooms are greater than total rooms.")
        confidence -= 35

    if population_per_household > 10:
        warnings.append("Population per household is unusually high for this California neighborhood dataset.")
        confidence -= 30

    if bedrooms_per_room > 0.8:
        warnings.append("Bedroom-to-room ratio is unusually high.")
        confidence -= 20

    if total_rooms < 100:
        warnings.append("Total rooms look very low for a neighborhood-level record.")
        confidence -= 20

    if not (stats["lat_min"] <= latitude <= stats["lat_max"]):
        warnings.append(
            f"Latitude is outside the dataset range ({stats['lat_min']:.2f} to {stats['lat_max']:.2f})."
        )
        confidence -= 15

    if not (stats["lon_min"] <= longitude <= stats["lon_max"]):
        warnings.append(
            f"Longitude is outside the dataset range ({stats['lon_min']:.2f} to {stats['lon_max']:.2f})."
        )
        confidence -= 15

    confidence = max(confidence, 10)
    return warnings, confidence, population_per_household, bedrooms_per_room


def market_analyst_agent(predicted_price, median_income_scaled, rooms_per_household, latitude, longitude, stats):
    parts = []

    if median_income_scaled > stats["median_income_median"]:
        parts.append("Income levels are stronger than the California median in the training data, which supports higher pricing.")
    else:
        parts.append("Income levels are closer to or below the dataset median, which limits pricing strength.")

    if rooms_per_household > stats["rooms_per_household_median"]:
        parts.append("Space availability per household is comparatively healthy.")
    else:
        parts.append("Space per household is tighter than stronger-performing neighborhoods.")

    if stats["lat_min"] <= latitude <= stats["lat_max"] and stats["lon_min"] <= longitude <= stats["lon_max"]:
        parts.append("The location falls inside the California training range, so interpretation is more reliable.")
    else:
        parts.append("The location sits outside the training range, so the estimate should be treated more cautiously.")

    if predicted_price > stats["price_mean"]:
        parts.append("The predicted value is above the dataset average, which suggests a relatively stronger market position.")
    else:
        parts.append("The predicted value is below the dataset average, which suggests a more moderate market segment.")

    return " ".join(parts)


def risk_agent(predicted_price, population_per_household, bedrooms_per_room, stats):
    parts = []

    if population_per_household > stats["population_per_household_median"]:
        parts.append("Population density is elevated, which can reduce livability and create infrastructure pressure.")
    else:
        parts.append("Population density looks relatively manageable.")

    if bedrooms_per_room > stats["bedrooms_per_room_median"]:
        parts.append("Room composition appears tighter than average, which may signal crowding risk.")
    else:
        parts.append("Room composition does not show strong space stress.")

    pricing_ratio = predicted_price / stats["price_mean"]

    if pricing_ratio > 1.5:
        parts.append("The estimate is far above the dataset average, which introduces clear overpricing risk.")
    elif pricing_ratio > 1.2:
        parts.append("The estimate is moderately above the dataset average, so pricing risk should be watched.")
    elif pricing_ratio < 0.6:
        parts.append("The estimate is far below the dataset average, which may point to weaker market conditions.")
    else:
        parts.append("The estimate is within a more reasonable range relative to the dataset average.")

    return " ".join(parts)


def investment_agent(predicted_price, median_income_scaled, rooms_per_household, population_per_household, stats):
    score = 50

    if median_income_scaled > stats["median_income_median"]:
        score += 15
    if rooms_per_household > stats["rooms_per_household_median"]:
        score += 10
    if population_per_household < stats["population_per_household_median"]:
        score += 10

    pricing_ratio = predicted_price / stats["price_mean"]
    if 0.8 <= pricing_ratio <= 1.2:
        score += 15
    elif 0.6 <= pricing_ratio < 0.8:
        score += 10
    elif 1.2 < pricing_ratio <= 1.5:
        score -= 10
    elif pricing_ratio > 1.5:
        score -= 25
    else:
        score -= 15

    score = max(min(score, 100), 0)

    if score >= 75:
        verdict = "Strong investment opportunity."
        label = "🟢 Strong Opportunity"
        tone = "success"
    elif score >= 60:
        verdict = "Moderate investment opportunity."
        label = "🟠 Moderate Opportunity"
        tone = "warning"
    else:
        verdict = "Cautious investment opportunity."
        label = "🔴 Cautious Opportunity"
        tone = "error"

    return score, verdict, label, tone


def build_strengths_and_risks(
    predicted_price: float,
    median_income: int,
    rooms_per_household: float,
    population_per_household: float,
    bedrooms_per_room: float,
    stats: dict,
):
    strengths = []
    risks = []

    if median_income >= 90000:
        strengths.append({
            "title": "Income Strength",
            "text": "The area has strong purchasing power, which usually supports better demand and price resilience."
        })
    else:
        risks.append({
            "title": "Income Strength",
            "text": "Income is not especially high, so demand support is more limited."
        })

    if 2 <= population_per_household <= 4:
        strengths.append({
            "title": "Density Balance",
            "text": "Population per household is in a healthy range, which supports livability."
        })
    elif population_per_household > 4:
        risks.append({
            "title": "Density Pressure",
            "text": "Population per household is elevated, which can create crowding and infrastructure pressure."
        })
    else:
        risks.append({
            "title": "Density Balance",
            "text": "The density pattern is unusual relative to the training data and should be interpreted cautiously."
        })

    if 4 <= rooms_per_household <= 6:
        strengths.append({
            "title": "Space per Household",
            "text": "The neighborhood has a good space-to-household balance."
        })
    elif rooms_per_household > 6:
        strengths.append({
            "title": "Space per Household",
            "text": "The neighborhood appears spacious relative to household count."
        })
    else:
        risks.append({
            "title": "Space per Household",
            "text": "Space per household is tighter than ideal, which may reduce comfort and attractiveness."
        })

    if 0.1 <= bedrooms_per_room <= 0.3:
        strengths.append({
            "title": "Bedroom Ratio",
            "text": "The room composition looks balanced and not overly cramped."
        })
    else:
        risks.append({
            "title": "Bedroom Ratio",
            "text": "The bedroom-to-room ratio is outside a comfortable range, which may point to crowding or inefficient layout."
        })

    pricing_ratio = predicted_price / stats["price_mean"]
    if 0.8 <= pricing_ratio <= 1.2:
        strengths.append({
            "title": "Pricing Stretch",
            "text": "The predicted value sits in a healthy zone relative to the dataset average."
        })
    elif 0.6 <= pricing_ratio < 0.8:
        strengths.append({
            "title": "Pricing Stretch",
            "text": "The predicted value looks somewhat below average, which may create value upside."
        })
    elif 1.2 < pricing_ratio <= 1.5:
        risks.append({
            "title": "Pricing Stretch",
            "text": "The predicted value is above average, so overpricing risk is starting to matter."
        })
    else:
        risks.append({
            "title": "Pricing Stretch",
            "text": "The predicted value looks far above average, which introduces strong overpricing risk."
        })

    return strengths, risks


def final_recommendation_llm(market_view, risk_view, investment_score, investment_verdict):
    prompt = f"""
You are a senior real estate AI strategist.

Write:
1. A short final recommendation in 2 sentences
2. A one-line reason summary

Inputs:
Market Analyst: {market_view}
Risk Agent: {risk_view}
Investment Score: {investment_score}/100
Investment Verdict: {investment_verdict}

Keep it concise, clear, and decision-oriented.
"""
    return ask_llm(prompt)


def reasoning_breakdown_llm(market_view, risk_view, investment_score, investment_verdict):
    prompt = f"""
Explain how the AI reached its decision for a California neighborhood investment analysis.

Structure it into 3 short sections:
- Market view
- Risk view
- Final balance

Inputs:
Market Analyst: {market_view}
Risk Agent: {risk_view}
Investment Score: {investment_score}/100
Investment Verdict: {investment_verdict}

Keep it easy to understand for a non-technical user.
"""
    return ask_llm(prompt)

# =========================================================
# CHECKS
# =========================================================
if model is None:
    st.error("Saved model not found at `models/real_estate_rf.pkl`.")
    st.stop()

if dataset_bundle is None:
    st.error("Dataset not found at `data/housing.csv`.")
    st.stop()

df, X_ref, stats = dataset_bundle

# =========================================================
# SESSION STATE
# =========================================================
if "analysis_context" not in st.session_state:
    st.session_state.analysis_context = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# HERO
# =========================================================
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">🏠 Neighborhood Intelligence AI</div>
        <div class="hero-sub">
            A California-focused real estate analysis system that combines a trained machine learning model,
            structured reasoning, and a local LLM for interpretable investment guidance.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.header("Neighborhood Inputs")
    st.caption("This model is trained on the California housing dataset, so best results come from California-like neighborhood inputs.")

    median_income = st.number_input(
        "Median Income ($)",
        min_value=10000,
        max_value=500000,
        value=80000,
        step=1000,
        format="%d",
        help="This is scaled internally to match the training data representation."
    )
    st.caption("Typical practical input: $20,000–$150,000.")

    housing_median_age = st.number_input(
        "Housing Median Age",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        format="%d",
        help="Median age of housing units in the neighborhood."
    )
    st.caption("Higher values can indicate more established areas, but age alone does not determine value.")

    total_rooms = st.number_input(
        "Total Rooms",
        min_value=100,
        max_value=100000,
        value=1500,
        step=10,
        format="%d",
        help="Neighborhood-level total, not a single house."
    )
    st.caption("Each row in this dataset represents a California block group, not one property.")

    total_bedrooms = st.number_input(
        "Total Bedrooms",
        min_value=50,
        max_value=100000,
        value=300,
        step=10,
        format="%d",
        help="Neighborhood-level total bedrooms."
    )
    st.caption("This should generally remain well below total rooms.")

    population = st.number_input(
        "Population",
        min_value=100,
        max_value=500000,
        value=20000,
        step=100,
        format="%d",
        help="Total people in the neighborhood area."
    )
    st.caption("Used to infer density and neighborhood crowding.")

    households = st.number_input(
        "Households",
        min_value=50,
        max_value=100000,
        value=500,
        step=10,
        format="%d",
        help="Neighborhood-level number of households."
    )
    st.caption("Used with population and room totals to derive area-level density signals.")

    st.markdown("### 📍 California Location Limits")

    latitude = st.number_input(
        f"Latitude ({stats['lat_min']:.2f} to {stats['lat_max']:.2f})",
        min_value=float(stats["lat_min"]),
        max_value=float(stats["lat_max"]),
        value=34.0,
        step=1.0,
        format="%.0f",
        help="Stay within the dataset range for more reliable results."
    )
    st.caption(f"For accurate results, keep latitude between {stats['lat_min']:.2f} and {stats['lat_max']:.2f}.")

    longitude = st.number_input(
        f"Longitude ({stats['lon_min']:.2f} to {stats['lon_max']:.2f})",
        min_value=float(stats["lon_min"]),
        max_value=float(stats["lon_max"]),
        value=-118.0,
        step=1.0,
        format="%.0f",
        help="Stay within the dataset range for more reliable results."
    )
    st.caption(f"For accurate results, keep longitude between {stats['lon_min']:.2f} and {stats['lon_max']:.2f}.")

    analyze = st.button("Analyze Neighborhood", use_container_width=True)

# =========================================================
# ANALYSIS
# =========================================================
if analyze:
    median_income_scaled = median_income / 10000
    rooms_per_household = total_rooms / households

    warnings, confidence, population_per_household, bedrooms_per_room = validate_inputs(
        population=population,
        households=households,
        total_rooms=total_rooms,
        total_bedrooms=total_bedrooms,
        latitude=latitude,
        longitude=longitude,
        stats=stats,
    )

    features = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income_scaled,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
    }])

    predicted_price = float(model.predict(features)[0])

    market_view = market_analyst_agent(
        predicted_price=predicted_price,
        median_income_scaled=median_income_scaled,
        rooms_per_household=rooms_per_household,
        latitude=latitude,
        longitude=longitude,
        stats=stats,
    )

    risk_view = risk_agent(
        predicted_price=predicted_price,
        population_per_household=population_per_household,
        bedrooms_per_room=bedrooms_per_room,
        stats=stats,
    )

    investment_score, investment_verdict, verdict_label, verdict_tone = investment_agent(
        predicted_price=predicted_price,
        median_income_scaled=median_income_scaled,
        rooms_per_household=rooms_per_household,
        population_per_household=population_per_household,
        stats=stats,
    )

    strengths, risks = build_strengths_and_risks(
        predicted_price=predicted_price,
        median_income=median_income,
        rooms_per_household=rooms_per_household,
        population_per_household=population_per_household,
        bedrooms_per_room=bedrooms_per_room,
        stats=stats,
    )

    final_summary = final_recommendation_llm(
        market_view=market_view,
        risk_view=risk_view,
        investment_score=investment_score,
        investment_verdict=investment_verdict,
    )

    explanation_text = reasoning_breakdown_llm(
        market_view=market_view,
        risk_view=risk_view,
        investment_score=investment_score,
        investment_verdict=investment_verdict,
    )

    context = {
        "predicted_price": int(predicted_price),
        "confidence": int(confidence),
        "warnings": warnings,
        "market_view": market_view,
        "risk_view": risk_view,
        "investment_score": int(investment_score),
        "investment_verdict": investment_verdict,
        "verdict_label": verdict_label,
        "verdict_tone": verdict_tone,
        "final_summary": final_summary,
        "explanation_text": explanation_text,
        "strengths": strengths,
        "risks": risks,
        "population_per_household": round(population_per_household, 2),
        "bedrooms_per_room": round(bedrooms_per_room, 2),
        "rooms_per_household": round(rooms_per_household, 2),
        "inputs": {
            "median_income": median_income,
            "housing_median_age": housing_median_age,
            "total_rooms": total_rooms,
            "total_bedrooms": total_bedrooms,
            "population": population,
            "households": households,
            "latitude": latitude,
            "longitude": longitude,
        },
    }

    st.session_state.analysis_context = context

    save_history({
        "predicted_price": int(predicted_price),
        "investment_score": int(investment_score),
        "confidence": int(confidence),
        "latitude": latitude,
        "longitude": longitude,
    })

# =========================================================
# RESULTS
# =========================================================
ctx = st.session_state.analysis_context

if ctx:
    st.subheader("📌 Final Recommendation")
    if ctx["verdict_tone"] == "success":
        st.success(ctx["verdict_label"])
    elif ctx["verdict_tone"] == "warning":
        st.warning(ctx["verdict_label"])
    else:
        st.error(ctx["verdict_label"])

    st.write(ctx["final_summary"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Estimated Value", f"${ctx['predicted_price']:,}")
    m2.metric("Confidence", f"{ctx['confidence']}%")
    m3.metric("Investment Score", f"{ctx['investment_score']}/100")

    if ctx["warnings"]:
        st.markdown(
            "<div class='warning-box'><b>Data Quality Warnings</b><br>" +
            "<br>".join([f"• {w}" for w in ctx["warnings"]]) +
            "</div>",
            unsafe_allow_html=True,
        )

    st.subheader("📍 Location Context")
    map_df = pd.DataFrame({"lat": [ctx["inputs"]["latitude"]], "lon": [ctx["inputs"]["longitude"]]})
    st.map(map_df)
    st.caption(
        f"Source limitation: this model is trained only on California housing data. "
        f"Latitude should stay between {stats['lat_min']:.2f} and {stats['lat_max']:.2f}, "
        f"and longitude between {stats['lon_min']:.2f} and {stats['lon_max']:.2f} for the most reliable results."
    )

    st.subheader("🧠 Deciding Factors For This Recommendation")
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>✅ Strengths</div>", unsafe_allow_html=True)
        if ctx["strengths"]:
            for item in ctx["strengths"]:
                st.markdown(
                    f"""
                    <div class="factor-good">
                        <div class="factor-title">{item['title']}</div>
                        <div>{item['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No strong positive drivers were detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>⚠️ Risks</div>", unsafe_allow_html=True)
        if ctx["risks"]:
            for item in ctx["risks"]:
                st.markdown(
                    f"""
                    <div class="factor-risk">
                        <div class="factor-title">{item['title']}</div>
                        <div>{item['text']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No major risk drivers were detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    top_left, top_right = st.columns(2)

    with top_left:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>📊 Market Analyst</div>", unsafe_allow_html=True)
        st.write(ctx["market_view"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>⚠️ Risk Agent</div>", unsafe_allow_html=True)
        st.write(ctx["risk_view"])
        st.markdown("</div>", unsafe_allow_html=True)

    with top_right:
        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>💼 Investment Agent</div>", unsafe_allow_html=True)
        st.write(f"**Score:** {ctx['investment_score']}/100")
        st.write(f"**Verdict:** {ctx['investment_verdict']}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
        st.markdown("<div class='soft-title'>📘 Why This Matters</div>", unsafe_allow_html=True)
        st.write(
            "- Income affects demand and willingness to pay.\n"
            "- Density affects livability and infrastructure pressure.\n"
            "- Space ratios help capture neighborhood comfort.\n"
            "- Pricing relative to the California dataset average affects valuation risk."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔍 How AI Reached This Decision"):
        st.write_stream(stream_words(ctx["explanation_text"]))

    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)
    st.markdown("<div class='soft-title'>📝 Feedback</div>", unsafe_allow_html=True)
    feedback = st.radio(
        "Was this analysis helpful?",
        ["yes", "no"],
        horizontal=True,
        key="feedback_radio",
    )
    if st.button("Save Feedback"):
        save_feedback({
            "predicted_price": ctx["predicted_price"],
            "investment_score": ctx["investment_score"],
            "confidence": ctx["confidence"],
            "feedback": feedback,
        })
        st.success("Feedback saved.")
    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HISTORY
# =========================================================
st.divider()
st.subheader("📁 Recent Analyses")

history = load_json_list(HISTORY_PATH)

if history:
    for item in reversed(history[-5:]):
        price = item.get("predicted_price", item.get("price", 0))
        score = item.get("investment_score", item.get("score", 0))
        confidence = item.get("confidence", 0)
        lat = item.get("latitude", "N/A")
        lon = item.get("longitude", "N/A")

        st.write(
            f"💰 ${price:,} | Score {score}/100 | Confidence {confidence}% | ({lat}, {lon})"
        )
else:
    st.caption("No saved analyses yet.")

# =========================================================
# CHAT
# =========================================================
st.divider()
st.subheader("💬 Ask AI")
st.caption("Ask follow-up questions about the current analysis.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask a question about the analysis...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.analysis_context:
        c = st.session_state.analysis_context
        prompt = f"""
You are a real estate AI assistant.

Answer the user's follow-up question based only on this active California neighborhood analysis:

Estimated Value: ${c['predicted_price']}
Confidence: {c['confidence']}%
Warnings: {c['warnings']}
Market Analyst: {c['market_view']}
Risk Agent: {c['risk_view']}
Investment Score: {c['investment_score']}/100
Investment Verdict: {c['investment_verdict']}
Reasoning: {c['explanation_text']}
Strengths: {c['strengths']}
Risks: {c['risks']}
Original Inputs: {c['inputs']}

User question:
{user_input}

Answer naturally, clearly, and stay grounded in this context.
"""
    else:
        prompt = f"""
You are a real estate AI assistant.
The user has not analyzed a neighborhood yet.

User question:
{user_input}

Tell them to run an analysis first, then ask follow-up questions.
"""

    assistant_response = ask_llm(prompt)
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    st.rerun()