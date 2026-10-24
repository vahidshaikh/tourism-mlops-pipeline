"""
Streamlit app - Wellness Tourism Package purchase prediction.

Loads the model that the GitHub Actions pipeline committed to this folder,
collects the customer details into a one-row DataFrame and shows the
predicted purchase probability.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_tourism_model_v1.joblib"
METRICS_PATH = APP_DIR / "model_metrics.json"
DEFAULT_THRESHOLD = 0.50

st.set_page_config(page_title="Wellness Package Predictor", page_icon="🧳", layout="centered")


# ----------------------------------------------------------------------------
# Load model (cached so it is only read once per server session)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


if not MODEL_PATH.exists():
    st.error(
        f"Model file `{MODEL_PATH.name}` was not found. Run the GitHub Actions "
        "pipeline first - it trains the model and commits it to this folder."
    )
    st.stop()

model = load_model()
metrics = load_metrics()
threshold = float(metrics.get("classification_threshold", DEFAULT_THRESHOLD))

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("🧳 Wellness Tourism Package Predictor")
st.write(
    "Enter a customer's details to estimate how likely they are to purchase the "
    "**Wellness Tourism Package** - so the sales team can prioritise who to contact."
)

# ----------------------------------------------------------------------------
# Inputs (ranges follow the training data)
# ----------------------------------------------------------------------------
st.subheader("Customer details")
c1, c2 = st.columns(2)

with c1:
    age = st.number_input("Age", min_value=18, max_value=80, value=36)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced", "Unmarried"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly income", min_value=1000, max_value=150000,
                                     value=22000, step=500)
    city_tier = st.selectbox("City tier", [1, 2, 3], help="Tier 1 > Tier 2 > Tier 3")
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    passport = st.selectbox("Holds a valid passport?", ["No", "Yes"])

with c2:
    num_persons = st.number_input("Number of persons visiting", min_value=1, max_value=10, value=3)
    num_children = st.number_input("Children (under 5) visiting", min_value=0, max_value=5, value=1)
    num_trips = st.number_input("Average trips per year", min_value=1, max_value=25, value=3)
    property_star = st.selectbox("Preferred property star rating", [3, 4, 5])
    type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
    product_pitched = st.selectbox(
        "Product pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
        help="In the training data each product is pitched to one designation "
             "(Basic -> Executive, Standard -> Senior Manager, Deluxe -> Manager, "
             "Super Deluxe -> AVP, King -> VP).")
    pitch_duration = st.number_input("Duration of pitch (minutes)", min_value=1, max_value=130, value=15)
    num_followups = st.number_input("Number of follow-ups", min_value=0, max_value=10, value=4)
    pitch_score = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)

# One-row DataFrame with exactly the columns the model was trained on
input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": pitch_duration,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": num_persons,
    "NumberOfFollowups": num_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": num_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": num_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Show the input sent to the model"):
    st.dataframe(input_df)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_df)[0, 1])
    will_buy = probability >= threshold

    st.subheader("Result")
    st.metric("Probability of purchase", f"{probability:.1%}")
    st.progress(min(max(probability, 0.0), 1.0))

    if will_buy:
        st.success("✅ **Likely to purchase** - worth prioritising for outreach.")
    else:
        st.warning("⚠️ **Unlikely to purchase** - lower priority for this package.")
    st.caption(f"Decision threshold: {threshold:.0%} (customers at or above it are flagged as likely buyers).")

# ----------------------------------------------------------------------------
# Sidebar - model information
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("About the model")
    st.write("XGBoost classifier trained and versioned by the GitHub Actions pipeline.")
    test_metrics = metrics.get("test_metrics")
    if test_metrics:
        st.subheader("Hold-out test performance")
        st.write(f"- Recall: **{test_metrics['recall']:.2f}**")
        st.write(f"- Precision: **{test_metrics['precision']:.2f}**")
        st.write(f"- F1: **{test_metrics['f1']:.2f}**")
        st.write(f"- ROC-AUC: **{test_metrics['roc_auc']:.2f}**")
    if metrics.get("trained_at_utc"):
        st.caption(f"Trained: {metrics['trained_at_utc']} UTC")
