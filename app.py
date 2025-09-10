# app.py – Streamlit ROI Predictor + Gemini AI Recommendations

import google.generativeai as genai
import os 
import streamlit as st
import pandas as pd
from model import load_model, predict   # <- helper functions


# ---------------------------------------------------
# 1. Set up Gemini API
# ---------------------------------------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


model = genai.GenerativeModel("gemini-1.5-flash")


def get_recommendations(df):
    """Generate AI marketing recommendations based on predicted ROI."""
    avg_roi = df["Predicted_ROI"].mean()
    avg_budget = df["Budget"].mean()
    avg_spend = df["Spend"].mean()

    prompt = f"""
    You are a marketing analytics assistant.

    ROI SUMMARY:
    - The average predicted ROI across campaigns is {avg_roi:.2f}.
    - The average budget across campaigns is {avg_budget:.2f}.
    - The average spend across campaigns is {avg_spend:.2f}.

    ACTIONABLE ADVICE:
    Based on the above ROI, budget and spend, give clear, actionable marketing recommendations.
    For example: if ROI is low, suggest optimizing spend or targeting.
    If ROI is high, suggest scaling campaigns or testing new audiences.
    Provide the recommendations as bullet points under clear subheadings for ROI SUMMARY and ACTIONABLE ADVICE.
    """
    response = model.generate_content(prompt)
    return response.text



# ---------------------------------------------------
# 2. Load model once & cache it
# ---------------------------------------------------
@st.cache_resource
def load_bundle():
    return load_model("roi_ensemble.joblib")
bundle = load_bundle()


# ---------------------------------------------------
# 3. UI
# ---------------------------------------------------
st.set_page_config(page_title="ADwiser - ROI Predictor", page_icon="📈", layout="wide")
st.title("📊 ADwiser - Marketing ROI Prediction")

st.info("Upload a CSV with the same columns used during training "
        "(Budget, Spend, Impressions, Clicks, Conversions, Revenue, CTR, etc.)")

uploaded = st.file_uploader("Choose a CSV file", type="csv")

if uploaded:
    try:
        df = pd.read_csv(uploaded)

        # quick column sanity check
        required = {
            "Budget", "Spend", "Impressions", "Clicks",
            "Conversions", "Revenue", "CTR", "conversion_rate", "Engagement_rate"
        }
        missing = required - set(df.columns)
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        st.success(f"✅ File loaded: {len(df)} rows")
        st.dataframe(df.head())

        # ---------------------------------------------------
        # Prediction + Recommendations
        # ---------------------------------------------------
        if st.button("🚀 Predict ROI"):
            with st.spinner("Working on recommendations..."):
                # Run predictions
                preds = predict(df, bundle)
                df["Predicted_ROI"] = preds

                # Show predictions
                st.subheader("📊 Predictions")
                st.dataframe(df.head(15))

                # AI Recommendations
                st.subheader("📌 AI Recommendations")
                try:
                    recs = get_recommendations(df)
                    st.markdown(recs)
                except Exception as e:
                    st.error(f"⚠️ Failed to fetch Gemini recommendations: {e}")

                # Download predictions
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Predictions",
                    data=csv,
                    file_name="roi_predictions.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    except Exception as e:
        st.error(f"Error reading CSV: {e}")


