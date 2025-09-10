# 📊 ADwiser – Marketing ROI Predictor + Gemini AI Recommendations

ADwiser is a Streamlit web application that predicts the **Return on Investment (ROI)** of marketing campaigns and generates **AI-powered recommendations** using Google Gemini (Generative AI).

It allows marketers to upload campaign data, instantly predict ROI for each row, and get actionable advice on how to optimize budget and spend.

---

## ✨ Features

- **ROI Prediction:** Upload your marketing dataset and get predicted ROI values using a blended RandomForest + XGBoost model.
- **AI Recommendations:** Automatically generates tailored marketing advice with Google Gemini, based on predicted ROI, average budget, and average spend.
- **Downloadable Results:** Download your enriched dataset (with predicted ROI) as a CSV file.
- **Interactive UI:** Built with Streamlit for a clean, responsive interface.

---

## 🖥 Live Demo

[![Open in Streamlit](https://adwiser.streamlit.app/)

---

## 📂 Project Structure

adwiser/

app.py # Main Streamlit app

model.py # Feature engineering, load_model, predict helpers

roi_ensemble.joblib # Pretrained blended model (RF + XGB)

requirements.txt # Python dependencies

README.md # Project documentation


## ⚙️ Requirements

Python 3.11+ is recommended.

Main libraries:
- Streamlit
- Pandas / NumPy
- Scikit-learn
- XGBoost
- Joblib
- Google-GenerativeAI

All dependencies are listed in `requirements.txt`.
they can be installed via a 'pip install requirements.txt command'

---

## Implementation
- For running the frontend the UI present in the app.py file has to be initiated. to run on a local system
  run the command 'streamlit run app.py' in the terminal after cloning a copy of this repository will all dependencies installed.

- on loading of UI follow the below instructions:

- Input CSV — expected columns
 
During runtime the app checks for the presence of a set of required columns. Provide a CSV with at least these columns (same names, case-sensitive):
 
Budget, Spend, Impressions, Clicks, Conversions, Revenue, CTR, conversion_rate, Engagement_rate etc.
 
 
model.py also engineers features from columns like Revenue, Spend, Clicks, Impressions, Conversions and creates log transforms, ratios and interaction features. If your CSV has extra columns they will be ignored; if columns are missing, the app will show an error.

 - Working of the model
 
create_features() adds derived metrics (efficiency, per-impression/revenue costs, log transforms on skewed numeric columns).
 
The target ROI is square-root transformed during training (y = sqrt(ROI + 1)) to stabilize variance; predictions are inverted before returning results.
 
Two models trained and blended: XGBoost (70%) + RandomForest (30%). The bundle stored with joblib contains both models, the fitted RobustScaler, and the final feature list used for alignment on inference
 
Acknowledgements & references
Core model + pipeline: model.py
Streamlit UI + Gemini integration: app.py
Dependencies: requirements.txt
 

