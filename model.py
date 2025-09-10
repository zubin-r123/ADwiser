# model.py
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# -------------------------------------------------------------------
# 1. Feature Engineering
# -------------------------------------------------------------------
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratios & log transforms – keep identical to your old pipeline."""
    df = df.copy()
    eps = 1e-6
    if "Revenue" in df and "Spend" in df:
        df["Efficiency_Score"] = df["Revenue"] / (df["Spend"] + eps)
    if "Conversions" in df and "Clicks" in df:
        df["Click_Quality"] = df["Conversions"] / (df["Clicks"] + eps)
    if "Spend" in df and "Budget" in df:
        df["Budget_Utilization"] = df["Spend"] / (df["Budget"] + eps)
    if "Impressions" in df and "Revenue" in df:
        df["Revenue_per_Impression"] = df["Revenue"] / (df["Impressions"] + eps)
    if "Spend" in df and "Revenue" in df:
        df["Cost_per_Revenue"] = df["Spend"] / (df["Revenue"] + eps)

    if "CTR" in df and "conversion_rate" in df:
        df["CTR_x_Conversion_Rate"] = df["CTR"] * df["conversion_rate"]
    if "Engagement_rate" in df and "CTR" in df:
        df["Engagement_x_CTR"] = df["Engagement_rate"] * df["CTR"]
    if "Spend" in df and "Clicks" in df:
        df["Spend_per_Click"] = df["Spend"] / (df["Clicks"] + eps)

    skewed = ["Budget","Spend","Impressions","Clicks","Conversions","Revenue"]
    for f in skewed:
        if f in df:
            df[f"log_{f}"] = np.log1p(df[f])
    return df

# -------------------------------------------------------------------
# 2. Train & Save
# -------------------------------------------------------------------
def train_and_save(
    path_excel="Marketing_Analytics_with_Budget_Cleaned.xlsx",
    save_path="roi_ensemble.joblib"
):
    # ---- Load ----
    df = pd.read_excel(path_excel)
    df = df.select_dtypes(exclude=["datetime64[ns]"])

    # ---- Engineer ----
    df = create_features(df)

    # ---- Drop extreme outliers in ROI ----
    roi_mean, roi_std = df["ROI"].mean(), df["ROI"].std()
    df = df[np.abs(df["ROI"] - roi_mean) <= 3 * roi_std]

    # ---- Encode categoricals ----
    categorical = df.select_dtypes(include="object").columns
    df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # ---- Features / Target ----
    X = df.drop("ROI", axis=1)
    y = np.sqrt(df["ROI"] + 1)   # √ transform

    # ---- Split & Scale ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=5, labels=False)
    )
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ---- Fit Models (quick defaults; tune offline if needed) ----
    xgb = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist"
    )
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42
    )

    xgb.fit(X_train_s, y_train)
    rf.fit(X_train_s, y_train)

    # ---- Blend & Evaluate ----
    blend = 0.7 * xgb.predict(X_test_s) + 0.3 * rf.predict(X_test_s)
    y_pred = np.square(blend) - 1
    y_test_actual = np.square(y_test) - 1

    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    r2 = r2_score(y_test_actual, y_pred)
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

    # ---- Persist Everything ----
    bundle = {
        "xgb": xgb,
        "rf": rf,
        "scaler": scaler,
        "features": list(X.columns)
    }
    joblib.dump(bundle, save_path)
    print(f"✅ Model bundle saved to {save_path}")

# -------------------------------------------------------------------
# 3. Load & Predict
# -------------------------------------------------------------------
def load_model(path="roi_ensemble.joblib"):
    return joblib.load(path)

def predict(df_new: pd.DataFrame, bundle) -> np.ndarray:
    """Takes raw DataFrame with same schema as training, returns predicted ROI."""
    df_proc = create_features(df_new)
    df_proc = pd.get_dummies(df_proc, drop_first=True)
    df_proc = df_proc.reindex(columns=bundle["features"], fill_value=0)
    scaled = bundle["scaler"].transform(df_proc)
    xgb_pred = bundle["xgb"].predict(scaled)
    rf_pred  = bundle["rf"].predict(scaled)
    blended  = 0.7 * xgb_pred + 0.3 * rf_pred
    return np.maximum(np.square(blended) - 1, 0)

# -------------------------------------------------------------------
if __name__ == "__main__":
    train_and_save()   # run once to build & store roi_ensemble.joblib
