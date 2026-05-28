"""
Bangalore House Price Predictor - ML Pipeline
Covers: Data cleaning, outlier removal, feature engineering, model training
"""
import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("BANGALORE HOUSE PRICE PREDICTOR - ML PIPELINE")
print("=" * 60)

df = pd.read_csv("data/bangalore_house_prices.csv")
print(f"\n[1] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"    Columns: {list(df.columns)}")

# ─────────────────────────────────────────────
# 2. DATA CLEANING
# ─────────────────────────────────────────────
print("\n[2] DATA CLEANING")

# --- Extract BHK from 'size' column ---
def extract_bhk(size_str):
    try:
        tokens = str(size_str).split()
        return int(tokens[0])
    except:
        return np.nan

df["bhk"] = df["size"].apply(extract_bhk)
print(f"    BHK extracted. Unique values: {sorted(df['bhk'].dropna().unique())}")

# --- Convert total_sqft to numeric (handle ranges like '1200 - 1500') ---
def convert_sqft(sqft_str):
    tokens = str(sqft_str).split("-")
    if len(tokens) == 2:
        try:
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        except:
            return np.nan
    try:
        return float(sqft_str)
    except:
        return np.nan

df["total_sqft_num"] = df["total_sqft"].apply(convert_sqft)
print(f"    total_sqft converted. NaN count: {df['total_sqft_num'].isna().sum()}")

# --- Drop rows missing critical fields ---
initial_rows = len(df)
df.dropna(subset=["total_sqft_num", "bath", "bhk", "price", "location"], inplace=True)
print(f"    Dropped {initial_rows - len(df)} rows with missing critical fields → {len(df)} remaining")

# --- Fill missing balcony with median ---
df["balcony"] = df["balcony"].fillna(df["balcony"].median())

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n[3] FEATURE ENGINEERING")

df["price_per_sqft"] = (df["price"] * 100000) / df["total_sqft_num"]
print(f"    price_per_sqft stats:\n"
      f"      mean={df['price_per_sqft'].mean():.0f}, "
      f"std={df['price_per_sqft'].std():.0f}, "
      f"min={df['price_per_sqft'].min():.0f}, "
      f"max={df['price_per_sqft'].max():.0f}")

# ─────────────────────────────────────────────
# 4. OUTLIER REMOVAL
# ─────────────────────────────────────────────
print("\n[4] OUTLIER REMOVAL")
before = len(df)

# Business Rule 1: Min 300 sqft per BHK (a 2BHK below 600 sqft is suspicious)
df = df[df["total_sqft_num"] >= df["bhk"] * 300]
print(f"    Rule 1 (min 300 sqft/BHK): removed {before - len(df)} rows → {len(df)} remaining")

# Business Rule 2: Bathrooms should not exceed BHK + 2
before = len(df)
df = df[df["bath"] <= df["bhk"] + 2]
print(f"    Rule 2 (bath ≤ BHK+2): removed {before - len(df)} rows → {len(df)} remaining")

# Business Rule 3: Remove extreme price_per_sqft outliers per location
# (keep within mean ± 1 std dev per location group)
def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby("location"):
        mean = np.mean(subdf["price_per_sqft"])
        std = np.std(subdf["price_per_sqft"])
        reduced = subdf[(subdf["price_per_sqft"] > (mean - std)) &
                        (subdf["price_per_sqft"] <= (mean + std))]
        df_out = pd.concat([df_out, reduced], ignore_index=True)
    return df_out

before = len(df)
df = remove_pps_outliers(df)
print(f"    Rule 3 (price_per_sqft ±1σ per location): removed {before - len(df)} rows → {len(df)} remaining")

# Business Rule 4: 2 BHK price should be lower than 3 BHK in same location
def remove_bhk_outliers(df):
    exclude_indices = []
    for loc, loc_df in df.groupby("location"):
        bhk_stats = {}
        for bhk, bhk_df in loc_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": np.mean(bhk_df["price_per_sqft"]),
                "std": np.std(bhk_df["price_per_sqft"]),
                "count": len(bhk_df)
            }
        for bhk, bhk_df in loc_df.groupby("bhk"):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats["count"] > 5:
                exclude_indices += bhk_df[
                    bhk_df["price_per_sqft"] < (stats["mean"])
                ].index.tolist()
    return df.drop(exclude_indices, errors="ignore")

before = len(df)
df = remove_bhk_outliers(df)
print(f"    Rule 4 (BHK pricing logic): removed {before - len(df)} rows → {len(df)} remaining")

# ─────────────────────────────────────────────
# 5. PREPARE FEATURES
# ─────────────────────────────────────────────
print("\n[5] PREPARING FEATURES")

# Keep only locations with enough data
loc_counts = df["location"].value_counts()
df["location_clean"] = df["location"].apply(
    lambda x: x if loc_counts[x] >= 10 else "other"
)

# One-hot encode locations
location_dummies = pd.get_dummies(df["location_clean"], prefix="loc", drop_first=False)
df = pd.concat([df, location_dummies], axis=1)

# Feature columns
location_cols = [c for c in df.columns if c.startswith("loc_")]
feature_cols = ["total_sqft_num", "bath", "bhk", "balcony"] + location_cols

X = df[feature_cols]
y = df["price"]

print(f"    Features: {len(feature_cols)} total ({len(location_cols)} location dummies)")
print(f"    Dataset ready: X={X.shape}, y={y.shape}")

# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n[6] Split: {len(X_train)} train / {len(X_test)} test")

# ─────────────────────────────────────────────
# 7. MODEL SELECTION
# ─────────────────────────────────────────────
print("\n[7] MODEL COMPARISON")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10),
    "Lasso": Lasso(alpha=1, max_iter=10000),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1,
                                                    max_depth=4, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"mae": mae, "rmse": rmse, "r2": r2, "model": model}
    print(f"    {name:25s} → R²={r2:.4f}  MAE={mae:.2f}L  RMSE={rmse:.2f}L")

# Pick best model by R²
best_name = max(results, key=lambda k: results[k]["r2"])
best_model = results[best_name]["model"]
print(f"\n    ✓ Best model: {best_name} (R²={results[best_name]['r2']:.4f})")

# ─────────────────────────────────────────────
# 8. CROSS VALIDATION
# ─────────────────────────────────────────────
print("\n[8] CROSS VALIDATION (5-fold on best model)")
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring="r2")
print(f"    CV R² scores: {[round(s, 4) for s in cv_scores]}")
print(f"    Mean CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─────────────────────────────────────────────
# 9. SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("\n[9] SAVING MODEL ARTIFACTS")
os.makedirs("model", exist_ok=True)

# Save model
with open("model/house_price_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save feature columns
with open("model/feature_columns.json", "w") as f:
    json.dump(list(X.columns), f)

# Save location list
locations_list = sorted(df["location_clean"].unique().tolist())
with open("model/locations.json", "w") as f:
    json.dump(locations_list, f)

# Save model metadata
metadata = {
    "model_name": best_name,
    "r2_score": round(results[best_name]["r2"], 4),
    "mae_lakhs": round(results[best_name]["mae"], 2),
    "rmse_lakhs": round(results[best_name]["rmse"], 2),
    "cv_r2_mean": round(cv_scores.mean(), 4),
    "cv_r2_std": round(cv_scores.std(), 4),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "feature_count": len(feature_cols),
    "location_count": len(locations_list),
    "all_model_results": {k: {"r2": round(v["r2"], 4), "mae": round(v["mae"], 2)}
                          for k, v in results.items()}
}
with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"    ✓ Model saved: model/house_price_model.pkl")
print(f"    ✓ Feature cols saved: model/feature_columns.json")
print(f"    ✓ Locations saved: model/locations.json")
print(f"    ✓ Metadata saved: model/metadata.json")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE!")
print(f"Final model: {best_name}")
print(f"Test R²: {results[best_name]['r2']:.4f}")
print(f"Test MAE: {results[best_name]['mae']:.2f} Lakhs")
print("=" * 60)
