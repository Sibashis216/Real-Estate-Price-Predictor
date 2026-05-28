# 🏠 Real-Estate — AI House Price Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Deployed-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)

**An end-to-end Machine Learning web application that predicts residential property prices across Bangalore using Gradient Boosting — from raw data to a live deployed REST API.**

[🚀 Live Demo](https://real-estate-price-predictor-983b.onrender.com/) &nbsp;&nbsp;  &nbsp;&nbsp; [🐛 Report Bug](../../issues)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [ML Pipeline](#-ml-pipeline)
- [Data Cleaning & Outlier Removal](#-data-cleaning--outlier-removal)
- [Model Comparison](#-model-comparison)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Docker Deployment](#-docker-deployment)
- [API Reference](#-api-reference)
- [Key Learnings](#-key-learnings)

---

## 🎯 Project Overview

Real estate pricing in Bangalore is highly non-linear — a 2BHK in Koramangala can cost 3× more than a similar flat in Kengeri. This project tackles that complexity by:

- Ingesting and cleaning a realistic Bangalore housing dataset (8,000+ listings)
- Applying **domain-aware outlier removal** using business logic rules
- Training and comparing **5 regression models** with cross-validation
- Serving predictions via a **Flask REST API** with a polished UI
- Containerizing everything with **Docker** for reproducible, cloud-ready deployment

> **Problem Type:** Supervised Regression  
> **Target Variable:** Property price in Indian Rupees (Lakhs)  
> **Best Model:** Gradient Boosting — R² = 0.9732, MAE = 5.69 Lakhs

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data Processing | `Pandas`, `NumPy` | Cleaning, transformation, feature engineering |
| Machine Learning | `Scikit-Learn` | Model training, cross-validation, evaluation |
| API Server | `Flask` | REST API endpoints |
| Production Server | `Gunicorn` | Multi-worker WSGI server |
| Frontend | HTML5, CSS3, Vanilla JS | Interactive prediction UI |
| Containerization | `Docker` (multi-stage) | Reproducible builds |
| Orchestration | `Docker Compose` | Local multi-service management |
| Deployment | `Render.com` | Free cloud hosting with HTTPS |

---

## 🔬 ML Pipeline

The pipeline runs fully automated — from raw CSV to serialized model — in a single script (`train_model.py`).

```
Raw CSV (8,000 rows)
       │
       ▼
 ┌─────────────────────┐
 │   1. Data Cleaning  │  ← Extract BHK, convert sqft ranges, drop nulls
 └─────────────────────┘
       │
       ▼
 ┌──────────────────────────┐
 │  2. Feature Engineering  │  ← price_per_sqft, location cleaning
 └──────────────────────────┘
       │
       ▼
 ┌──────────────────────────────┐
 │  3. Outlier Removal (4 Rules)│  ← Business logic + statistical filtering
 └──────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────────┐
 │  4. Feature Preparation          │  ← One-hot encode locations (30 dummies)
 │     34 total features            │
 └──────────────────────────────────┘
       │
       ▼
 ┌──────────────────────────────┐
 │  5. Train 5 Models + Compare │  ← Select best by R²
 └──────────────────────────────┘
       │
       ▼
 ┌───────────────────────────────────────┐
 │  6. Save Artifacts                    │
 │   house_price_model.pkl               │
 │   feature_columns.json                │
 │   locations.json · metadata.json      │
 └───────────────────────────────────────┘
```

---

## 🧹 Data Cleaning & Outlier Removal

This was the most critical phase. Raw real estate data contains systematic errors that destroy model accuracy if ignored.

### Cleaning Steps

| Step | Action | Rationale |
|---|---|---|
| BHK extraction | Parse `"2 BHK"` → `2` | Model needs numeric input |
| Sqft conversion | `"1200 - 1500"` → `1350` (midpoint) | Range listings are common |
| Null removal | Drop rows missing sqft / bath / BHK / price / location | ~418 rows removed |
| Balcony imputation | Fill nulls with column median | Minor missing data |

### Outlier Removal — 4 Business Logic Rules

```python
# Rule 1 — Minimum livable area per bedroom
df = df[df["total_sqft_num"] >= df["bhk"] * 300]
# A 2BHK below 600 sqft is physically implausible

# Rule 2 — Bathroom cap
df = df[df["bath"] <= df["bhk"] + 2]
# 5 bathrooms in a 2BHK is a data error

# Rule 3 — Statistical: price/sqft within ±1σ per location
# Removes extreme luxury outliers and data entry errors per neighbourhood

# Rule 4 — BHK pricing logic
# A 2BHK in the same location should not be cheaper than a 1BHK
# (indicates a misclassified or fraudulent listing)
```

> **Impact:** These 4 rules reduced noise from 7,582 → 3,953 clean records, improving model R² from ~0.81 to **0.97**.

---

## 📊 Model Comparison

All models were evaluated on an 80/20 train-test split with the same feature set.

| Model | R² Score | MAE (Lakhs) | RMSE (Lakhs) |
|---|---|---|---|
| Linear Regression | 0.9515 | 10.07 | 23.08 |
| Ridge Regression | 0.9515 | 9.51 | 23.07 |
| Lasso Regression | 0.9217 | 16.31 | 29.33 |
| Random Forest | 0.9622 | 6.22 | 20.37 |
| **Gradient Boosting** ✅ | **0.9732** | **5.69** | **17.15** |

### Why Gradient Boosting Won

Gradient Boosting builds an ensemble of weak learners **sequentially** — each new tree corrects the residual errors of the previous one. This makes it exceptionally good at capturing the non-linear, location-driven price patterns in this dataset.

### Cross-Validation (5-Fold)

```
Fold scores: [0.9644, 0.9330, 0.8772, 0.8325, 0.9166]
Mean CV R²:  0.9047 ± 0.046
```

The slight drop from test R² (0.97) to CV R² (0.90) is expected — it reflects genuine variance across different data splits and confirms the model generalizes well.

---

## 📁 Project Structure

```
bangalore-price-predictor/
│
├── data/
│   ├── generate_data.py          # Synthetic dataset generator
│   └── bangalore_house_prices.csv
│
├── model/                        # Generated after training
│   ├── house_price_model.pkl     # Serialized Gradient Boosting model
│   ├── feature_columns.json      # Ordered feature list (34 features)
│   ├── locations.json            # 30 Bangalore locations
│   └── metadata.json             # Model metrics & training info
│
├── templates/
│   └── index.html                # Full-stack UI (HTML + CSS + JS)
│
├── app.py                        # Flask REST API
├── train_model.py                # End-to-end ML pipeline
├── requirements.txt
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml
└── README.md
```

---

## ⚡ Getting Started

### Prerequisites
- Python 3.9+
- pip

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/bangalore-price-predictor.git
cd bangalore-price-predictor

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset
python data/generate_data.py

# 5. Train the model (runs full pipeline, prints metrics)
python train_model.py

# 6. Start the Flask server
python app.py
```

Open → **http://localhost:5000**

---

## 🐳 Docker Deployment

The Dockerfile uses a **multi-stage build** — stage 1 installs dependencies, stage 2 copies only what's needed into a slim runtime image. The model is trained **at build time** inside the container.

```bash
# Build image (trains model inside Docker)
docker build -t bangalore-price-predictor .

# Run container
docker run -p 5000:5000 bangalore-price-predictor

# OR with Docker Compose
docker-compose up --build
```

### Deploy to Render (Free, HTTPS, no credit card)

```
1. Push repo to GitHub
2. render.com → New → Web Service → Connect repo
3. Runtime: Docker | Port: 5000 | Instance: Free
4. Deploy → live URL in ~5 minutes
```

---

## 🔌 API Reference

### `POST /predict`

Predict the price of a property.

**Request Body:**
```json
{
  "location": "Koramangala",
  "sqft": 1200,
  "bhk": 2,
  "bath": 2,
  "balcony": 1
}
```

**Response:**
```json
{
  "price_lakhs": 100.84,
  "price_crores": 1.008,
  "price_per_sqft": 8403,
  "location": "Koramangala",
  "sqft": 1200,
  "bhk": 2,
  "bath": 2,
  "balcony": 1
}
```

### Other Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/predict` | POST | Price prediction |
| `/api/locations` | GET | List of all 30 locations |
| `/api/metadata` | GET | Model metrics and info |
| `/health` | GET | Health check |

---

## 💡 Key Learnings

**1. Data quality > model complexity**
The jump from a messy dataset (R²≈0.81) to a clean one (R²≈0.97) was achieved purely through better outlier removal — not a fancier model.

**2. Domain knowledge is irreplaceable**
Statistical outlier methods alone aren't enough. Rules like "min 300 sqft per bedroom" require understanding how real estate actually works.

**3. Engineering matters as much as modelling**
A 97% accurate model locked in a notebook is useless. Wrapping it in a versioned API, containerizing it, and deploying it transforms it into a product.

**4. Cross-validation prevents overconfidence**
The gap between test R² (0.97) and CV R² (0.90) is a healthy reminder to never trust a single train-test split.

---

## 🔮 Future Improvements

- [ ] Add geospatial coordinates (lat/lng) for richer location encoding
- [ ] Include features: floor number, building age, proximity to metro
- [ ] Integrate XGBoost / LightGBM for comparison
- [ ] Add SHAP values for model explainability
- [ ] Build automated retraining pipeline with new data
- [ ] Add prediction confidence intervals
- [ ] Implement A/B testing between model versions

---

## 👤 Author

**Sibashis Patnaik**  
Aspiring Data Scientist 
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/sibashispatnaik2000)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/Sibashis216?tab=repositories)


---

<div align="center">
<sub>Built with Python, Scikit-Learn, Flask & Docker &nbsp;|&nbsp; Deployed on Render</sub>
</div>
