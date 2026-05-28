# 🏠 BengaluruEstimate — AI House Price Predictor

A production-grade Machine Learning project that predicts Bangalore housing prices using Gradient Boosting, served via Flask, containerized with Docker.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Algorithm | Gradient Boosting |
| R² Score (Test) | 0.9732 |
| Mean Absolute Error | ~5.69 Lakhs |
| RMSE | ~17.15 Lakhs |
| 5-Fold CV R² | 0.9047 ± 0.046 |

---

## 🗂️ Project Structure

```
bangalore-price-predictor/
├── data/
│   ├── generate_data.py       # Dataset generation script
│   └── bangalore_house_prices.csv
├── model/
│   ├── house_price_model.pkl  # Trained model (generated)
│   ├── feature_columns.json   # Feature list (generated)
│   ├── locations.json         # Location list (generated)
│   └── metadata.json          # Model metrics (generated)
├── templates/
│   └── index.html             # Beautiful UI
├── static/                    # Static assets
├── app.py                     # Flask API
├── train_model.py             # ML pipeline
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🚀 Running Locally (3 Options)

### Option 1 — Python directly (quickest)

```bash
# 1. Clone / download the project
cd bangalore-price-predictor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate dataset & train model
python data/generate_data.py
python train_model.py

# 5. Start the Flask server
python app.py
```

Open → http://localhost:5000

---

### Option 2 — Docker (recommended)

```bash
# Build the image (trains model inside the container)
docker build -t bangalore-price-predictor .

# Run the container
docker run -p 5000:5000 bangalore-price-predictor
```

Open → http://localhost:5000

---

### Option 3 — Docker Compose

```bash
docker-compose up --build
```

Open → http://localhost:5000

---

## 🌐 Deployment Options

### ▶ Render (free tier — easiest)

1. Push code to GitHub
2. Go to https://render.com → New → Web Service
3. Connect your GitHub repo
4. Set:
   - **Environment**: Docker
   - **Dockerfile path**: `./Dockerfile`
   - **Port**: `5000`
5. Click **Deploy**

Your live URL: `https://your-app-name.onrender.com`

---

### ▶ Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
railway domain   # Get your public URL
```

---

### ▶ Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

fly auth login
fly launch          # Detects Dockerfile automatically
fly deploy
```

---

### ▶ AWS EC2

```bash
# On your EC2 instance (Ubuntu)
sudo apt update && sudo apt install docker.io docker-compose -y
sudo systemctl start docker

git clone <your-repo>
cd bangalore-price-predictor
sudo docker-compose up -d

# Access via: http://<EC2-Public-IP>:5000
# Open port 5000 in EC2 Security Groups
```

---

### ▶ Google Cloud Run

```bash
# Build & push to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT/house-predictor

# Deploy to Cloud Run
gcloud run deploy house-predictor \
  --image gcr.io/YOUR_PROJECT/house-predictor \
  --platform managed \
  --port 5000 \
  --allow-unauthenticated
```

---

## 🔌 API Reference

### POST `/predict`

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
  "price_lakhs": 87.34,
  "price_crores": 0.873,
  "price_per_sqft": 7278,
  "location": "Koramangala",
  "sqft": 1200,
  "bhk": 2,
  "bath": 2,
  "balcony": 1
}
```

### GET `/api/locations` — List of all locations
### GET `/api/metadata` — Model info and metrics
### GET `/health` — Health check

---

## 🧹 Data Cleaning & Outlier Rules

| Rule | Description |
|---|---|
| Missing values | Drop rows with null sqft, bath, BHK, price, location |
| Sqft range | Convert "1200 - 1500" → 1350 (midpoint) |
| Min sqft/BHK | Remove if sqft < BHK × 300 (e.g., 2BHK must be ≥ 600 sqft) |
| Bath limit | Remove if bath > BHK + 2 |
| Price/sqft σ | Per-location: remove outliers beyond ±1 std deviation |
| BHK logic | 2BHK price should not be lower than 1BHK in same location |

---

## 🛠 Tech Stack

- **Python 3.11**
- **Pandas** — data manipulation
- **Scikit-Learn** — ML (GradientBoostingRegressor, LinearRegression, etc.)
- **Flask** — REST API server
- **Gunicorn** — production WSGI server
- **Docker** — containerization
- **HTML/CSS/JS** — frontend UI
