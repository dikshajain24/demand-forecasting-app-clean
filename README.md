# 📈 Retail Demand Forecasting — Streamlit Demo

**Author:** Diksha Jain  
**Repo:** demand-forecasting-app-clean  

---

## 🔗 Live demo
👉 [Try the app here] https://demand-forecasting-app-clean-9snmibdgpvyqqgzsvx3cbf.streamlit.app/ 


---

## 📌 Project summary
This project demonstrates a complete **retail demand forecasting pipeline**:  

- **Dataset**: [Rossmann Store Sales (Kaggle)](https://www.kaggle.com/competitions/rossmann-store-sales)  
- **ETL**: Clean raw store-level sales data  
- **Feature Engineering**: Lag features, rolling averages, calendar variables, promotions  
- **Model**: LightGBM regression baseline  
- **Forecasting**: Next 7-day and 14-day sales forecasts per store  
- **App**: Interactive Streamlit dashboard with historical vs. forecast plots + downloadable CSVs  

---

## 📂 Repo structure
app/forecast_app.py # Streamlit dashboard
src/ # ETL, features, training, prediction scripts
models/ # trained model, metrics.json, forecast CSVs
data/processed/ # processed dataset (small sample included)
data/raw/ # raw Kaggle data (ignored via .gitignore)
requirements.txt
README.md
.gitignore

yaml

---

## ⚙️ Local setup and usage

### 1. Clone repo
```bash
git clone https://github.com/dikshajain24/demand-forecasting-app-clean.git
cd demand-forecasting-app-clean
2. Create & activate virtual environment
bash
Copy code
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
3. Install dependencies
bash
pip install -r requirements.txt
4. Download dataset (if reproducing training)
Get train.csv and store.csv from Kaggle’s Rossmann Store Sales.
Place them in:
bash
data/raw/train.csv
data/raw/store.csv
5. Run ETL + feature engineering
bash
python src/etl.py
python src/features.py
6. Train model and generate predictions
bash
python src/train.py
python src/predict.py --days 7
python src/predict.py --days 14
7. Run the Streamlit app
bash
streamlit run app/forecast_app.py
🌐 Deployment (Streamlit Cloud)
This repo is deployed on Streamlit Community Cloud:

Repo: GitHub → dikshajain24/demand-forecasting-app-clean
Branch: main
Entry file: app/forecast_app.py
Dependencies: from requirements.txt


📸 Demo Screenshot
Here’s a preview of the dashboard <img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/60a0f74f-dbdb-4c97-af71-116781285bd8" />



📝 Notes
Raw Kaggle data (data/raw/) is excluded from the repo via .gitignore.

A small processed dataset (data/processed/features.parquet) and precomputed forecasts (models/next_7day_preds.csv, models/next_14day_preds.csv) are included so the Streamlit app runs without needing heavy training in the cloud.
If lightgbm fails to install on Streamlit Cloud, pin version in requirements.txt:
ini
lightgbm==3.3.2

👤 Contact
GitHub: dikshajain24
LinkedIn: Diksha Jain
Email: dikshajain2406@gmail.com
