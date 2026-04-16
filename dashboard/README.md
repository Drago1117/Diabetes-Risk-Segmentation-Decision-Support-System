# Diabetes Risk Dashboard - Deploy Instructions

**Local:** `python app_fixed.py` → http://127.0.0.1:8050/

**Render:** 
1. Push to GitHub
2. New Web Service → Connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `gunicorn dashboard.app_fixed:server`

**Features:**
- Interactive risk/segment prediction
- SHAP insights, recs
- Demo data (parquet optimized)

CRISP-DM complete. Render link ready for report!
