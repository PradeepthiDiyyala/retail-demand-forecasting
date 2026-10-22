# Retail Demand Forecasting & Inventory Optimization

End-to-end machine-learning application for forecasting retail demand and turning forecasts into inventory decisions.

## Features
- Reproducible retail sales data generation
- Time-series feature engineering with lags and rolling statistics
- Calendar, promotion and seasonal features
- Chronological train/test validation
- Random Forest and gradient boosting forecasting models
- MAE, RMSE and MAPE evaluation
- Multi-step demand forecasting
- Safety-stock and reorder-point calculations
- Stockout-risk indicators
- FastAPI REST service
- Streamlit forecasting dashboard
- Automated tests

## Run
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
python scripts/generate_data.py
python scripts/train.py
python scripts/forecast.py
uvicorn api.main:app --reload
streamlit run dashboard/app.py
```

The repository uses synthetic retail data so the complete workflow is reproducible without proprietary sales data.
