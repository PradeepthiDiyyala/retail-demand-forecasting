from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
from predict import forecast
from inventory import inventory_policy

app = FastAPI(title='Retail Demand Forecasting API', version='1.0.0')


class Request(BaseModel):
    store_id: str = Field(min_length=1)
    product_id: str = Field(min_length=1)
    days: int = Field(ge=1, le=30, default=7)


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/forecast')
def get_forecast(req: Request):
    try:
        path = Path(__file__).resolve().parents[1] / 'data' / 'sales.csv'
        df = pd.read_csv(path)
        history = df[(df.store_id == req.store_id) & (df.product_id == req.product_id)].copy()
        if history.empty:
            raise HTTPException(404, 'Store/product combination not found')
        result = forecast(history, req.days)
        policy = inventory_policy(result.units_sold.values)
        return {
            'store_id': req.store_id,
            'product_id': req.product_id,
            'forecast': result[['date', 'units_sold']].to_dict('records'),
            'inventory_policy': policy
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
