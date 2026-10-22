from pathlib import Path
import joblib
import pandas as pd
from features import create_features, prepare_xy

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = joblib.load(ROOT / 'models' / 'demand_model.joblib')


def forecast(history, days=7):
    df = history.copy()
    df['date'] = pd.to_datetime(df.date)
    forecasts = []
    for _ in range(days):
        next_date = df.date.max() + pd.Timedelta(days=1)
        base = df[df.date == df.date.max()].copy()
        base['date'] = next_date
        base['units_sold'] = 0
        base['promotion'] = 0
        base['holiday'] = int(next_date.month == 12 and next_date.day >= 15)
        combined = pd.concat([df, base], ignore_index=True)
        features = create_features(combined)
        row = features[features.date == next_date].copy()
        x, _ = prepare_xy(row)
        x = x.reindex(columns=BUNDLE['features'], fill_value=0)
        prediction = max(0, float(BUNDLE['model'].predict(x)[0]))
        base['units_sold'] = round(prediction, 2)
        forecasts.append(base.iloc[0])
        df = pd.concat([df, base], ignore_index=True)
    return pd.DataFrame(forecasts)


def forecast_from_csv(path, days=7):
    return forecast(pd.read_csv(path), days)


if __name__ == '__main__':
    result = forecast_from_csv(ROOT / 'data' / 'sales.csv', 7)
    print(result[['date', 'store_id', 'product_id', 'units_sold']].head(12).to_string(index=False))
