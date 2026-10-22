import numpy as np
import pandas as pd
from pathlib import Path


def generate_sales(days=730, stores=8, products=12, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2024-01-01', periods=days, freq='D')
    rows = []
    for store in range(1, stores + 1):
        store_factor = rng.uniform(0.75, 1.35)
        for product in range(1, products + 1):
            base = rng.uniform(18, 90)
            trend = rng.uniform(-0.015, 0.03)
            phase = rng.uniform(0, 6.28)
            price = rng.uniform(8, 120)
            for i, date in enumerate(dates):
                dow = date.dayofweek
                seasonal = (
                    1 + 0.18 * np.sin(2 * np.pi * dow / 7 + phase)
                    + 0.12 * np.sin(2 * np.pi * i / 365 + phase)
                )
                holiday = int(
                    (date.month == 12 and date.day >= 15)
                    or (date.month == 11 and date.day >= 20)
                )
                promotion = int(rng.random() < (0.12 + 0.18 * holiday))
                temperature = 20 + 12 * np.sin(2 * np.pi * (i % 365) / 365) + rng.normal(0, 2)
                demand = base * store_factor * (1 + trend * i) * seasonal
                demand *= 1 + 0.25 * promotion + 0.12 * holiday
                demand *= 1 + 0.025 * (temperature - 20) + rng.normal(0, 0.16)
                units = max(0, int(round(demand)))
                rows.append([
                    date, f'S{store:02d}', f'P{product:02d}', price,
                    units, promotion, holiday, temperature
                ])
    return pd.DataFrame(rows, columns=[
        'date', 'store_id', 'product_id', 'price', 'units_sold',
        'promotion', 'holiday', 'temperature'
    ])


def save_dataset(path='data/sales.csv'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = generate_sales()
    df.to_csv(path, index=False)
    return df


if __name__ == '__main__':
    df = save_dataset()
    print(f'Generated {len(df):,} daily store-product observations')
