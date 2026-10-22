import numpy as np
import pandas as pd

TARGET = 'units_sold'


def create_features(df, lags=(1, 7, 14, 28), windows=(7, 14, 28)):
    out = df.copy()
    out['date'] = pd.to_datetime(out['date'])
    out = out.sort_values(['store_id', 'product_id', 'date'])
    out['day_of_week'] = out.date.dt.dayofweek
    out['day_of_month'] = out.date.dt.day
    out['month'] = out.date.dt.month
    out['week_of_year'] = out.date.dt.isocalendar().week.astype(int)
    out['quarter'] = out.date.dt.quarter
    out['sin_dow'] = np.sin(2 * np.pi * out.day_of_week / 7)
    out['cos_dow'] = np.cos(2 * np.pi * out.day_of_week / 7)
    out['sin_year'] = np.sin(2 * np.pi * out.date.dt.dayofyear / 365)
    out['cos_year'] = np.cos(2 * np.pi * out.date.dt.dayofyear / 365)
    group = out.groupby(['store_id', 'product_id'], group_keys=False)
    for lag in lags:
        out[f'lag_{lag}'] = group[TARGET].shift(lag)
    for window in windows:
        out[f'rolling_mean_{window}'] = group[TARGET].transform(
            lambda series: series.shift(1).rolling(window, min_periods=3).mean()
        )
        out[f'rolling_std_{window}'] = group[TARGET].transform(
            lambda series: series.shift(1).rolling(window, min_periods=3).std()
        )
    out['price_change'] = (
        group['price'].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    )
    return out


def prepare_xy(df):
    x = df.drop(columns=['date', TARGET])
    x = pd.get_dummies(x, columns=['store_id', 'product_id'], dtype=float)
    return x, df[TARGET]
