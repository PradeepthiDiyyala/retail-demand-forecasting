import numpy as np


def inventory_policy(forecast_units, lead_time_days=3, service_level=0.95, review_period=7):
    z = 1.645 if service_level >= 0.95 else 1.28
    values = np.asarray(forecast_units, dtype=float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    safety = z * std * np.sqrt(lead_time_days)
    reorder = mean * lead_time_days + safety
    target = mean * (lead_time_days + review_period) + safety
    return {
        'avg_daily_demand': round(mean, 2),
        'demand_std': round(std, 2),
        'safety_stock': round(safety, 2),
        'reorder_point': round(reorder, 2),
        'target_stock': round(target, 2)
    }
