from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from data_generator import save_dataset
from features import create_features, prepare_xy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'sales.csv'
MODEL = ROOT / 'models' / 'demand_model.joblib'
REPORT = ROOT / 'reports' / 'metrics.json'


def main():
    if not DATA.exists():
        save_dataset(DATA)
    raw = pd.read_csv(DATA)
    features = create_features(raw).dropna().reset_index(drop=True)
    cutoff = features.date.max() - pd.Timedelta(days=60)
    train = features[features.date < cutoff]
    test = features[features.date >= cutoff]
    x_train, y_train = prepare_xy(train)
    x_test, y_test = prepare_xy(test)
    x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

    models = {
        'random_forest': RandomForestRegressor(
            n_estimators=120, max_depth=14, min_samples_leaf=2,
            n_jobs=-1, random_state=42
        ),
        'hist_gradient_boosting': HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.07, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42
        )
    }
    results = {}
    best = None
    best_rmse = float('inf')

    for name, model in models.items():
        pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('model', model)
        ])
        pipe.fit(x_train, y_train)
        prediction = pipe.predict(x_test)
        valid = y_test.abs() > 1
        safe_mape = float(
            (abs(y_test[valid] - prediction[valid]) / y_test[valid]).mean()
        ) if valid.any() else 0.0
        metrics = {
            'mae': mean_absolute_error(y_test, prediction),
            'rmse': mean_squared_error(y_test, prediction) ** 0.5,
            'mape': safe_mape
        }
        results[name] = metrics
        if metrics['rmse'] < best_rmse:
            best_rmse = metrics['rmse']
            best = name
            best_pipe = pipe

    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({'model': best_pipe, 'features': list(x_train.columns)}, MODEL)
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({
        'best_model': best,
        'test_window_start': str(test.date.min()),
        'test_window_end': str(test.date.max()),
        'models': results
    }, indent=2))
    print(json.dumps({'best_model': best, **results[best]}, indent=2))


if __name__ == '__main__':
    main()
