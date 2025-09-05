# scripts/evaluate.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
import joblib
import json
import yaml
import os

# оценка качества модели
def evaluate_model():
    # 1) читаем файл с гиперпараметрами
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    id_col = params["id_col"]
    index_col = params["index_col"]
    target_col = params["target_col"]
    n_splits = params["n_splits"]
    metrics = params["metrics"]
    n_jobs = params["n_jobs"]

    # 2) загружаем данные и модель
    data = pd.read_csv("data/initial_data.csv")
    data = data.drop(columns=[id_col, index_col])
    pipeline = joblib.load("models/fitted_model.pkl")

    # 3) кросс-валидация
    X = data.drop(columns=[target_col])
    y = data[target_col]

    cv_strategy = StratifiedKFold(n_splits=n_splits)

    cv_res = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cv_strategy,
        scoring=metrics,
        n_jobs=n_jobs,
        return_train_score=False
    )

    # 4) усреднение результатов по логике из условия
    cv_summary = {}
    for key, value in cv_res.items():
        cv_summary[f"{key}"] = round(value.mean(), 2)

    # 5) сохраняем результаты
    os.makedirs("cv_results", exist_ok=True)
    with open("cv_results/cv_res.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    evaluate_model()