# scripts/fit.py

import os
import yaml
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression


# обучение модели
def fit_model():
    # 1) Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    id_col = params["id_col"]
    index_col = params["index_col"]
    target_col = params["target_col"]
    one_hot_drop = params["one_hot_drop"]

    # 2) Загрузите результат предыдущего шага: data/initial_data.csv
    data_path = "data/initial_data.csv"
    data = pd.read_csv(data_path)
    data = data.drop(columns=[id_col, index_col])

    # 3) Реализуйте основную логику шага с использованием гиперпараметров
    # Разделим на X/y
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Определим типы признаков по train-датасету
    cat_features = X.select_dtypes(include="object").columns.tolist()
    num_features = X.select_dtypes(include=["float"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop=one_hot_drop, handle_unknown="ignore"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Модель: логистическая регрессия
    model = LogisticRegression(C=1, penalty="l2", max_iter=1000)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Обучаем
    pipeline.fit(X, y)

    # 4) Сохраните обученную модель
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/fitted_model.pkl")


if __name__ == "__main__":
    fit_model()
