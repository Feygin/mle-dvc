# scripts/fit.py

import os
import yaml
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from category_encoders import CatBoostEncoder
from catboost import CatBoostClassifier


# обучение модели
def fit_model():
    # 1) Прочитайте файл с гиперпараметрами params.yaml
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    index_col = params["index_col"]
    target_col = params["target_col"]
    one_hot_drop = params["one_hot_drop"]
    auto_class_weights = params["auto_class_weights"]

    # 2) Загрузите результат предыдущего шага: data/initial_data.csv
    data_path = "data/initial_data.csv"
    data = pd.read_csv(data_path, index_col=index_col)

    # 3) Реализуйте основную логику шага с использованием гиперпараметров
    # Разделим на X/y
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Определим типы признаков по train-датасету
    cat_features = X.select_dtypes(include="object")
    potential_binary_features = cat_features.nunique() == 2

    binary_cat_features = cat_features[potential_binary_features[potential_binary_features].index]
    other_cat_features = cat_features[potential_binary_features[~potential_binary_features].index]
    num_features = X.select_dtypes(["float"])

    preprocessor = ColumnTransformer(
        transformers=[
            ("binary", OneHotEncoder(drop=one_hot_drop), binary_cat_features.columns.tolist()),
            ("cat", CatBoostEncoder(return_df=False), other_cat_features.columns.tolist()),
            ("num", StandardScaler(), num_features.columns.tolist()),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = CatBoostClassifier(auto_class_weights=auto_class_weights)

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

if __name__ == '__main__':
    fit_model()