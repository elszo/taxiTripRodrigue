import sqlite3
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import common

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    X_train = data_train.drop(columns=['trip_duration', 'log_trip_duration'])
    y_train = data_train['log_trip_duration']
    return X_train, y_train

def fit_model(X_train, y_train):
    print(f"Fitting a model")

    num_features = ['pickup_longitude',	'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'hour']
    cat_features = ['weekday', 'month']
    train_features = num_features + cat_features

    column_transformer = ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown="ignore"), cat_features),
        ('scaling', StandardScaler(), num_features)]
    )

    pipeline = Pipeline(steps=[
        ('ohe_and_scaling', column_transformer),
        ('regression', RandomForestRegressor(n_estimators=100, max_depth=10))
    ])

    model = pipeline.fit(X_train[train_features], y_train)
    y_pred_train = model.predict(X_train[train_features])
    score = mean_squared_error(y_train, y_pred_train, squared=False)
    print(f"Score on train data {score:.2f}")
    return model

if __name__ == "__main__":

    X_train, y_train = load_train_data(common.DB_PATH)
    model = fit_model(X_train, y_train)
    common.persist_model(model, common.MODEL_PATH)