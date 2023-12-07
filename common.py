import pickle
import os
import pandas as pd

# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(ROOT_DIR, 'config.ini')

# Using INI configuration file
from configparser import ConfigParser

config = ConfigParser()
config.read(CONFIG_PATH)
DB_PATH = str(config.get("PATHS", "DB_PATH"))
MODEL_PATH = str(config.get("PATHS", "MODEL_PATH"))
RANDOM_STATE = int(config.get("ML", "RANDOM_STATE"))

# # Doing the same with a YAML configuration file
# import yaml
#
# with open("config.yml", "r") as f:
#     config_yaml = yaml.load(f, Loader=yaml.SafeLoader)
#     DB_PATH = str(config_yaml['paths']['db_path'])
#     MODEL_PATH = str(config_yaml['paths']["model_path"])
#     RANDOM_STATE = int(config_yaml["ml"]["random_state"])

# SQLite requires the absolute path
# DB_PATH = os.path.abspath(DB_PATH)
DB_PATH = os.path.join(ROOT_DIR, os.path.normpath(DB_PATH))

def load_data_csv(path):

    data = pd.read_csv(path, compression='zip')
    #os.remove(path)

    return data

def preprocess_data(data):

    data = data.drop(columns=['id', 'dropoff_datetime', 'vendor_id', 'store_and_fwd_flag', 'passenger_count'])
    data['pickup_datetime'] = pd.to_datetime(data['pickup_datetime'])

    data['weekday'] = data['pickup_datetime'].dt.weekday
    data['month'] = data['pickup_datetime'].dt.month
    data['hour'] = data['pickup_datetime'].dt.hour

    data['pickup_date'] = data['pickup_datetime'].dt.date
    df_abnormal_dates = data.groupby('pickup_date').size()
    abnormal_dates = df_abnormal_dates[df_abnormal_dates < df_abnormal_dates.quantile(0.02)]
    data['abnormal_period'] = data['pickup_datetime'].dt.date.isin(abnormal_dates.index).astype(int)
    return data

def persist_model(model, path):
    print(f"Persisting the model to {path}")
    model_dir = os.path.dirname(path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(path, "wb") as file:
        pickle.dump(model, file)
    print(f"Done")

def load_model(path):
    print(f"Loading the model from {path}")
    with open(path, "rb") as file:
        model = pickle.load(file)
    print(f"Done")
    return model