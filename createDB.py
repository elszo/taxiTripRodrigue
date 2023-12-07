import sqlite3
import os
import common
from sklearn.model_selection import train_test_split

def createDBTaxi(path, data_train, data_test):

    db_dir = os.path.dirname(path)
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)

    print(f"Saving train and test data to a database: {path}")
    with sqlite3.connect(common.DB_PATH) as con:
        data_train.to_sql(name='train', con=con, if_exists="replace")
        data_test.to_sql(name='test', con=con, if_exists="replace")



if __name__ == "__main__":
    path = './data/New_York_City_Taxi_Trip_Duration.zip'
    data = common.load_data_csv(path)
    data = common.preprocess_data(data)
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=common.RANDOM_STATE)
    createDBTaxi(common.DB_PATH, data_train, data_test)