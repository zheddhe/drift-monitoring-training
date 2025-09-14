import datetime
import time
# import random
import logging
# import uuid
# import pytz
import pandas as pd
# import io
import psycopg
import joblib
from prefect import task, flow
from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from typing import LiteralString

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
RANDOM_STATE = 42

create_table_statement: LiteralString = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
)
"""


def load_data(path: str):
    return pd.read_parquet(path)


def load_model(path: str):
    with open(path, 'rb') as f_in:
        return joblib.load(f_in)


def prepare_db():
    conn = psycopg.connect("host=localhost port=5432 user=postgres password=example")
    cur = conn.cursor()
    res = cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(res.fetchall()) == 0:
        cur.execute("create database test;")
        conn.commit()
    conn.close()
    conn = psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=example")
    cur = conn.cursor()
    cur.execute(create_table_statement)
    conn.commit()
    conn.close()


@task
def calculate_metrics(i):
    # Get the raw data for the current time interval : slicing over one day
    raw_data_slice = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin_date + datetime.timedelta(i))) &
        (raw_data.lpep_pickup_datetime < (begin_date + datetime.timedelta(i + 1)))
    ]

    # Fill missing values in the numeric and categorical features with 0
    # raw_data_slice.fillna(0, inplace=True)

    # Add a prediction column to the current data using the model
    raw_data_slice['prediction'] = model.predict(
        raw_data_slice[num_features + cat_features].fillna(0)
    )

    # Calculate the monitoring metrics for the current data
    report.run(
        reference_data=reference_data,
        current_data=raw_data_slice,
        column_mapping=column_mapping
    )
    result = report.as_dict()

    # Extract the monitoring metrics from the result
    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']

    # Return the current time interval and the monitoring metrics
    return i, prediction_drift, num_drifted_columns, share_missing_values


@flow
def batch_monitoring_backfill():

    # Prepare the database for storing metrics
    prepare_db()

    # Create a connection to the database with autocommit enabled
    conn = psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True
    )

    # Calculate metrics for each time interval
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    for i in range(0, 27):  # working with february data
        # Calculate metrics for the current time interval
        result = calculate_metrics(i)

        # Insert the calculated metrics into the database
        with conn.cursor() as curr:
            curr.execute(
                "insert into dummy_metrics(timestamp, prediction_drift, num_drifted_columns,"
                " share_missing_values) values (%s, %s, %s, %s)",
                (begin_date + datetime.timedelta(i), result[1], result[2], result[3])
            )

        # Wait for the specified timeout period before sending the next batch of data
        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)

        # Update the timestamp of the last sent data
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=10)

        # Log a message to indicate that the data has been sent
        logging.info("data sent")

    # Close the connection to the database
    conn.close()


if __name__ == '__main__':
    # from the baseline created
    reference_data_path = 'data/reference.parquet'
    model_path = 'models/lin_reg.bin'
    raw_data_path = 'data/green_tripdata_2022-02.parquet'

    begin_date = datetime.datetime(2022, 2, 1, 0, 0)
    num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
    cat_features = ['PULocationID', 'DOLocationID']

    column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features,
        target=None
    )

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    # getting previously calculated referenced data
    reference_data = load_data(reference_data_path)

    # load model
    model = load_model(model_path)

    # load static data for batch processing
    raw_data = load_data(raw_data_path)

    # run prefect pipelline
    batch_monitoring_backfill()
