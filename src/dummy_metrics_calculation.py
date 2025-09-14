# Import necessary libraries
import datetime
import time
import random
import logging
import uuid
import pytz
# import pandas as pd
# import io
import psycopg
from typing import LiteralString

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

# Timeout for sending data
SEND_TIMEOUT = 10

# Create table statement for PostgreSQL
create_table_statement: LiteralString = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    timestamp timestamp,
    value1 integer,
    value2 varchar,
    value3 float
)
"""


def prep_db():
    """
    Prepare the PostgreSQL database by creating the necessary database and table.
    """
    # Connect to PostgreSQL server
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example",
        autocommit=True
    ) as conn:
        # Check if the 'test' database exists, create it if not
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:
            conn.execute("create database test;")
        # Connect to the 'test' database
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            # Create the 'dummy_metrics' table
            conn.execute(create_table_statement)


def calculate_dummy_metrics_postgresql(curr):
    """
    Generate random dummy metrics and insert them into the 'dummy_metrics' table in PostgreSQL.
    """
    # Generate random values for the metrics
    value1 = random.randint(0, 1000)
    value2 = str(uuid.uuid4())
    value3 = random.random()

    # Execute SQL query to insert the metrics into the table
    curr.execute(
        "insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
        (datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
    )


def main():
    """
    Main function to generate and send dummy metrics to the PostgreSQL database.
    """
    # Prepare the PostgreSQL database
    prep_db()

    # Initialize last send time
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)

    # Connect to the 'test' database
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True
    ) as conn:
        # Loop to generate and send dummy metrics
        for i in range(0, 100):
            # Create a cursor for executing SQL queries
            with conn.cursor() as curr:
                # Calculate and insert dummy metrics into the database
                calculate_dummy_metrics_postgresql(curr)

            # Calculate time elapsed since last send
            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()

            # Wait if SEND_TIMEOUT hasn't elapsed since last send
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)

            # Update last send time to current time
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)

            # Log message indicating data has been sent
            logging.info("data sent")


if __name__ == '__main__':
    # Call the main function if the script is executed directly
    main()
