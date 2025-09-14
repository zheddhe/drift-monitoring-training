import joblib
import pandas as pd
from evidently.report import Report
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.ui.workspace import Workspace
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.metric_preset import DataDriftPreset
import logging
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a Parquet file.

    :param path: The path to the Parquet file to load.
    :return: The DataFrame containing the data from the Parquet file.
    """
    return pd.read_parquet(path)


def load_model(path: str):
    """
    Load a pre-trained machine learning model from disk.

    :param path: The path to the model file to load.
    :return: The loaded machine learning model.
    """
    with open(path, 'rb') as f_in:
        return joblib.load(f_in)


def prepare_data(df: pd.DataFrame, num_features: list, cat_features: list) -> pd.DataFrame:
    """
    Prepare the input data by filtering and selecting features.

    :param df: The input DataFrame.
    :param num_features: List of numerical feature names.
    :param cat_features: List of categorical feature names.
    :return: The prepared DataFrame.
    """
    filtered_data = df.loc[
        (df.lpep_pickup_datetime >= datetime.datetime(2022, 2, 2, 0, 0)) &
        (df.lpep_pickup_datetime < datetime.datetime(2022, 2, 3, 0, 0))
    ]
    return filtered_data[num_features + cat_features]


def add_prediction_column(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'prediction' column to the DataFrame using the provided model.

    :param model: The machine learning model for making predictions.
    :param df: The DataFrame to which the prediction column will be added.
    :return: The DataFrame with the added prediction column.
    """
    df['prediction'] = model.predict(df.fillna(0))
    return df


def create_and_run_test_suite(ref_data, curr_data, column_mapping, save_file=False) -> TestSuite:
    """
    Create and run a test suite for dataset summary.

    :param ref_data: The reference data for comparison.
    :param curr_data: The current data for comparison.
    :param column_mapping: The column mapping information.
    :param save_file: Boolean flag to save the test suite report as HTML.
    :return: The TestSuite object.
    """
    test_suite = TestSuite(tests=[DataDriftTestPreset()])
    test_suite.run(reference_data=ref_data, current_data=curr_data, column_mapping=column_mapping)

    if save_file:
        test_suite.save_html("data_drift_test_suites.html")

    return test_suite


def create_and_run_report(reference_data, current_data, column_mapping, save_file=True) -> Report:
    """
    Generate a classification report using Evidently.

    :param reference_data: The reference data for comparison.
    :param current_data: The current data for comparison.
    :param column_mapping: The column mapping information.
    :param save_file: Boolean flag to save the report as HTML.
    :return: The Report object.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )

    if save_file:
        report.save_html("data_report.html")

    return report


def add_report_to_workspace(
    workspace_name: str, project_name: str,
    project_description: str, test_suite, report
):
    """
    Adds a report to a workspace.

    :param workspace_name: The name of the workspace.
    :param project_name: The name of the project.
    :param project_description: The description of the project.
    :param test_suite: The test suite report to be added to the workspace.
    :param report: The data drift report to be added to the workspace.
    """
    workspace = Workspace(workspace_name)
    project = workspace.create_project(project_name)
    project.description = project_description
    for item in [test_suite, report]:
        workspace.add_report(project.id, item)


if __name__ == "__main__":
    # Constants for workspace and project details
    WORKSPACE_NAME = "NYC-monitoring-workspace"
    PROJECT_NAME = "monitoring_test_suite_and_report"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Define file paths
    reference_data_path = 'data/reference.parquet'
    current_data_path = 'data/green_tripdata_2022-02.parquet'
    model_path = 'models/lin_reg.bin'

    # Data labeling
    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["PULocationID", "DOLocationID"]

    # Define column mapping
    column_mapping = ColumnMapping(
        prediction='prediction',
        numerical_features=num_features,
        categorical_features=cat_features,
        target=None
    )

    logging.info('Loading datasets...')
    current_data = load_data(current_data_path)
    reference_data = load_data(reference_data_path)
    logging.info('Data loaded successfully.')

    logging.info('Loading the model...')
    model = load_model(model_path)
    logging.info('Model loaded successfully.')

    logging.info('Filtering current data...')
    problematic_data = prepare_data(current_data, num_features, cat_features)
    logging.info('Data prepared successfully.')

    logging.info('Computing predictions...')
    problematic_data = add_prediction_column(model, problematic_data)
    logging.info('Predictions computed successfully.')

    logging.info('Creating and running test suite...')
    test_suite = create_and_run_test_suite(reference_data, problematic_data, column_mapping)
    logging.info('Test suite completed.')

    logging.info('Creating and running report...')
    report = create_and_run_report(reference_data, problematic_data, column_mapping)
    logging.info('Report generated successfully.')

    logging.info('Adding report to workspace...')
    add_report_to_workspace(WORKSPACE_NAME, PROJECT_NAME, PROJECT_DESCRIPTION, test_suite, report)
    logging.info('Report added to workspace.')
