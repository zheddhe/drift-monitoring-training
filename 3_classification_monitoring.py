# import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.ensemble import RandomForestClassifier
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.ui.workspace import Workspace
from typing import cast


def get_data():
    """
    Loads the iris dataset from sklearn and creates a DataFrame.
    """
    # Load the iris dataset from sklearn
    iris_data = datasets.load_iris()
    iris_data = cast(Bunch, iris_data)

    # Create a DataFrame from the dataset's features and target values
    iris_df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

    # Create a binary classification problem
    positive_class = 1
    iris_df['target'] = iris_data.target
    iris_df['target'] = (iris_df['target'] == positive_class).astype(int)

    return iris_df


def prepare_data(dataframe, ref_sample_size, curr_sample_size):
    """
    Prepares reference and current datasets for comparison.
    """
    # Create reference and current datasets for comparison
    reference_data = dataframe.sample(n=ref_sample_size, replace=False)
    current_data = dataframe.sample(n=curr_sample_size, replace=False)

    return reference_data, current_data


def get_prediction(reference_data, current_data):
    """
    Generates predictions for reference and current data using RandomForestClassifier.
    """
    # Create a copy of the dataframes to avoid modifying the original data
    reference_data = reference_data.copy()
    current_data = current_data.copy()

    # Create a simple RandomForestClassifier
    model = RandomForestClassifier(random_state=42)  # Added random_state for reproducibility
    model.fit(reference_data.drop(columns=['target']), reference_data['target'])

    # Generate predictions for reference and current data
    reference_data['prediction'] = model.predict_proba(
        reference_data.drop(columns=['target'])
    )[:, 1]
    current_data['prediction'] = model.predict_proba(
        current_data.drop(columns=['target'])
    )[:, 1]

    return reference_data, current_data


def generate_classification_report(reference_data, current_data):
    """
    Generates a classification report using Evidently.
    """
    classification_report = Report(metrics=[
        ClassificationPreset(probas_threshold=0.5),
    ])

    # Generate the report
    classification_report.run(reference_data=reference_data, current_data=current_data)

    return classification_report


def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")


if __name__ == "__main__":
    # Defining workspace and project details
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "classification_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Get the iris dataset
    iris_df = get_data()

    # Prepare reference and current datasets
    # TODO : complete with ref_sample_size=50 and curr_sample_size=50
    reference_data, current_data = prepare_data(iris_df, ref_sample_size=50, curr_sample_size=50)

    # Get predictions for reference and current data
    # TODO : complete
    reference_data_with_pred, current_data_with_pred = get_prediction(reference_data, current_data)

    # Generate the classification report
    # TODO : complete
    classification_report = generate_classification_report(
        reference_data_with_pred,
        current_data_with_pred
    )

    # Set workspace
    workspace = Workspace(WORKSPACE_NAME)

    # Add report to workspace
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, classification_report)
