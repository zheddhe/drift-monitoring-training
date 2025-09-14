import warnings
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.utils import Bunch
from evidently.report import Report
from sklearn.exceptions import UndefinedMetricWarning
from evidently.metric_preset import RegressionPreset
from evidently.ui.workspace import Workspace
from typing import cast

# --- Filtres de warnings (ciblés) ---
# 1) R² indéfini quand des segments ont < 2 lignes
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 2) FutureWarnings générés par Evidently sur groupby.apply
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"DataFrameGroupBy\.apply operated on the grouping columns",
)


def get_data():
    """
    Loads the diabetes dataset from sklearn and creates a DataFrame.
    """
    diabetes_data = datasets.load_diabetes()
    diabetes_data = cast(Bunch, diabetes_data)
    diabetes_df = pd.DataFrame(
        diabetes_data.data,
        columns=diabetes_data.feature_names
    )
    diabetes_df['target'] = diabetes_data.target
    noise = pd.Series(
        np.random.normal(0.0, 3.0, len(diabetes_df)),
        index=diabetes_df.index,
        dtype="float64",
    )
    diabetes_df["target"] = diabetes_df["target"].astype("float64")
    diabetes_df['prediction'] = diabetes_df["target"] + noise
    return diabetes_df


def prepare_data(dataframe, ref_sample_size, curr_sample_size):
    """
    Prepares reference and current datasets for comparison.
    """
    reference_data = dataframe.sample(n=ref_sample_size, replace=False, random_state=42)
    current_data = dataframe.sample(n=curr_sample_size, replace=False, random_state=43)
    return reference_data, current_data


def generate_regression_performance_report(reference_data, current_data):
    """
    Generates a regression performance report using Evidently.
    """
    # Create a report instance for regression with RegressionPreset()
    regression_performance_report = Report(metrics=[RegressionPreset()])

    # Run the report
    regression_performance_report.run(
        reference_data=reference_data.sort_index(),
        current_data=current_data.sort_index()
    )
    return regression_performance_report


def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    This function will be useful to you
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
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "regression_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Get the diabetes dataset
    diabetes_df = get_data()

    # Prepare reference and current datasets by taking a sample of 50 rows
    reference_data, current_data = prepare_data(
        diabetes_df,
        ref_sample_size=50,
        curr_sample_size=50
    )

    # Generate the regression performance report
    regression_report = generate_regression_performance_report(reference_data, current_data)

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, regression_report)
