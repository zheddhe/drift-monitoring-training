import json
# import pandas as pd
import numpy as np
from sklearn import datasets
from evidently.report import Report
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric

# Fetch the dataset
adult_data = datasets.fetch_openml(
    name='adult',
    version=2,
    as_frame='auto'
)
# Transform into dataframe
adult = adult_data.frame

# Split the dataset for drift detection into reference and current data
adult_ref = adult[~adult.education.isin(
    [
        'Some-college',
        'HS-grad',
        'Bachelors'
    ]
)]
adult_cur = adult[adult.education.isin(
    [
        'Some-college',
        'HS-grad',
        'Bachelors']
)]

# Introduce missing values for demonstration
adult_cur.iloc[:2000, 3:5] = np.nan

# Initialize the report with desired metrics
data_drift_dataset_report = Report(metrics=[
    DatasetDriftMetric(),
    DataDriftTable(),
])

# Run the report
data_drift_dataset_report.run(
    reference_data=adult_ref,
    current_data=adult_cur,
)

# Convert the JSON string to a Python dictionary for pretty printing
report_data = json.loads(data_drift_dataset_report.json())

# Save the report in JSON format with indentation for better readability
with open('data_drift_report.json', 'w') as f:
    json.dump(report_data, f, indent=4)

# save HTML
data_drift_dataset_report.save_html("Data drift report.html")

# in a notebook run :
# data_drift_dataset_report.show()
