"""Data drift monitoring using Evidently.

Compares reference and current prediction data to detect drift in features and targets.
Generates test results and HTML reports for monitoring dashboard.
"""

import pandas as pd
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestAccuracyScore,
    TestNumberOfMissingValues,
    TestShareOfDriftedColumns,
    TestTargetFeaturesCorrelations,
)


reference_data = pd.read_csv("monitoring/reference_database.csv")
current_data = pd.read_csv("monitoring/prediction_database.csv", na_values=["Nan", "nan", "NaN", "Nan "])

if "prediction" in current_data.columns:
    current_data = current_data.rename(columns={"prediction": "target"})

feature_columns = [col for col in reference_data.columns if col.startswith("f_")]
all_columns = feature_columns + ["target"]


# Filter both dataframes to only these columns in this exact order
reference_data = reference_data[all_columns]
current_data = current_data[all_columns]


# Test amount of missing values
data_test = TestSuite(
    tests=[
        TestNumberOfMissingValues(),
        TestTargetFeaturesCorrelations(),
        TestShareOfDriftedColumns(),
        TestAccuracyScore(),
    ]
)
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result["summary"]["all_passed"])


# 5. Run the Report
report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])

# report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html("monitoring/monitoring_report.html")
