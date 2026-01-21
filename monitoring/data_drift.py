import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset

reference_data = pd.read_csv('monitoring/reference_database.csv')
current_data = pd.read_csv('monitoring/prediction_database.csv')

if 'prediction' in current_data.columns:
    current_data = current_data.rename(columns={'prediction': 'target'})
target_columns = ['brightness', 'contrast', 'sharpness', 'target']


# Filter both dataframes to only these columns in this exact order
reference_data = reference_data[target_columns]
current_data = current_data[target_columns]

print("Columns in Reference:", reference_data.columns.tolist())
print("Columns in Current:", current_data.columns.tolist())

# 5. Run the Report
report = Report(metrics=[
    DataDriftPreset(), 
    DataQualityPreset(), 
    TargetDriftPreset()
]) 

#report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('monitoring/monitoring_report.html')

