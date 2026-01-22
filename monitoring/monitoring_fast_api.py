import io
import json
import pandas as pd
from google.cloud import storage
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from datetime import datetime
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import (TestNumberOfMissingValues, TestTargetFeaturesCorrelations, 
                             TestShareOfDriftedColumns)


BUCKET_NAME = "mlopsproject-data"
app = FastAPI()

def fetch_data_from_gcp(n=50):
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    #Pull Reference Data from Bucket
    blob_ref = bucket.blob("monitoring/reference_database.csv")
    reference_data = pd.read_csv(io.BytesIO(blob_ref.download_as_bytes()))

    #Pull JSONs from Bucket
    blobs = list(bucket.list_blobs(prefix="predictions/"))
    blobs.sort(key=lambda x: x.updated, reverse=True)

    current_rows = []
    for blob in blobs[:n]:
        if not blob.name.endswith('.json'):
            continue
    
        data = json.loads(blob.download_as_text())
        row = {f"f_{i}": val for i, val in enumerate(data["features"])}
        row["target"] = data["prediction"]
        current_rows.append(row)
    
    if not current_rows:
        return reference_data, pd.DataFrame(columns=reference_data.columns)

    current_data = pd.DataFrame(current_rows)
    return reference_data, current_data

@app.get("/report", response_class=HTMLResponse)
async def get_report():
    reference_data, current_data = fetch_data_from_gcp(n=50)
    #If no predictions have been made yet
    if current_data.empty:
        return HTMLResponse(content="<h1>No predictions found in bucket yet.</h1>", status_code=200)

    feature_columns = [col for col in reference_data.columns if col.startswith('f_')]
    all_columns = feature_columns + ['target']

    reference_data = reference_data[all_columns]
    current_data = current_data[all_columns]
    
    # The Datadrifting Tests
    data_test = TestSuite(tests=[
        TestNumberOfMissingValues(), 
        TestTargetFeaturesCorrelations(), 
        TestShareOfDriftedColumns()
    ])
    data_test.run(reference_data=reference_data, current_data=current_data)
    
    #Create data drifting Html Report
    report = Report(metrics=[
        DataDriftPreset(), 
        DataQualityPreset(), 
        TargetDriftPreset()
    ]) 
    report.run(reference_data=reference_data, current_data=current_data)


    report_filename = "monitoring_report.html"
    report.save_html(report_filename)
    
    #Upload report to the Bucket
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        blob_name = f"reports/drift_report_{timestamp}.html"
        blob = bucket.blob(blob_name)
        
        
        blob.upload_from_filename(report_filename, content_type="text/html")
        print(f"DEBUG: Report saved permanently to {blob_name}")
    except Exception as e:
        print(f"CLOUD ERROR (Saving Report): {e}")

    
    with open(report_filename, "r", encoding="utf-8") as f:
        html_content = f.read()

    return HTMLResponse(content=html_content)



