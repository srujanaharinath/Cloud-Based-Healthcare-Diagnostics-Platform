from airflow import DAG
from datetime import datetime, timedelta
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.operators.dataproc import  DataprocCreateClusterOperator
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from airflow.providers.google.cloud.operators.dataproc import DataprocDeleteClusterOperator
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator

default_args = {
    'depends_on_past': False   
}

CLUSTER_NAME = 'inference'
REGION='us-central1'
PROJECT_ID='sharp-imprint-420622'
PYSPARK_URI='gs://big_data_diseases/notebooks/inference.py'


CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n2-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 30},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n2-standard-2",
        "disk_config": {"boot_disk_type": "pd-standard", "boot_disk_size_gb": 30},
    },
    "software_config": {
        "image_version": "2.2-ubuntu22"
    }
}


PYSPARK_JOB = {
    "reference": {"project_id": PROJECT_ID},
    "placement": {"cluster_name": CLUSTER_NAME},
    "pyspark_job": {"main_python_file_uri": PYSPARK_URI},
}

with DAG(
    'model_inference',
    default_args=default_args,
    description='model_inference',
    schedule_interval=None,
    start_date = days_ago(2)
) as dag:

    transfer_file_task = GCSToGCSOperator(
    task_id='transfer_file',
    source_bucket='symptoms_incoming',
    source_object='symptoms.csv',
    destination_bucket='big_data_diseases',
    destination_object='incoming/symptoms.csv',
    move_object=True,
    )

    create_cluster = DataprocCreateClusterOperator(
        task_id="create_cluster",
        project_id=PROJECT_ID,
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
        cluster_name=CLUSTER_NAME,
    )

    submit_job = DataprocSubmitJobOperator(
        task_id="pyspark_task", 
        job=PYSPARK_JOB, 
        region=REGION, 
        project_id=PROJECT_ID
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_cluster", 
        project_id=PROJECT_ID, 
        cluster_name=CLUSTER_NAME, 
        region=REGION
    )
    
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    destination_object = f'archive/symptoms_{current_timestamp}.csv'

    transfer_file_task2 = GCSToGCSOperator(
    task_id='transfer_file2',
    source_bucket='big_data_diseases',
    source_object='incoming/symptoms.csv',
    destination_bucket='big_data_diseases',
    destination_object=destination_object,
    move_object=True,
    )

    transfer_file_task >> create_cluster >> submit_job >> delete_cluster >> transfer_file_task2 