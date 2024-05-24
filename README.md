# AirflowPipeline
Airflow DAG that extracts data from google sheets, transforms them,
joins them, uploads results to BigQuery, then finally trains an ARIMA model,
and stores it in a MLflow server for later use.

Adjust the model settings in dags/tasks/task2/model_config.yaml

Adjust features used in training in dags/tasks/task2/features.yaml

Adjust BigQuery info in dags/tasks/task1/pipeline_configs/bigqueryconfig.yaml
