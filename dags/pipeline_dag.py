""" Airflow DAG that extracts data from google sheets, transforms them,
joins them, uploads results to BigQuery, then finally trains a ARIMA model,
and stores it in a MLflow server for later use.

Adjust the model settings in dags/tasks/task2/model_config.yaml
Adjust features used in training in dags/tasks/task2/features.yaml
Adjust BigQuery info in dags/tasks/task1/pipeline_configs/bigqueryconfig.yaml
"""

from datetime import timedelta
from airflow import DAG
import sys
import os
sys.path.insert(0, os.getcwd())
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from tasks.task1.transformers.labour_transformer_google import LabourTransformer
from tasks.task1.transformers.invoice_transformer_google import InvoiceTransformer
from tasks.task1.transformers.sales_transformer_google import SalesDataTransformer
from tasks.task1.database import schema, load_latest_data_from_bigquery, project, table_id
from tasks.task1.full_pipeline import Pipeline
from tasks.task1.utils import check_for_sheet_updates, load_config_yaml_file
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
import mlflow
import yaml

# Make pipeline run regardless of updates.
force_run = False

# mlflow settings
mlflow_tacking_uri = ""
experiment = ""

# Load model config
model_conf = load_config_yaml_file("./dags/tasks/task2/model_config.yaml")

args = {
    'owner': "",
    'depends_on_past': False,
    "start_date": days_ago(31),
    "email": "",
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=2)
}


def updates(sheet_config_path:str, table_id:str, update_dump_path:str):
    """ Scans google sheets for any changes which will trigger the pipeline.

    Parameters
    ----------
    sheet_config_path : str 
        path that contains the google sheet ids, sheet names, etc.
    project : str
        google cloud project id.
    table_id: str
        table in big query.
    update_dump_path : str 
        path to store temporary file which holds info for 
        updated sheets.

    Returns
    -------
    bool
    """
    update_dict = check_for_sheet_updates(sheet_config_path, table_id)
    print(update_dict)
    # If there is nothing in these, there are no updates
    if update_dict["pnl_sheets"].get('year_start') or update_dict["sales_sheet"]:
        with open(update_dump_path, 'w') as f:
            yaml.dump(update_dict, f)
        return True
    else:
        return False


def get_latest_data(client, table_id):
    """ Gets a sample of the data contained in bigquery.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
    table_id: str
      table_id where data will be retrieved from

    Returns
    -------
    pd.DataFrame
    """
    latest_data = load_latest_data_from_bigquery(client, table_id, limit=1)
    latest_data = latest_data.reindex(sorted(latest_data.columns), axis=1)

    if latest_data.empty:
        latest_data = None

    return latest_data


def run_extract(transformer, sheet_config_path, extracted_name):
    """Runs the supplied data transformer calling its extract() method.

    Parameters
    ----------
    transformer : Transformer
      data transformer
    sheet_config_path : str
      path to yaml containing google sheets
    extract_task_name : str
      name of the task that extracted the data

    Returns
    -------
    dict or pd.DataFrame
    """
    transformer_inst = transformer(config_path=sheet_config_path)
    transformer_inst.extract()
    return getattr(transformer_inst, extracted_name)
    

def run_transformer(transformer, sheet_config_path:str, extract_task_name:str, 
                    extracted_value_name:str, ti):
    """Runs the supplied data transformer calling its transform() method.

    Parameters
    ----------
    transformer : Transformer
      data transformer
    sheet_config_path : str
      path to yaml containing google sheets
    extract_task_name : str
      name of the task that extracted the data
    extracted_value : str
      name of the extracted data attribute for transformer

    Returns
    -------
    pd.DataFrame
    """
    extracted = ti.xcom_pull(task_ids=extract_task_name)
    transformer_inst = transformer(config_path=sheet_config_path)
    setattr(transformer_inst, extracted_value_name, extracted)
    return transformer_inst.transform()


def run_pipeline(ti):
    """Runs the Pipeline class from the full_pipeline module.

    Returns
    -------
    pd.DataFrame
    """
    # Pull transformed data
    invoices = ti.xcom_pull(task_ids="run_invoice_transformer")
    labour = ti.xcom_pull(task_ids="run_labour_transformer")
    sales = ti.xcom_pull(task_ids="run_sales_transformer")

    # Transform data.
    transformer = Pipeline(invoices, sales, labour)
    return transformer.transform()


def check_schema(ti):
    """Makes sure the merged data has the same columns as the data contained 
    in bigquery. Will add empty columns of the dataframe is missing some.

    Returns
    -------
    pd.DataFrame
    """
    df = ti.xcom_pull(task_ids="merge_data")
    latest_df = ti.xcom_pull(task_ids="get_latest_data")

    if latest_df is not None:
        for col in latest_df.columns:
                if col not in df.columns:
                    df[col] = 0.00  
        # Sort updated_data's columns
        df = df.reindex(sorted(df.columns), axis=1)

    return df


def cleanup(sheet_update_path:str):
    """delete temporary updates.yaml file in sheet_update_path.

    Parameters
    ----------
    sheet_update_path: str
    """
    try:
        os.remove(sheet_update_path)
    except FileNotFoundError:
        # This should happen when there are no updates
        pass

    for file in os.listdir("./artifacts/outputs"):
        try:
            if file != "labour_transformed.csv":
                os.remove(os.path.join("./artifacts/outputs", file))
        except Exception as e:
            print("error in deleting outputs:", e)


def load_data_to_bigquery(client, table_id, schema, ti):
    """ Load the result from check_schema to BigQuery
    """
    from tasks.task1.database import load
    df = ti.xcom_pull(task_ids="check_schema")
    df['ds'] = df['ds'].astype(str)
    if df is not None:
        # Load into biqquery
        load(client, df, table_id, schema)


def extract_data(client, table_id):
    return load_latest_data_from_bigquery(client, table_id)


def train_model(ti):
    """Trains ARIMA model and logs it to mlflow tracking server"""
    from tasks.task2.training import train, predictor_columns as features
    from tasks.task1.utils import reformat_model_params

    model_name = "ARIMA3_1_0_polyfeatures"

    df = ti.xcom_pull(task_ids="extract_data_from_bigquery")

    mlflow.set_tracking_uri(mlflow_tacking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run():
        # Remove unknown future y
        train_df = df[df['y'] != 0.00]

        X = train_df[features] 
        y = train_df['y']

        # Train and save model
        model = train(train_df, model_config=model_conf)

        params = reformat_model_params(model.get_model_params().get("arima_params"))
        mlflow.log_params(params)

        # Log the model instance input parameters
        mlflow.log_params(model.model['ARIMA'].get_params())
        
        mlflow.log_param("features", features)

        # Get insample predictions and log performance
        model.get_insample_predictions()
        mlflow.log_metrics(model.error_metrics(y))

        dataset = mlflow.data.from_pandas(df)
        mlflow.log_input(dataset, context="Training")
        
        mlflow.sklearn.log_model(
            model.model, 
            artifact_path="model", 
            registered_model_name=model_name,
            code_paths=["./dags/tasks"]
        )



with DAG(
    dag_id="ETL",
    description="Dag that pulls data from google sheets, runs transformations, and uploads to bigquery",
    default_args=args,
    catchup=False,
) as pipeline_dag:
    # Get BigQuery Client
    bq_hook = BigQueryHook(gcp_conn_id="bq", use_legacy_sql=False)
    client = bq_hook.get_client()

    if not force_run:
        sheet_update_path = "./dags/tasks/task1/pipeline_configs/updates.yaml"
        # Cancel the pipeline run if no updates.
        updates = ShortCircuitOperator(
            task_id="check_for_updates",
            python_callable=updates,
            op_kwargs= {
                "sheet_config_path": "./dags/tasks/task1/pipeline_configs/googlesheets.yaml",
                "table_id": table_id,
                "update_dump_path": "./dags/tasks/task1/pipeline_configs/updates.yaml"
            }
        )
    else:
        sheet_update_path = "./dags/tasks/task1/pipeline_configs/googlesheets.yaml"

        updates = PythonOperator(
            task_id="check_for_updates",
            python_callable=updates,
            op_kwargs= {
                "sheet_config_path": "./dags/tasks/task1/pipeline_configs/googlesheets.yaml",
                "table_id": table_id,
                "update_dump_path": "./dags/tasks/task1/pipeline_configs/updates.yaml"
            }
        )

    # Get the most recent data that was appended to BigQuery
    sample = PythonOperator(
        task_id = "get_sample_data",
        python_callable=get_latest_data,
        op_args=[client, table_id]
    )

    # Extract and run transformers
    extract_lab = PythonOperator(
        task_id="extract_labour",
        python_callable=run_extract,
        op_args=[LabourTransformer, sheet_update_path, "labour_dict"]
    )

    labour = PythonOperator(
        task_id="run_labour_transformer",
        python_callable = run_transformer,
        op_kwargs={
            "transformer": LabourTransformer,
            "sheet_config_path": sheet_update_path,
            "extract_task_name": "extract_labour",
            "extracted_value_name": "labour_dict"
        }
    )

    extract_inv = PythonOperator(
        task_id = "extract_invoices",
        python_callable=run_extract,
        op_args=[InvoiceTransformer, sheet_update_path, "invoice_dict"]

    )

    iv_transformer = PythonOperator(
        task_id="run_invoice_transformer",
        python_callable = run_transformer,
        op_kwargs={
            "transformer": InvoiceTransformer,
            "sheet_config_path": sheet_update_path,
            "extract_task_name": "extract_invoices",
            "extracted_value_name": "invoice_dict"
        }
    )
  

    extract_sales = PythonOperator(
        task_id="extract_sales",
        python_callable=run_extract,
        op_args=[SalesDataTransformer, sheet_update_path, "sales"]
    )

    sales_transformer = PythonOperator(
        task_id="run_sales_transformer",
        python_callable = run_transformer,
        op_kwargs={
            "transformer": SalesDataTransformer,
            "sheet_config_path": sheet_update_path,
            "extract_task_name": "extract_sales",
            "extracted_value_name": "sales"
        }
    )

    merge_data = PythonOperator(
        task_id="merge_data",
        python_callable=run_pipeline
    )

    check_schema = PythonOperator(
        task_id="check_schema",
        python_callable=check_schema
    )

    load = PythonOperator(
        task_id="load_to_bigquery",
        python_callable=load_data_to_bigquery,
        op_args=[client, table_id, schema]
    )

    if not force_run:
        clean = PythonOperator(
            task_id="cleanup",
            python_callable=cleanup,
            op_args=[sheet_update_path]
        )
    else:
        clean = EmptyOperator(
            task_id="cleanup"
        )

    extract = PythonOperator(
        task_id="extract_data_from_bigquery",
        python_callable=extract_data,
        op_args=[client, table_id]
    )

    train = PythonOperator(
        task_id="train_ARIMA",
        python_callable=train_model
    )

# Dependencies 
updates >> sample >> [extract_inv, extract_lab, extract_sales]
[extract_lab>>labour, extract_inv>>iv_transformer, extract_sales>>sales_transformer] >> merge_data >> check_schema
check_schema >> load >> clean >> extract >> train

