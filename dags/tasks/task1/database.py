""" Collection of functions for interacting with BigQuery.
"""

import pandas as pd
import numpy as np
import os
from google.cloud import bigquery
from datetime import datetime
import yaml


with open("./dags/tasks/task1/pipeline_configs.yaml", "r") as f:
    bq_config = yaml.safe_load(f)
    project = bq_config["project"]
    dataset_id = bq_config["dataset"]
    table_id = bq_config["output_table_id"]


schema = [
        bigquery.SchemaField(name='ds', field_type="STRING", mode="REQUIRED"),
        bigquery.SchemaField(name="y", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="Sales", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="Sales_lagged1", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="Sales_lagged2", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="Sales_lagged3", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature1", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature2", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature3", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature4", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature5", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature6", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature7", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature8", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature9", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature10", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature11", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature12", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature13", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature14", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature15", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="feature16", field_type="FLOAT", mode="REQUIRED"),
        bigquery.SchemaField(name="Ingest_ts", field_type="DATETIME", mode="REQUIRED")
    ]



def create_dataset(client, name):
    dataset = bigquery.Dataset(f"{client.project}.{name}")
    dataset.location ="US"
    dataset = client.create_dataset(dataset, timeout=30)


def create_table(client, project, schema, dataset, table_name="data"):
    dataset = client.get_dataset(f"{project}.{dataset}")
    table_id = f"{client.project}.{dataset.dataset_id}.{table_name}"
    table = bigquery.Table(table_id, schema=schema)
    client.create_table(table, timeout=30)


def load(client, df, table_id, schema, sort_by='ds'):
    """Loads the data to the bigquery table and adds a ingestion 
    timestamp.

    Parameters
    ----------
    client : google.cloud.bigquery.Client
    df : pd.DataFrame
      Dataframe to be uploaded to bigquery
    table_id: str
      table_id where df will be loaded to.
    schema : list[bigquery.SchemaField]
        schema of the table that df must follow. 
        (excluding Ingest_ts which is added in this function)
    sort_by : str
        column name to sort df by. Default is 'ds'.
    """
    import time
    from tasks.task1.utils import fix_column_names
    # Format column names by replacing spaces with dashes, and other fixes.
    df = fix_column_names(df)

    job_config = bigquery.LoadJobConfig(schema=schema)

    # Add variable that records data ingestion time
    df['Ingest_ts'] = datetime.now()

    # Load to bigquery
    df = df.sort_values(by=sort_by) if sort_by is not None else df
    job = client.load_table_from_dataframe(
        df, table_id, job_config=job_config
    )
    
    # Wait until the upload is complete
    while job.state != "DONE":
        print("job status:", job.state)
        time.sleep(3)
        job.reload()
    

def load_latest_data_from_bigquery(client, table_id, limit:int=None) -> pd.DataFrame:
    """
    Loads the data by keeping the latest ingestion timestamp (Ingest_ts) for a given date (ds).
    
    Returns
    -------
    pd.DataFrame
    """
    
    query = f"""
            SELECT * EXCEPT(ds_b, Ingest_ts_b)
            FROM {table_id} as a 
            JOIN (SELECT ds as ds_b, max(Ingest_ts) as Ingest_ts_b
                FROM {table_id}
                GROUP BY ds
                ) as b
            ON a.Ingest_ts = b.Ingest_ts_b AND a.ds = b.ds_b
            ORDER BY b.ds_b
            """
    # Limit the amount of rows
    query = query + f"\nLIMIT {limit}" if limit is not None else query
    
    job = client.query(query)
    df = job.result().to_dataframe()
    return df



def get_modified_date(client):
    query = f"""
            SELECT last_modified_time
            FROM {dataset_id}.__TABLES__ where table_id = {table_id}
            """
    
    job = client.query(query)
    return float(job.result().to_dataframe()["last_modified_time"].item())


def main():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "./credentials.json"
    client = bigquery.Client(project)
    dataset_name = "name"
    table_name = "name"
    create_dataset(client, name=dataset_name)
    create_table(client, project, schema, dataset_name, table_name)
    

if __name__ == "__main__":
    main()