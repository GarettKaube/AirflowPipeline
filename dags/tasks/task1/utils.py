"""Collection of helper functions
"""

import datetime as dt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from google.cloud import bigquery
from tasks.task1.database import load_latest_data_from_bigquery
import os
import numpy as np
import pandas as pd
from googleapiclient.discovery import build  


def initialize_directories():
    """
    Creates required directories.
    """
    import os
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./crashes", exist_ok=True)
    os.makedirs("./artifacts/outputs", exist_ok=True)
    os.makedirs("./artifacts/models", exist_ok=True)
    os.makedirs("./src/model/reports", exist_ok=True)


def add_week(x, start = 1):
    """
    Generates custom week numbers based on specified year start (Oct 1) and end (Sept 30)
    
    Parameters
    ----------
    x : pd.DataFrame
    """
    
    index = np.arange(len(x.index)) + 1
    max_ = len(index) 
    
    for j, i in enumerate(index):
        
        if i > 52:
            # set the next 52 values to 1 to convert the values to be in range 1-52
            for k in range(52):
                if j + k < max_:
                    index[j+k] = 1
            # convert next 52 values to be in range 1-52
            for g in range(1, 52):
                if j + g < max_: # stop when we reach the bottom of the array
                    index[j+g] = 1 + g    
                     
    x['week'] = index

    return x


def add_week_given_size(size):
    """
    Generates custom week numbers based on specified year start ( first monday of Oct) 
    and end (last monday of Sept) October is week 40
    
    Parameters
    ----------
    size: int
        size of dataframe 
    """
    
    index = np.arange(size) + 1
    max_ = len(index) 
    
    for j, i in enumerate(index):
        
        if i > 52:
            # set the next 52 values to 1 to convert the values to be in range 1-52
            for k in range(52):
                if j + k < max_:
                    index[j+k] = 1
            # convert next 52 values to be in range 1-52
            for g in range(1, 52):
                if j + g < max_: # stop when we reach the bottom of the array
                    index[j+g] = 1 + g    
                    
    return index


def add_week_start_at_40(df:pd.DataFrame):
    """ Adds week numbers which starts at 40. Eg: 40, 41,..., 52, 1, 2, ...
    Parameters
    ----------
    df: pd.DataFrame
        dataframe to add week numbers
    """
    index = np.arange(13) + 1
    # The entire sales_tracker data starts at week 40 i.e. the first week on october 
    for i in range(13):
        index[i] = 40 + i
    rest_of_weeks = add_week_given_size(len(df) - 13)
    index = list(index) + list(rest_of_weeks)
    df['week_'] = index

    return df   


def get_year(x):
    """
    Generates year numbers 
    """
    year = [f"{i}-{i+1}" for i in range(2015, dt.date.today().year+1)]
    
    year_list = []
    year_index = 0
    weeks = x['week']
    for j, w in enumerate(weeks):
        # If week is > 51 then we are in the next year next iteration so we increase year_index to go to next year
        if w < 52: 
            year_list.append(year[year_index])
        else:
            year_list.append(year[year_index])
            year_index +=1
        
            
    x['year'] = year_list
    return x


def get_date_given_week_and_year(
        x, 
        return_month_only = True, 
        year_col = "year", 
        week_col = "week"
    ):
    """
    Find the month or date given the year and week number

    Parameters
    ----------
    x : pd.DataFrame
    return_month_only : bool
        if false, date variable is created instead of month
    year_col : str
        name of the year column in x
    week_col : str
        name of the week column in x

    """
    year = x[year_col]
    week = x[week_col]
    
    dates = []
    for i in range(len(year)):
        date = str(year[i]) + '-W' + str(week[i])
        res = datetime.strptime(date + '-1', "%Y-W%W-%w")
        if return_month_only:
            dates.append(res.month)
        else:
            dates.append(res)
    if return_month_only:
        x['month'] = dates
    else:
        x['date'] = dates
    return x


def get_date(x, year_col = 'year', month_col = 'month'):
    """
    Creates timestamp given year and month columns of x
    """
    y = int(x[year_col])
    m = int(x[month_col])
    return pd.Timestamp(y, m, 1)
    


def reformat_model_params(params):
     params = params.reset_index()
     params['index'] = params['index'].str.replace("^", "_")
     params = params.set_index("index", drop=True)[0].to_dict()
     return params


def load_config_yaml_file(path):
    """ Loads the config_path yaml file

    Parameters
    ----------
    config_path: str
        Path to yaml file

    Returns
    -------
    dict
        """
    import yaml
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_spreadsheet(spreadsheet_id:str, sheet_name:str):
    """ Gets the google sheet and returns the pandas dataframe of it.
    Parameters
    ----------
    spreadsheet_id : str 
      id of the google sheet
    sheet_name: str
      name of the sheet in the google sheet.

    Returns
    -------
    pd.DataFrame
    
    """
    from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
    hook = GoogleBaseHook("googlesheets")
    creds = hook.get_credentials()
    service = build("sheets", "v4", credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    result = (
        sheet.values()
        .get(spreadsheetId=spreadsheet_id, range=sheet_name)
        .execute()
    )
    values = result.get("values", [])
    data = pd.DataFrame(values)

    if not values:
        print("No data found.")

    return data



def check_for_sheet_updates(sheet_config_path, table_id) -> dict:
    """ Gets modification time of data from bigquery, gets latest modification time of
    the google sheets, and compares them. if latest data modify time < modification time of a google sheet,
    then the google sheet will be used for updating.
    Returns a dictionary that contains keys "pnl_sheets" and "sales_sheet" with
    lists as their values. If the lists are empty, there are no updates.

    Parameters
    ----------
    sheet_config_path : str 
      path that contains the google sheet ids, sheet names, etc.
    table_id: str
      table_id where data will be retrieved from

    Returns
    -------
    dict:
        dictionary that contains keys "pnl_sheets" and "sales_sheet" with
        lists as their values.
    """
    from sklearn.linear_model import LogisticRegression
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    from datetime import datetime
    from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
    from tasks.task1.database import get_modified_date
    hook = GoogleBaseHook("googlesheets")
    creds = hook.get_credentials()
    
    from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
    
    bq_hook = BigQueryHook(gcp_conn_id="bq", use_legacy_sql=False)
    client = bq_hook.get_client()

    # Store the google sheet info for the updated sheets
    updates = {
        "pnl_sheets": {'year_start':{}}, 
        "sales_sheet": {}
    }

    try:
        latest_transformed_data = load_latest_data_from_bigquery(client, table_id)
        if latest_transformed_data.empty:
            latest_transformed_data = None
    # Handle the case when there is no data stored currently
    except IndexError:
        latest_transformed_data = None

    # Handle the case that we have no data, we will just set the latest file creation date to a date long ago
    if latest_transformed_data is not None:
        # Get creation time        
        # Convert to readable data
        modified_time = get_modified_date(client)
        data_creation_date = pd.Timestamp(
                # Divide modified_time by 1000 since modified_time is in miliseconds
                datetime.fromtimestamp(modified_time/1000.0).strftime('%Y-%m-%d %H:%M:%S'), 
                tz = "US/Mountain"
            )
    else:
        data_creation_date = pd.Timestamp("2000-01-01", tz = "US/Mountain")


    # Get google sheet authorization
    scope = ["https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/spreadsheets"]
    client = gspread.authorize(creds)
    headers = {'Authorization': f'Bearer {creds.get_access_token().access_token}'}

    # Get google sheet configs
    google_sheet_cfg = Path(sheet_config_path)
    with open(google_sheet_cfg, 'r') as f:
            google_config  = yaml.safe_load(f)

    # Get latest changes to the google sheets
    pnl = google_config["pnl_sheets"].get("year_start")
    for year in pnl:
        # Get sheet id
        sheet_id = pnl[year].get('id')
        # Open the google sheet and check the for the latest changes
        modify_time = get_sheet_modify_time(client, headers, sheet_id)

        # Check if there is updates to the data and add it to the updates dictionary
        if modify_time > data_creation_date:
            updates["pnl_sheets"]['year_start'][year] = {
                'id': sheet_id,
                'sales_sheet_name': pnl[year].get('sales_sheet_name'),
                'labour_sheet_name': pnl[year].get('labour_sheet_name'),
            }
    
    # Check the sales sheet
    sales_id = google_config["sales_sheet"].get("id")
    modify_time = get_sheet_modify_time(client, headers, sales_id)

    if modify_time > data_creation_date:
        updates["sales_sheet"] = {
            'id': sales_id,
            'sheet_name': google_config["sales_sheet"].get("sheet_name"),
        }
        
        latest_year = list(pnl.keys())[-1]
        sheet_id = pnl[latest_year].get('id')
        updates["pnl_sheets"]['year_start'][latest_year] = {
                'id': sheet_id,
                'sales_sheet_name': pnl[latest_year].get('sales_sheet_name'),
                'labour_sheet_name': pnl[latest_year].get('labour_sheet_name'),
            }

    return updates


def get_sheet_modify_time(client, headers, id):
    """
    Gets the latest modification time of a google sheet.

    Parameters
    ----------
    client : gspread.client.Client
    headers : dict
      http headers for request
    id : str
      id of the sheet

    Returns
    -------
    pd.Timestamp
    """
    import requests
    wb = client.open_by_url(
            f"https://docs.google.com/spreadsheets/d/{id}/edit#gid=0"
        )

    revisions_uri = f'https://www.googleapis.com/drive/v3/files/{wb.id}/revisions'
    response = requests.get(revisions_uri, headers=headers).json()
    return pd.Timestamp(response['revisions'][-1]['modifiedTime'], tz = "US/Mountain")


def fix_column_names(df:pd.DataFrame) -> pd.DataFrame:
    """
    Formats column names by replacing spaces and other special characters.
    
    Parameters
    ----------
    pd.DataFrame
    
    Returns
    -------
    df: pd.DataFrame
      with renamed columns
    """
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("/", "_and_")
    df.columns = df.columns.str.replace("-", "")
    return df






def main():
    pass

if __name__ == "__main__":
    main()
