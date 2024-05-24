import datetime as dt
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, "./src")
from tasks.task1.utils import get_spreadsheet
import warnings
import logging
import yaml
from googleapiclient.errors import HttpError 
from tasks.task1.transformers.transformerbase import Transformer


logging.basicConfig(filename="./crashes/crash-{}.log".format(dt.date.today()), level=logging.ERROR, 
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(asctime)s - %(message)s")

c_handler.setFormatter(c_format)

logger.addHandler(c_handler)


def generate_months():
    """ Creates a list of month name strings starting at October and 
    ending at September: ["October", ..., "September"]

    Returns
    -------
    list:
        list of month name strings
    """
    import calendar
    return [calendar.month_name[i] for i in list(range(10,13)) + list(range(1, 10))]


class LabourTransformer(Transformer):
    """
    Interfaces with google sheets and lags all the variables once

    Attributes
    ----------
    labour_df : pd.DataFrame
        dataframe of the cleaned labour data. Will be None if transform()
        has not been called.
    _invoice_yamls : dict
        dictionary containing all the years and googlesheet information.
    """
    def __init__(self, config_path) -> None:
        super(LabourTransformer, self).__init__(config_path)
        self.labour_df = None

        self._invoice_yamls = self.get_sheets_by_category("pnl_sheets")
        
    
    def transform(self):
        """Cleans labour files

        Returns
        -------
        pd.DataFrame:
            Cleaned and merged labour data.
        """
        labour_list = []
        try:
            for year_start in self.labour_dict:
                labour_df = self.labour_dict[year_start]
                print(labour_df)
                labour_df_clean = self.clean_labour_data(labour_df)
                labour_df_clean = self.add_date_vars(labour_df_clean, year_start)
                if labour_df_clean is not None:       
                    if type(labour_df_clean) != str:
                        labour_list.append(labour_df_clean)
        except NameError as e:
            print("Run extract first", e)
            logger.error(e)
            raise e
        else:    
            labour = pd.concat(labour_list)
            self.labour_df = labour[labour.columns.dropna()]
            
            # Lag the features forward once as we will not have the current month values right away
            for col in self.labour_df.columns.drop(['date', 'year', 'month']):
                self.labour_df[col] = self.labour_df[col].shift(1)
            
            self.labour_df.columns.name = None

            # Do [1:] since the entire first row will be 0.00
            return self.labour_df.fillna(0.00)[1:]\
                .reset_index(drop=True)
    

    def extract(self):
        """Iterates over years in self._invoice_yamls to retreive their respective
        googlesheet and stores them in a dictionary self.labour_dict.

        Returns
        -------
        None
        """
        self.labour_dict = {}
        years = self._invoice_yamls['year_start']
        for year in years:
            sheet_name = self._invoice_yamls['year_start'][year].get("labour_sheet_name")
            id = self._invoice_yamls['year_start'][year].get("id")
            try:
                labour_df = get_spreadsheet(id, sheet_name)
            except HttpError as err:
                print(err)
            else:
                self.labour_dict[year] = labour_df
        

    def clean_labour_data(self, df):
        """ Cleans df by removing special characters, dealing with NaN's,
        dropping unneeded columns/rows, and more.

        Parameters
        ----------
        df : pd.DataFrame
          dataframe to be cleaned

        Returns
        -------
        pd.DataFrame:
            clean data.
        """
        df = df.replace({None, np.nan}).dropna().iloc[:-1] 

        # First row contains column names
        df.columns = df.iloc[0]

        df.set_index("SALES", inplace=True)
        df = df.loc["SALES":"TOTAL", "OCT.":"TOTAL"].drop(["SALES", "TOTAL"])
        # Keep even columns and drop TOTAL column
        df = df.iloc[:, [i%2 == 0 for i in range(df.shape[1])]]\
            .drop(["TOTAL"], axis=1)\
            .replace({"\$": '', ",": '', "": "0.00"}, regex=True)\
            .astype(float)

        # Make the dataframe column names the month names. Starting from october
        df.columns = generate_months()
        df = df.fillna(0.00).transpose()
        return df.reset_index(drop=True)


def add_date_vars(self, df, year_start):
    """ adds date, year, and month variables to df.

    Parameters
    ----------
    df : pd.DataFrame
          dataframe to add the variables to.
    year_start: int

    Returns
    -------
    pd.DataFrame
    """
    # Create date time variables
    df["date"] = pd.date_range(start=f"10/1/{year_start}", end=f"09/30/{int(year_start) + 1}", periods = 12)
    df["year"] = df["date"].apply(lambda x: x.year)
    df["month"] = df["date"].apply(lambda x: x.month)

    return df

def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
