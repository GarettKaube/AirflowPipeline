import sys
from pathlib import Path
sys.path.insert(0, "./src")
import argparse
import datetime as dt
import pandas as pd
import os
import tasks.task1.utils as utils
import logging
import yaml
from tasks.task1.transformers.transformerbase import Transformer

utils.initialize_directories()
logging.basicConfig(filename="./crashes/crash-{}.log".format(dt.date.today()), level=logging.ERROR, 
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger(__name__)

info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_format = logging.Formatter("%(asctime)s - %(message)s")

info_handler.setFormatter(info_format)

logger.addHandler(info_handler)


class SalesDataTransformer(Transformer):
    """ Class for extracting data from "sales_sheet" and transforming it.

    Attributes
    ----------
    sales : pd.DataFrame | None
    year_column_name : str
    week_column_name : str
    month_column_name : str

    Methods
    -------
    extract()
      extracts the sales data from sales_sheet
    transform()
      cleans the sales data from sales_sheet
    pre_aggregation()
      returns a pd.DataFrame of the sales data that has not been aggregated from monthly to yearly sales.
    """
    def __init__(self, 
                 config_path,
                 year_column_name="year_separated", 
                 week_column_name="week_", 
                 month_column_name="month",
        ) -> None:
        """
        Parameters
        ----------
        config_path : str
            location of googlesheet config
        year_column_name : str
          name of the column that indicates the year
        week_column_name : str
          name of the column that indicates the week
        month_column_name : str
          name of the column that indicates the month
        """
        super(SalesDataTransformer, self).__init__(config_path)
        # Get the config for the sales google sheet
        self.sales_yaml = self.get_sheets_by_category("sales_sheet")
         
        self.sales = None
        self.year_column_name = year_column_name
        self.week_column_name = week_column_name
        self.month_column_name = month_column_name

        self.df_sales_monthly = None
        self.sales_pre_aggregation = None


    def extract(self):
        """ Retrieves the sales data
        """
        id = self.sales_yaml.get('id')
        sheet_name = self.sales_yaml.get('sheet_name')
        self.sales = utils.get_spreadsheet(id, sheet_name)


    def transform(self):
        """ Gets sales tracker excel file and converts it to a pd.DataFrame with monthly sales and date time columns
        """

        # Select the sales up to week 52 for each year. index 52 is the END OF YEAR row in the sales_tracker excel file
        
        # First row is the column names
        self.sales.columns=self.sales.iloc[0]
        sales =  self.sales.iloc[1:53].drop(["START", "GOAL", None], axis=1)\
                    .fillna(0.00)\
                    .replace({'': 0.00,})\
                    .replace({'\$': '', ",": ''} , regex=True).astype(float)\
                    .unstack()
        
        sales_df = self.__create_sales_df(sales)      
        
                          
        sales_df_dates = self.__get_date_time_cols(sales_df)

        # Store df_with_new_dates for logging and debugging 
        self.sales_pre_aggregation = sales_df_dates

        df_sales_monthly = self.__aggregate(sales_df_dates)
        
        # Adds date (ds) column in form "Y-M-01"
        df_sales_monthly["ds"] = df_sales_monthly.apply(utils.get_date, axis = 1)
        self.df_sales_monthly = df_sales_monthly
        self.__check_if_year_end()

        self.df_sales_monthly = self.df_sales_monthly.rename(
            {0:"Sales", "0":"Sales"},
            axis=1
        )

        # Set columns name to nothing
        self.df_sales_monthly.columns.name = None
        
        return self.df_sales_monthly
    

    def __create_sales_df(self, sales):
        sales_df = pd.DataFrame(
            sales.values, 
            index = pd.date_range(
                start = "10/5/2015", 
                freq="W-MON", 
                periods=sales.shape[0]
            )
        )\
        .reset_index()\
        .rename({"index": "date"}, axis=1)
        
        return sales_df


    def __get_date_time_cols(self, df:pd.DataFrame):
        """Adds year, month, and datetime columns and sums the sales by month
        Input:
            df: pd.DataFrame of Sales data
        """
        
        # Make column containing the week number of the year where the first week of October is considered the first week here
        df["date"] = pd.to_datetime(df["date"])
        df_with_week = utils.add_week(df)

        # Add column with true week number, the Sales_tracker starts at week 40 in october
        # Add a column containing year pairs (eg. 2021-2022) and year for the given week 
        df_with_year_week = utils.add_week_start_at_40(utils.get_year(df_with_week))

        df_with_year_week[self.year_column_name] = df_with_year_week["date"]\
            .apply(lambda x : x.year)
        
        df_with_year_month_week = utils.get_date_given_week_and_year(
            df_with_year_week, 
            year_col=self.year_column_name, 
            week_col=self.week_column_name)

        # Add modified date using the true week numbers and the year
        df_with_dates = utils.get_date_given_week_and_year( 
            df_with_year_month_week, 
            return_month_only = False, 
            year_col = self.year_column_name, 
            week_col = self.week_column_name
        ) 
        df_with_dates[self.week_column_name] = df_with_dates[self.week_column_name].astype(int)
        return df_with_dates


    def __aggregate(self, df):
        # take the sum of sales for each month
        df_sales_monthly = df[[0, self.year_column_name, self.month_column_name]]\
            .groupby([self.year_column_name, self.month_column_name]).sum()\
            .reset_index()\
            .rename({self.year_column_name:"year"}, axis=1)
        return df_sales_monthly


    def pre_aggregation(self):
        return self.sales_pre_aggregation

    
    def __check_if_year_end(self):
        # For the case if we have some sales data for october of new year start but none beyond
        # Filter out monthly sales that are 0.00 since that means there is no observation
        if self.df_sales_monthly[self.df_sales_monthly.iloc[:,2] != 0.00].iloc[-1, 1] == 9:
            year = dt.date.today().year
            self.df_sales_monthly.loc[len(self.df_sales_monthly)] = [year, 10, 0.00, pd.Timestamp(f"{year}-10-01")]
    

if __name__ == "__main__":
    pass
