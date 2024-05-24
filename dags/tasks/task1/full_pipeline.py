""" Module containing Pipeline class which combines data and runs it 
through a sklearn pipeline.

"""
import datetime as dt
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, "./src")
import logging
import yaml


# Set up
logger=logging.getLogger("full_pipeline_logger")

info_handler = logging.StreamHandler()
info_handler.setLevel(logging.INFO)
info_format = logging.Formatter("%(asctime)s - %(message)s")

info_handler.setFormatter(info_format)

logger.addHandler(info_handler)

crash_handler = logging.FileHandler(filename="./crashes/crash-{}.log".format(dt.datetime.now().strftime("%Y-%m-%d_%H_%M")))
crash_handler.setLevel(logging.ERROR)
crash_handler_format = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
crash_handler.setFormatter(crash_handler_format)
logger.addHandler(crash_handler)



today = dt.date.today()
year = today.year
month = today.month



class Pipeline:
    """combines invoices, sales, labour data and runs it 
    through a sklearn pipeline.

    Atributes
    ---------
    sales_invoices: pd.DataFrame
        dataframe from resulting merge of sales and invoice dataframes when
        transform method is called
    df_with_labour: pd.DataFrame
        dataframe from resulting merge of labour and sales_invoices dataframes when
        transform method is called
    full_data: pd.DataFrame
        dataframe of transformed data after the transform method is called
    status: int
    _columns : list
      columns in sales and invoices dataframes to keep
    """
    def __init__(self, invoices:pd.DataFrame, sales:pd.DataFrame, 
                 labour:pd.DataFrame) -> None:
        """
        Parameters
        ----------
        invoices : pd.DataFrame
        sales : pd.DataFrame
        labour : pd.DataFrame

        Returns
        -------
        None
        """
        self.sales_invoices = None
        self.df_with_labour = None
        self.full_data = None

        self.invoices = invoices
        self.sales = sales
        self.labour = labour

        self.get_cutoff_dates(invoices, labour)

        # Status of the job
        self.status = None

        config_path = Path(__file__).parent / "pipeline_configs/full_pipeline_config.yaml"
        with open(config_path, 'r') as f:
            self._columns = yaml.safe_load(f)["columns"]
    
    
    def get_cutoff_dates(self, invoices, labour):
        """Get date of first observation for invoices and labour data and take the max of the two
        so that we know where there will not be any NaNs when data are joined.

        Parameters
        ----------
        invoices : pd.DataFrame
        labour : pd.DataFrame
        """
        self.invoices_date = dt.date(invoices.loc[0, 'year'], invoices.loc[0, 'month'], 1)
        self.labour_date = dt.date(labour.loc[0, 'year'], labour.loc[0, 'month'], 1)
        self.cut_off_date = max(self.invoices_date, self.labour_date)
        self.end = dt.date(labour['year'].iloc[-1], labour['month'].iloc[-1], 1)


    def merge_total_sales_monthly_and_invoices(self):
        """self.invoices right join with self.sales on year and month.
        """
        logger.info("Merging data...")
        sales_invoices = self.invoices.merge(self.sales, on=["year", "month"], how='right')
        self.sales_invoices = sales_invoices.rename({"0":"Sales"}, axis = 1)
        return self.sales_invoices
    

    def merge_sales_invoices_and_labour(self, df):
        """ Merges the self.sales_invoices data frame with the self.labour_df on year, month
        """
        self.df_with_labour = df.merge(self.labour, on=["year", "month"], how='left')
        return self.df_with_labour
        

    def transform(self) -> pd.DataFrame:
        """
        Merges self.invoices, self.sales, self.labour, creates some lagged variables, 
        removes unnecesary columns, and validates data.

        Returns
        -------
        pd.DataFrame: 
            cleaned, merged data
        """
        try:
            from tasks.task1.transformers.sklearn_transformers import transform_data
        except ModuleNotFoundError:
            from transformers.sklearn_transformers import transform_data
            
        # Merge the sales, invoices and labour data
        first_merge = self.merge_total_sales_monthly_and_invoices()
        merged_data = self.merge_sales_invoices_and_labour(first_merge)

        # Columns to keep
        subset_cols = self._columns + list(self.labour.columns.drop(['date', 'year', 'month']))
        self.full_data = transform_data(merged_data, subset_columns=subset_cols,
                            cutoff=self.cut_off_date, end=self.end
                        )
        logger.info("Done")

        # Validate data before returning anything
        self.validation(
            self.full_data.set_index(pd.to_datetime(self.full_data['ds']))
        )

        return self.full_data

    

    def save_output(self, path):
        assert self.full_data is not None, "Must run the pipeline first"
        self.full_data.to_csv(path, index=False)
    

    def validation(self, df:pd.DataFrame) -> None:
        """
        Checks properties of the data to check it was transformed properly.

        Parameters
        ----------
        df : pd.DataFrame
         dataframe to validate
        """
        problems = False
        for col in df.columns.drop('ds'):
            # Check for missing values:
            try:
                if df[col].isna().any():
                    raise ValueError
            except ValueError:
                logger.error(f"{col} column has NaN.")
                problems = True

            # make sure there are no negative values
            if col not in ["Mohawk_Staff__Flooring", "Blind_Installation", "Hardwood_Installation"]:
                try:
                    if (df[col] < 0.00).any():
                        raise ValueError
                except ValueError as err:
                    logger.error(f"{col} column has negative values.")
                    problems = True

            # make sure values in the columns are not very large
            try:
                if ((df[col] >= 999_999_999).any() or (df[col] <= -999_999_999).any()):
                    raise ValueError
            except ValueError as err:
                logger.error(f"{col} column has very large values.")
                problems = True   

            # Make sure data is correct data type 
            try:
                df[col].astype(float)
            except Exception as err:
                logger.error(f"{col} has non compatible data types.")
                problems = True

        # Make sure 'ds' column can be converted to date
        try:
            pd.to_datetime(df['ds'])
        except Exception as err:
            logger.error(f"Unable to convert ds column to datetime.")
            problems = True

        # Exit if there are data problems
        if problems:
            self.status = "Failed"
            sys.exit("Exiting due to data validation problems.\nCheck crashes for details.")
        else:
            "Complete"



def main():
    pass


if __name__ == "__main__":
    main()

