import sys
sys.path.insert(0, "./src")
import datetime as dt
import pandas as pd
import numpy as np
import tasks.task1.utils as utils
import logging
from googleapiclient.errors import HttpError 
from tasks.task1.transformers.transformerbase import Transformer

utils.initialize_directories()
logging.basicConfig(filename="./crashes/crash-{}.log".format(dt.date.today()), level=logging.ERROR, 
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger=logging.getLogger(__name__)

c_handler = logging.StreamHandler()
c_handler.setLevel(logging.INFO)
c_format = logging.Formatter("%(asctime)s - %(message)s")

c_handler.setFormatter(c_format)

logger.addHandler(c_handler)


class InvoiceTransformer(Transformer):
    """
    Attributes
    ----------
    self.invoices : pd.DataFrame
        stores the transformed data
    self.invoice_yamls : dict
        Config retrieved from the config_path yaml file
    """
    def __init__(self, config_path:str) -> None:
        """
        Parameters
        ----------
        config_path: str
          path to the google sheets config yaml file.

        """
        super(InvoiceTransformer, self).__init__(config_path)
        self.invoices = None
        
        self.invoice_yamls =self.get_sheets_by_category("pnl_sheets")
        

    def transform(self):
        """Cleans the extracted dataframes and merges them.
        """
        cleaned_inv = []
        years = self.invoice_yamls['year_start']
        for year in years:
            try:
                cleaned_inv.append(self.clean_invoice(str(year)))
            except Exception as err:
                print("Try run extract before transforming.")
                logger.error(err)
                raise err

        self.invoices = pd.concat(cleaned_inv)
        # Adjust dataframe for posssible missing entries due to it being near year end
        self.check_if_year_end()

        # remove column name
        self.invoices.columns.name = None

        return self.invoices.reset_index(drop=True)
    

    def extract(self):
        """Iterates over years in self.invoice_yamls to retreive their respective
        googlesheet and stores them in a dictionary self.invoice_dict.
        """
        years = self.invoice_yamls['year_start']
        self.invoice_dict = {}
        for year_start in years:
            invoice_df = self.get_invoice_file(year_start)
            self.invoice_dict[year_start] = invoice_df


    def get_invoice_file(self, year_start):
        # Retrieve the id of the google sheet
        id = self.invoice_yamls['year_start'][year_start].get("id")

        try:
            invoice_df = utils.get_spreadsheet(id, "Income")
        except HttpError as err:
            # Handle a different sheet name
            try:
                invoice_df = utils.get_spreadsheet(id, "Sheet1")
            except HttpError as err:
                print(err)
                print("Consider renaming the sheet to 'Income' or 'Sheet1'.")
                logger.error(err)
            else:
                return invoice_df 
        else:
            return invoice_df 
        

    def clean_invoice(self, year_start):
        """Used by the transform() method to clean the invoice file
        """
        invoice_df = self.invoice_dict[year_start]
        
        year_end = int(year_start) + 1
        if invoice_df is not None:
            invoice = invoice_df.replace({None:np.nan, "#DIV/0!": "100.00%"}).dropna().set_index(0)
            invoice.columns = invoice.iloc[0] # 0-th row contains the column names
            invoice = invoice.iloc[[2]].transpose().drop("TOTAL", axis=0) # Row 3 contains the data
            invoice = invoice[invoice["Total Sales"] != "100.00%"].fillna(0.00).replace({'': 0.00})
            
            # Remove special characters and convert to float
            invoice["Total Sales"] = invoice["Total Sales"].str.replace("$", "")
            invoice["Total Sales"] = invoice["Total Sales"].str.replace(",", "").astype(float)
            
            # Add datetime variables. 
            # Data always start on October of every year and ends on September
            invoice["date"] = pd.date_range(start=f"10/1/{year_start}", end=f"09/30/{year_end}", periods = 12)
            invoice["year"] = invoice["date"].apply(lambda x: x.year)
            invoice["month"] = invoice["date"].apply(lambda x: x.month)
            invoice.fillna(0.00)
            
            
            invoice.reset_index(inplace=True, drop = True)
            return invoice


    def check_if_year_end(self):
        """ 
        Corrects the invoice dataframe at year end by adding more rows so we can 
        forecast further into the future.
        """
        # For the case when we are at the year end, we won"t have a new invoice sheet to be able to forecast values for the next year so we add future dates to forecast
        if self.invoices[self.invoices["Total Sales"] != 0.00]["month"].iloc[-1] == 9:
            logger.info("YEAR END")
            year = dt.date.today().year
            self.invoices.loc[len(self.invoices)] = [0.00, pd.Timestamp(f"{year}-10-01"), year, 10]


def main():
    raise NotImplementedError


if __name__ == "__main__":
    main()
