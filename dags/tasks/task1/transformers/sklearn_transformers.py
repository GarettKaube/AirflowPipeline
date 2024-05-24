from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import pandas as pd
import sys
sys.path.insert(0, "./src")
from tasks.task1.utils import fix_column_names


def lag_var(df:pd.DataFrame, variable:str=None, lags:int=None):
    """
    Creates lagged versions of the specified variables
    """
    df = df.copy()
    if variable == None or lags == None:
        return df
    for i in range(1,lags+1):
        df[f"{variable}_lagged{i}"] = df[f"{variable}"].shift(i)
    return df


def shift_variable(df:pd.DataFrame, variable:str=None, n_shifts:int=None):
    """ Lags variable in df n_shift times.
    """
    df = df.copy()
    if variable == None or n_shifts == None:
        return df
    
    df[f"{variable}"] = df[f"{variable}"].shift(n_shifts)
    return df


def subset_cols(df, columns=None):
    if columns == None:
        return df
    return df[columns]


def convert_data_type(X, variable=None, type=None):
    if variable == None or type== None:
        return X
    X[variable] = X[variable].astype(type)
    return X


def remove_empty_rows(df, cutoff=None, end=None):
    """
    Removes empty rows at the bottom of the dataframe
    """
    if cutoff == None and end == None:
        return df
    import datetime as dt
    today = dt.date.today()
    year = today.year
    month = today.month
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds')
    #- pd.DateOffset(months=1)
    df = df.loc[cutoff:min(end, dt.date(year, month, 1))].reset_index()

    # Replace NaNs we just created in previous line with 0.00
    df["y"].iloc[-3:] = [0.00]*3
    if df['Sales'].iloc[-1] == 0:
        df = df.iloc[:-1]
    return df


def rename_var(df, variable, new_name):
    return df.rename({variable: new_name}, axis=1)


def transform_data(X, subset_columns:list, cutoff, end):
    """
    Sklearn pipeline that runs some transformations on the data.
    
    Parameters
    ----------
    X : pd.DataFrame
      data for pipeline.
    subset_columns : list
      columns to select from X.
    cutoff : datetime.date
      The earliest date present in X.
    end : datetime.date
      The final date present in X.
    
    Returns
    -------
    pd.DataFrame
    """

    pipeline = Pipeline(
        steps = [
            ("shift_target_var", FunctionTransformer(
                    shift_variable, kw_args= {'variable': "Total Sales",
                                            'n_shifts': -3}
                )
            ),
            ("convert_date_to_string", FunctionTransformer(
                    convert_data_type, kw_args={'variable': "ds",
                                                "type": str}
                )
            ),
            ("lag_sales", FunctionTransformer(
                    lag_var, kw_args={"variable": "Sales",
                                    "lags": 3}
                )
            ),
            ("Rename_target", FunctionTransformer(
                    rename_var, kw_args={"variable":"Total Sales",
                                         "new_name": "y"}
                )
            ),
            ("Subset_columns", FunctionTransformer(
                    subset_cols, kw_args={"columns": subset_columns}
                )
            ),
            ("remove_rows_with_only_0s", FunctionTransformer(
                    remove_empty_rows, kw_args={"cutoff": cutoff,
                                                "end":end}
                )
            ), 
            ("Fix column names", FunctionTransformer(
                    fix_column_names
                )
            )
        ]
    )

    return pipeline.transform(X)

if __name__ == "__main__":
    pass