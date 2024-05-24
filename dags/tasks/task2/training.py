from datetime import datetime
import pandas as pd
import os
try:
    from tasks.task2.model import ARIMAModel
except ModuleNotFoundError:
    from model import ARIMAModel
from random import randint
import mlflow
import yaml


with open("./dags/tasks/task2/features.yaml", 'r') as f:
    predictor_columns = yaml.safe_load(f)['features']


def dimension_reduction(X:pd.DataFrame, n_components:int=2):
    """Applies principle components to the input X.

    Parameters
    ----------
    X : pd.DataFrame
      Data for PCA
    n_components: int
      Number of components for PCA.
    
    Returns
    -------
    pd.DataFrame
        Dataframe containing the components.
    """
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    X_transformed = pca.fit_transform(X)

    return pd.DataFrame(X_transformed, index=pd.to_datetime(X.index))


def split_target_features(df:pd.DataFrame, predictor_cols, target_col='y'):
    X = df[predictor_cols]
    y = df[target_col]
    return X, y


def transform_data(df:pd.DataFrame, dim_reduction=False):
    """
    Drops NaN's, from df and converts 'ds' column to datetime index.
    Splits the data to features, target format.
    Makes sure the target is non-zero by making sure the target values don't go
    past the train cut-off.

    Parameters
    ----------
    df : pd.DataFrame 
        Dataframe that must have date variable 'ds' that can be converted to datetime
    train_cutoff : int 
        Integer indicating the index of the first non-zero entry of the target

    Returns
    -------
    X_transformed_df : pd.DataFrame
      features with PCA applied if dim_reduction is True
    y : pd.DataFrame
      all y values that are observed
    """
    # Make dates the index
    df.index = pd.DatetimeIndex(df['ds'])
    df = df.dropna()
    # Split target and predictor columns
    X, y = split_target_features(df, predictor_columns)

    # fit principle components and make dataframe with the components and set time index
    X_transformed = dimension_reduction(X) if dim_reduction else X

    return X_transformed, y


def get_train_cutoff(df:pd.DataFrame):
    """ Gets the last index of the data where y is not 0.00 i.e. y is observed

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame that contains the target feature y.

    Returns
    -------
    int:
        Index for df of last oberved y value.
    """
    return df[df['y'] != 0.00].dropna().shape[0] # for any index equal larger than this, y should be 0.00 


def train(data: pd.DataFrame, model_config:dict):
    """ Trains the Arima model given the model settings in model_config

    Parameters
    ----------
    data : pd.DataFrame 
        containing y, ds, and the features specified in features.yaml.
    model_config : dict 
        dictionary containing model settings such as order and intercept.

    Returns
    -------
    model: ARIMAModel 
        ARIMAModel instance that has been trained.
    """
    data_copy = data.copy()

    global train_cutoff

    train_cutoff = get_train_cutoff(data_copy)
    # Prepare data
    X_transformed, y_train = transform_data(data_copy)

    # Train ARIMA model, get predictions, confidence intervals, and residuals
    global model
    model = ARIMAModel(
        order=tuple(model_config["order"]),
        seasonal_order=tuple(model_config["seasonal_order"]),
        with_intercept=model_config["with_intercept"],
    )

    if model_config["scaling"]:
        model.add_scaling_step()
    
    if model_config["poly_features"]:
        model.add_poly_features_step(degree=2)
    
    model.train(X_transformed.iloc[:train_cutoff], y_train)
    
    return model



if __name__ == "__main__":
    pass