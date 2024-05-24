"""
Module that contains ARIMAsklearn class which wraps around the pmdarima ARIMA model
and also contains the ARIMAmodel class which contains methods for training the ARIMAsklearn model, and other
utility methods.
"""

import pandas as pd
import numpy as np
import pmdarima as pm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


# List of evaulatiuon metrics for ARIMAModel class
metrics = ["MSE", "RMSE", "MAE", "MAPE", "AIC", "Ljung-Box Test Pvalue"]



def root_mse(predicted, true):
    """
    Calculates root mean squared error
    """
    return np.sqrt(mean_squared_error(predicted, true))


# Create a custom sklearn estimator so we can use the Pipeline class from sklearn
class ARIMAsklearn(BaseEstimator, RegressorMixin):
    def __init__(self, order, seasonal_order, intercept) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.intercept = intercept
        self.conf_intervals = None
        self.fitted = None
        self.fitted_conf_interval = None
        self.resid = None
    

    def fit(self, X, y):
        self.model = pm.ARIMA(
            order=self.order, 
            seasonal_order=self.seasonal_order,
            with_intercept=self.intercept
        )
        # Validate array shapes
        check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        # Fit arima model and return
        self.model.fit(y, X)

        # Get and create dataframe of model fitted values
        fitted, self.fitted_conf_interval = self.model.predict_in_sample(
            X, return_conf_int=True
        )
        self.fitted = pd.DataFrame(
            {'y': fitted}, index=y.index
        )
        # Get model residuals
        self.resid = self.model.resid()

        return self
    

    def predict(self, X):
        check_is_fitted(self)
        check_array(X)
        predictions, self.conf_intervals = self.model.predict(
            X=X, 
            n_periods=X.shape[0], 
            return_conf_int=True
        )
        return predictions



class ARIMAModel(object):
    """ ARIMAModel instance will train a sklearn Pipeline with an ARIMAsklearn as the final
    step when the train() method is called.

    Attributes
    ----------
    metrics: dict
      dictionary of various performance metrics of the trained model
    insample_pred: array-like
      contains the fitted values of the model
    X_train : array-like
      data the model was trained on
    y_train : array-like
      labels the model was trained on
    steps: list
      steps of the sklearn pipeline
    seasonal_order : tuple
      seasonal order of the ARIMA model
    with_intercept: bool
      whether the model uses an intercept or not  
    
    Methods
    -------
    add_scaling_step()
    add_poly_features_step(degree)
    train(X, y)
        Trains the model
    get_model()
        Gets the model pipeline object
    predict(X)
        Returns a pd.Series of forecasts.
    get_insample_predictions()
        Returns a pd.Series of fitted values.
    save(path)
        Save model as pickle file to path
    load(path)
        load model pickle file to path
    get_residuals()
        Returns a pd.Series of the model residuals
    get_diagnostic_plot()
        Returns a matplotlib.pyplot of plots of ACF, residuals, and normal Q-Q plot.
    error_metrics(y)
        Returns dict of performance metrics
    get_model_params()
        Gets coeficients of the ARIMA model
    get_scaler_params()
        Returns dict of mean and standard_deviation of the scaler
    features_in()
        Returns names features used to train model
    get_feature_names()
        Returns array of features polynomial_transform made to train model
    
    """
    def __init__(self, order=None, seasonal_order=None, with_intercept=None) -> None:
        """
        Parameters
        ----------
        order : tuple
          order of the ARIMA model
        seasonal_order : tuple
          seasonal order of the ARIMA model
        with_intercept: bool
          whether the model uses an intercept or not  
        """
        # Dictionary to store metrics such as MSE
        self.metrics = {}
        self.insample_pred = None
        self.X_train = None
        self.y_train = None
        self.steps = []

        # Model specifications
        self.order = order
        self.seasonal_order = seasonal_order
        self.intercept = with_intercept


    def add_scaling_step(self):
        """Method for adding a scaling step to the pipeline
        Call before using the train method for scaling.
        """
        self.steps.append(("scale", StandardScaler()))

    
    def add_poly_features_step(self, degree):
        """Method for adding a polynomial feature transform step to the pipeline.
        Call before using the train method for polynomial features.
        """
        self.steps.append(
            ("polynomial_transform", PolynomialFeatures(degree=degree, include_bias=False))
        )


    def train(self, X, y):
        """ Builds the pipeline and trains the ARIMA model
        """
        import warnings
        warnings.filterwarnings("ignore")
        
        self.X_train = X
        self.y_train = y

        self.steps.append(
                ("ARIMA", ARIMAsklearn(
                            order=self.order, seasonal_order=self.seasonal_order, 
                            intercept=self.intercept 
                )
            )
        )

        self.model = Pipeline(self.steps)

        self.model.fit(self.X_train, self.y_train)
        

    def get_model(self):
        return self.model


    def predict(self, X):
        """ Predicts future y given regresors X
        Returns
        -------
        predictions_df : pd.DataFrame:
        intervals : array-like
        """
        predictions = self.model.predict(X)
        predictions_df = pd.DataFrame({'forecast': predictions}, index=X.index)
        intervals = self.model['ARIMA'].conf_intervals
        return predictions_df, intervals


    def get_insample_predictions(self):
        """
        Returns the fitted values of the model and their confidence intervals.
        
        Returns
        -------
        self.insample_pred : pd.Series
        insample_intervals : array-like
        """
        self.insample_pred = self.model['ARIMA'].fitted.rename({'y':0},axis=1)
        insample_intervals = self.model['ARIMA'].fitted_conf_interval
        return self.insample_pred, insample_intervals
    

    def save(self, path=None):
        """
        Saves the model in pickle format
        """
        import pickle
        if not path:
            path = f'./models/model.pickle'
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)


    def load(self, path):
        """
        Loads a saved pickle model
        """
        # Load a dummy ARIMAsklearn instance so pickle can load the model
        import pickle
        print(path)
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        

    def get_residuals(self):
        """
        Returns a pd.Series of the model residuals
        """
        return self.model['ARIMA'].resid 
    

    def get_diagnostic_plot(self):
        """ Returns a matplotlib.pyplot of plots of ACF, residuals, and normal Q-Q plot.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
        ax.grid(axis='y',zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.model['ARIMA'].model.plot_diagnostics(fig=fig)
        
        return fig
    

    def error_metrics(self, y) -> dict:
        """
        model: pm.ARIMA model

        Returns
        -------
        self.metrics: dict
         Dictionary of performance metrics.
        """
        start = self.model['ARIMA'].order[0]
        y_pred = self.insample_pred[start:]
        y_true = y[start:]

        # Calculate mean squared error (MSE), root MSE, mean absolute error, mean absolute percentage error, and AIC
        metric_functions = [mean_squared_error, root_mse, mean_absolute_error, mean_absolute_percentage_error, self.model['ARIMA'].model.aic]
        for i, metric in enumerate(metrics):
            try:
                self.metrics[metric.lower()] = metric_functions[i](y_pred, y_true)
            except TypeError:
                self.metrics[metric.lower()] = metric_functions[i]()
            except IndexError:
                pass
        # Ljung-Box test
        self.metrics["ljung-box test pvalue"] = acorr_ljungbox(self.get_residuals(), lags=[10])['lb_pvalue'].item()

        return self.metrics
    

    def get_model_params(self):
        """
        Get model parameters and rename the parameter names

        Returns
        -------
        dict 
            containing 'model_params', pd.Series of the Arima parameters,
            'scaler_mean', mean of the standard scaler, 'scaler_std',
            standard deviation of the standard scaler.
        """
        try:
            params = self.model['ARIMA'].model.params()
        except AttributeError as e:
            print("Model must be trained before retrieving parameters.")
        else:
        
            features = self.get_feature_names()   # The polynomial transformation will rename the features to x0, x1,...
            feature_names_in = self.features_in() # These are the actual names of the features
            
            try:
                rename = {param_name: actual_name for param_name, actual_name in 
                        zip(params.index.drop(['intercept'])[:len(features)], features)}
            # Handle when there is no intercept
            except KeyError:
                rename = {param_name: actual_name for param_name, actual_name in 
                        zip(params.index[:len(features)], features)}
            
            params = params.rename(rename)
            # Replace the features with their actual names
            params = params.reset_index().replace(
                {f"x{i}": j for i, j in enumerate(feature_names_in)}, 
                regex=True
            )\
            .set_index("index", drop=True).iloc[:,0]

            return {
                'arima_params': params, # pd.Series
                'scaler_mean': self.get_scaler_params()['mean'], # float
                'scaler_std': self.get_scaler_params()['std'], # float
            }
    

    def get_scaler_params(self):
        """
        Returns a dictionary containing mean and standard deviation of the scaler
        """
        try:
            return {
                    'mean': self.model['scale'].mean_, 
                    'std': self.model['scale'].scale_
                }
        # If there is no scaling step
        except Exception as e:
            return None
    

    def features_in(self):
        return self.model['scale'].feature_names_in_
    

    def get_feature_names(self):
        """
        Returns array of features polynomial_transform made to train model
        """
        return self.model['polynomial_transform'].get_feature_names_out()
    
        
            

if __name__ == "__main__":
    pass
    