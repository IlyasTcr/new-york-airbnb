import numpy as np
import itertools
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
from tensorflow.keras import layers

# Define a custom Keras model 
class KerasModel(BaseEstimator, TransformerMixin):
    # Initialize the model with default parameters
    def __init__(self, n_layers=3, n_units=512, dropout_rate=0.2, input_shape=[11], activation="relu",
                 output_units=1, output_activation="linear", optimizer="adam", loss="mae"):
        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.activation = activation
        self.output_units = output_units
        self.output_activation = output_activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = None
        self.history = None

    # Build the model architecture and compile it
    def _build_model(self):
        model = keras.Sequential()
        model.add(layers.Dense(units=self.n_units, activation=self.activation, input_shape=self.input_shape))
        model.add(layers.Dropout(rate=self.dropout_rate))
        for _ in range(self.n_layers-1):
            model.add(layers.Dense(units=self.n_units, activation=self.activation))
            model.add(layers.Dropout(rate=self.dropout_rate))

        model.add(layers.Dense(units=self.output_units, activation=self.output_activation))

        model.compile(optimizer=self.optimizer, loss=self.loss)

        return model

    def fit(self, X, y, **fit_dict):
        self.model = self._build_model()
        self.history = self.model.fit(X, y, **fit_dict)
        return self

    def predict(self, X):
        return self.model.predict(X)


# Generate a grid of parameters for grid search
def generate_grid(grid_params):
    keys, values = zip(*grid_params.items())
    param_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return param_dicts


# Perform a grid search over the parameters of a pipeline using a neural network model with validation data for early stopping
def custom_NN_grid_search(pipeline, grid_params, X_train, X_valid, y_train, y_valid, valid_step,
                          fit_dict, model_type, metric_function):
    """
    Perform a grid search over the parameters of a pipeline using a neural network model with validation data for early stopping.

    Args:
        pipeline (sklearn.pipeline.Pipeline): The pipeline to optimize.
        grid_params (dict): Dictionary of hyperparameter grids.
        X_train (pd.DataFrame): Training data (DataFrame).
        X_valid (pd.DataFrame): Validation data (DataFrame).
        y_train (pd.Series): Training labels (Series).
        y_valid (pd.Series): Validation labels (Series).
        valid_step (str): The name of the last step in the pipeline before the transformed data is given to the model
        fit_dict (dict): Dictionary of additional fit parameters.
        model_type (str): Type of the model
        metric_function (callable): The metric function to evaluate models.

    Returns:
        list: A list of tuples containing optimized pipelines, metrics, and parameter dictionaries.
    """
    param_dicts = generate_grid(grid_params)
    results = []
    for dictionary in param_dicts:
        pipeline.set_params(**dictionary)
        # Using the pipeline to transform the validation data used for early stopping 
        valid_pipeline = pipeline.named_steps[valid_step].fit(X_train, y_train)
        pipeline.fit(X_train, y_train, model__validation_data=(valid_pipeline.transform(X_valid), y_valid), **fit_dict)
        predictions = pipeline.predict(X_valid)
        if model_type == "bin_class":
            predictions = predictions >= 0.5
        elif model_type == "multi_class":
            predictions = np.argmax(predictions, axis=1)
        metric = metric_function(y_valid, predictions)
        results.append((copy.deepcopy(pipeline), metric, dictionary))
    return results

