import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from category_encoders import MEstimateEncoder
from functools import reduce

# Define a string match transformer class
class StringMatchTransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with a list of strings
    def __init__(self, strings):
        self.strings = strings

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to check if any of the provided strings are in each element of X
    def transform(self, X):
        X = pd.DataFrame(X)
        regex = '|'.join(self.strings)
        bools = X[0].str.lower().str.contains(regex)
        return bools.values.reshape(-1,1)


# Define a reviews threshold transformer class
class ReviewsThresholdTransformer(BaseEstimator, TransformerMixin):
    # Initialize the class with a threshold value
    def __init__(self, threshold):
        self.threshold = threshold

    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self

    # Transform method to check if each element of X is less than the threshold
    def transform(self, X):
        X = pd.DataFrame(X)
        X[1] = X[0] < self.threshold
        return X.to_numpy()


# Define a custom target encoder class
class CustomTargetEncoder(BaseEstimator):
    # Initialize the class with columns to encode and a smoothing parameter m
    def __init__(self, encoder_cols, m):
        self.encoder_cols = encoder_cols
        self.m = m
        self.kf = KFold(n_splits=3)
        self.encoder_list = []

    # Fit method for compatibility with sklearn API    
    def fit(self, X, y):
        return self
        
    # Fit transform method to fit and transform the data using KFold cross validation
    def fit_transform(self, X, y):
        dataframe_list = []
        for encode_index, pretrain_index in self.kf.split(X):
            encoder = MEstimateEncoder(cols=[self.encoder_cols], m=self.m)
            encoder.fit(X.iloc[encode_index, :], y.iloc[encode_index])
            dataframe_list.append(encoder.transform(X.iloc[pretrain_index, :]))
            self.encoder_list.append(encoder)
        return pd.concat(dataframe_list).to_numpy()
    
    # Transform method to transform the data using the fitted encoders and average the results
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=[self.encoder_cols])
        data_frames = [encoder.transform(X) for encoder in self.encoder_list]
        result_df = reduce(lambda df1, df2: df1 + df2, data_frames) / len(self.encoder_list)
        return result_df.to_numpy()

    
           