import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from category_encoders import MEstimateEncoder


class StringMatchTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strings):
        self.strings = strings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert numpy array to pandas Dataframe
        X = pd.DataFrame(X)
        regex = '|'.join(self.strings)
        bools = X[0].str.lower().str.contains(regex)
        
        # Convert Series back to numpy array
        return bools.values.reshape(-1,1)

    
class ReviewsThresholdTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Convert numpy array to pandas Dataframe
        X = pd.DataFrame(X)
        X[1] = X[0] < self.threshold
        
        # Convert Series back to numpy array
        return X.to_numpy()


class CustomTargetEncoder(BaseEstimator):
    def __init__(self, encoder_cols, m):
        self.encoder_cols = encoder_cols
        self.m = m
        self.kf = KFold(n_splits=3)
        self.encoder_list = []
        
        
    def fit(self, X, y):
        return self
        
    def fit_transform(self, X, y):
        dataframe_list = []
        for encode_index, pretrain_index in self.kf.split(X):
            encoder = MEstimateEncoder(cols=[self.encoder_cols], m=self.m)
            encoder.fit(X.iloc[encode_index, :], y.iloc[encode_index])
            dataframe_list.append(encoder.transform(X.iloc[pretrain_index, :]))
            self.encoder_list.append(encoder)
        return pd.concat(dataframe_list).to_numpy()
    
    def transform(self, X, y=None):
        X = pd.DataFrame(X, columns=[self.encoder_cols])
        data_frames = [encoder.transform(X) for encoder in self.encoder_list]
        result_df = reduce(lambda df1, df2: df1 + df2, data_frames) / len(self.encoder_list)
        return result_df.to_numpy()
    
           