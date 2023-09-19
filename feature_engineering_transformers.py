import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

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
