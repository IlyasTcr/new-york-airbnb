from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin        

class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, order_dic):
        self.order_dic = order_dic
    
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X):
        X_copy = X.copy()
        for column, lst in self.order_dic.items():
            X_copy[column] = X_copy[column].astype(CategoricalDtype(lst, ordered=True)).cat.codes
        return X_copy.to_numpy()        
