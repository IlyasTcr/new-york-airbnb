from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin        
from sklearn.preprocessing import MinMaxScaler, StandardScaler

    
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


class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler_option="none"):
        self.scaler_option = scaler_option

    def fit(self, X, y=None):
        
        if self.scaler_option == "minmax":
            self.scaler = MinMaxScaler().fit(X)
        elif self.scaler_option == "standard":
            self.scaler = StandardScaler().fit(X)
        else:
            self.scaler = None
            
        return self

    def transform(self, X, y=None):
        if self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X              
