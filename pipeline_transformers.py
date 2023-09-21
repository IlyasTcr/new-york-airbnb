from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin        
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Define a custom ordinal encoder class
class CustomOrdinalEncoder(BaseEstimator, TransformerMixin):
    # Initialize the class with an order dictionary
    def __init__(self, order_dic):
        self.order_dic = order_dic
    
    # Fit method for compatibility with sklearn API
    def fit(self, X, y=None):
        return self 
    
    # Transform method to encode categorical features as ordinal numbers
    def transform(self, X):
        X_copy = X.copy()
        for column, lst in self.order_dic.items():
            X_copy[column] = X_copy[column].astype(CategoricalDtype(lst, ordered=True)).cat.codes
        return X_copy.to_numpy()  


# Define a custom scaler class
class CustomScaler(BaseEstimator, TransformerMixin):
    # Initialize the class with a scaler option
    def __init__(self, scaler_option="none"):
        self.scaler_option = scaler_option

    # Fit method to fit the appropriate scaler based on the option provided
    def fit(self, X, y=None):
        
        if self.scaler_option == "minmax":
            self.scaler = MinMaxScaler().fit(X)
        elif self.scaler_option == "standard":
            self.scaler = StandardScaler().fit(X)
        else:
            self.scaler = None
            
        return self

    # Transform method to scale the data using the fitted scaler
    def transform(self, X, y=None):
        if self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X
      
