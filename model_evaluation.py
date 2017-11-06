import pandas as pd
import numpy as np
import ROC_curve as roc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

def preprocess(filename):
    X = pd.read_pickle(filename)
    y = X.pop('fraud').values
    numeric_columns = ['body_length', 'name_length', 'num_order', 'num_payouts', 'sale_duration2', 'user_age']
    ss = StandardScaler()
    X[numeric_columns] = ss.fit_transform(X[numeric_columns])
    return train_test_split(X, y, test_size=.1, random_state=42, stratify=y)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess('data/clean_data2.pkl')
    xg = XGBClassifier(scale_pos_weight=10, max_delta_step=1, colsample_bytree=.5, colsample_bylevel=.8, seed=42)
