import pandas as pd
import numpy as np
import ROC_curve as roc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess():
    X = pd.read_pickle('data/clean_data.pkl')
    X['sale_duration'] = X['sale_duration'].fillna(X['sale_duration'].mean())
    y = X.pop('fraud').values
    numeric_columns = ['body_length', 'name_length', 'num_order', 'num_payouts', 'sale_duration', 'sale_duration2', 'user_age']
    ss = StandardScaler()
    X[numeric_columns] = ss.fit_transform(X[numeric_columns])
    X = X[numeric_columns] #TEMPORARY, delete once feature space is finalized!!!
    return train_test_split(X, y, test_size=.2, random_state=42, stratify=y)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess(
