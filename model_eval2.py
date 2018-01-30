import pandas as pd
import numpy as np
import ROC_curve as roc
import profit_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def preprocess(filename):
    X = pd.read_pickle(filename)
    X['major_country'] = _create_major_country(X)
    X['fb_zero'] = _org_facebook_indicator(X)
    X['check_payout'] = _check_payout_indicator(X)

    numeric_columns = ['body_length', 'name_length', 'sale_duration2', 'event_duration',\
     'user_creation_duration', 'num_ticket_types', 'price_spread', \
     'avg_ticket_price', 'text_length', 'nunique_words']
    # ss = StandardScaler()
    # X[numeric_columns] = ss.fit_transform(X[numeric_columns])

    #create dummies for multi-category variables
    to_dummy_columns = ['channels', 'delivery_method', 'user_type']
    dummies = pd.get_dummies(X[to_dummy_columns], columns=to_dummy_columns)

    #list categorical columns to keep, then create final dataframe
    categorical_columns = ['fraud', 'fb_published', 'has_analytics', 'has_header', 'has_logo', 'listed', 'major_country', 'fb_zero', 'check_payout']
    final_columns = numeric_columns + categorical_columns
    X_final = pd.concat([X[final_columns], dummies], axis=1)
    X_final = X_final.reset_index(drop=True)
    y = X_final.pop('fraud').values
    return X_final, y

def _create_major_country(df):
    return df['country'].apply(lambda x: 1 if x in ['', 'NZ', 'US', 'CA', 'GB', 'AU'] else 0)

def _org_facebook_indicator(df):
    return df['org_facebook'].apply(lambda x: 1 if x>0 else 0)

def _check_payout_indicator(df):
    return df['payout_type'].apply(lambda x: 1 if x=='CHECK' else 0)

def feature_importance(X, y, classifiers, balancing=None):
    '''
    Need to adjust formatting.  Also, check model performance on test set after feature dropout!
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    model_number = 0
    for c in classifiers:
        save_path = ('visuals/model_{}.png'.format(model_number))
        c.fit(X_train, y_train)
        plt.close('all')
        plot_importance(c)
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
        model_number += 1
    return

if __name__ == '__main__':
    X, y = preprocess('data/clean_data3.pkl')
    xg1 = XGBClassifier(seed=42)
    xg2 = XGBClassifier(scale_pos_weight=10, max_delta_step=1, colsample_bytree=.5, colsample_bylevel=.8, seed=42)
    classifiers = [xg1]
    balancing = []
    # roc.plot_ROC_curve(classifiers, X, y, balancing=balancing, save_path=None)
    # profit_curve.plot_avg_profits(xg1, X, y, revenue=75, cost=25, save_path=None)
    feature_importance(X, y, classifiers)
