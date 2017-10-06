import numpy as np
import pandas as pd
from datetime import datetime

def convert_time(df, col):
    '''
    Converts time column from timestamp to datetime

    Inputs
    ------
    df: pandas dataframe
    col: STR or LIST - column name(s) where values are timestamps

    Output
    ------
    Returns a new dataframe with time column in datetime format
    '''

    if type(col) == str:
        col = list(col)
    for col_name in col:
        date_lst = []
        for i in range(df.shape[0]):
            date_lst.append(int(datetime.fromtimestamp(df.loc[i,col_name]).strftime('%j')))
        df[col_name] = np.array(date_lst)
    return df


def convert_fraud_col(df, col, drop_col=False):
    '''
    Creates new 'fraud' label column containing dummy variables; 1 for fraud,
    0 for not fraud.

    Inputs
    ------
    df: pandas dataframe
    col: STR - column name where values are strings indicating fraud or not
        fraud
    drop_col: BOOL - if true, will drop col from dataframe

    Output
    ------
    Returns a new dataframe with fraud column labeled 1 for fraudulant event
    and 0 for non fraudulant event.
    '''

    df['fraud'] = df[col].apply(lambda x: 1 if 'fraud' in x else 0)
    if drop_col == True:
        df.drop(col, axis=1, inplace=True)

    return df


def convert_spam_col(df, col, drop_col=False):
    '''
    Creates new 'spam' label column containing dummy variables; 1 for spam,
    0 for not spam.

    Inputs
    ------
    df: pandas dataframe
    col: STR - column name that contains spam labels as a string
    drop_col: BOOL - if True, will drop col from dataframe

    Output
    ------
    Returns a new dataframe with spam column labeled 1 for spam event
    and 0 for non spam event.
    '''

    df['spam'] = df[col].apply(lambda x: 1 if 'spam' in x else 0)

    return df


if __name__ == '__main__':
    df = pd.read_pickle('data/data.pkl')
    df = convert_time(df, ['event_created','event_end','event_published','event_start'])
    df = convert_fraud_col(df, 'acct_type')
    df.drop('acct_type')
