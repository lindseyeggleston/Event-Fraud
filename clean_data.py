import numpy as np
import pandas as pd
from datetime import datetime


def convert_time(df, cols):
    '''
    Converts time column from timestamp to datetime

    Inputs
    ------
    df: pandas dataframe
    cols: STR or LIST - column name(s) where values are timestamps

    Output
    ------
    Returns a new dataframe with time column in datetime format
    '''

    if type(cols) == str:
        cols = list(cols)

    df = df.dropna(subset = cols)
    df = df.reset_index(drop=True)
    for col_name in cols:
        if df[col_name].dtypes != 'int64':
            df[col_name] = df[col_name].apply(lambda x: int(x))
        df[col_name] = df[col_name].apply(lambda x: datetime.fromtimestamp(x))
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
    if drop_col == True:
        df.drop(col, axis=1, inplace=True)

    return df


def clean_data(df, time_cols):
    '''
    Cleans data columns and drops NaN values.

    Inputs
    ------
    df: Pandas DataFrame
    time_cols: LIST or STR - columns to be convert_time into datetime object

    Output
    ------
    A cleaned DataFrame
    '''

    df = convert_time(df, time_cols)
    
    df['has_header'] = df['has_header'].fillna(0, inplace=True).apply(lambda x: int(x))
    df['country'] = df['country'].apply(lambda x: '' if x==None else x)
    df['delivery_method'] = df['delivery_method'].apply(lambda x: int(x))
    df['listed'] = df['listed'].apply(lambda x: 1 if x=='y' else 0)

    df.drop(['sale_duration','venue_country','venue_latitude','venue_longitude',
            'venue_name','venue_state'], axis=1, inplace=True)
    df.dropna(subset=['org_facebook','org_twitter'],inplace=True)

    # last resort
    df.dropna(inplace=True)

    return df



if __name__ == '__main__':
    df = pd.read_pickle('data/data.pkl')
    time_cols = ['event_created','event_end','event_published','event_start',
            'approx_payout_date','user_created']
    df = clean_data(df, time_cols)
    df = convert_fraud_col(df, 'acct_type', True)

    # df.to_pickle('data/clean_data.pkl')
