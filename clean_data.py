import numpy as np
import pandas as pd
from datetime import datetime


def convert_time(df, col, create_duration=False):
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
        if df[col_name].dtypes != 'int64':
            df[col_name] = df[col_name].apply(lambda x: int(x))
        df[col_name] = df[col_name].apply(lambda x: datetime.fromtimestamp(x))
    return df


def create_duration_col(df, new_col_name, start_col, end_col, time_unit='days'):
    '''
    Creates a new column for the time elasped between the values of two current
    time related columns in the dataframe. Returns a float value of the seconds
    elasped between the two events

    Inputs
    ------
    df: pandas dataframe
    new_col_name: STR - name of new column to be constructed
    start_col: STR -
    end_col: STR -
    time_unit: {'seconds', 'minutes', 'hours', 'days'} -

    Output
    ------
    None
    '''

    time_conversion = {'seconds':1, 'minutes':60, 'hours':3600, 'days':86400}
    n = time_conversion[time_unit]

    dur = df[end_col] - df[start_col]
    dur = dur.apply(lambda x: x.total_seconds()/n)
    df[new_col_name] = dur

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


def view_batch_data(df, start_col, end_col):
    '''
    view analytics of few columns
    '''

    # df[[start:end]]
    pass


def clean_data(df):
    '''
    Cleans data columns. Must be run after convert_time funct.
    '''

    df['has_header'] = df['has_header'].fillna(0, inplace=True).apply(lambda x: int(x))

    dur = df['event_end']-df['event_start']
    dur = dur.apply(lambda x: x.total_seconds()/86400)
    df['event_duration'] = dur
    # last resort
    df.dropna(inplace=True)

    return df



if __name__ == '__main__':
    df = pd.read_pickle('data/data.pkl')
    df = convert_time(df, ['event_created','event_end','event_published','event_start'])
    df = convert_fraud_col(df, 'acct_type')
    df.drop('acct_type')
