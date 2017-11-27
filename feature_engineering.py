import pandas as pd
import numpy as np
import re
from datetime import datetime
from bs4 import BeautifulSoup
from collections import defaultdict

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


def create_duration_cols(df, cols, time_unit='days'):
    '''
    Creates multiple time duration related columns using the create_duration_col
    function.

    Inputs
    ------
    df: Pandas DataFrame
    cols: DICT - a dictionary with all string values. the keys represent a new
        column name and the values are lists containing the start and stop
        columns from the dataframe; e.g. [start_time, stop_time].
    time_unit: {'seconds', 'minutes', 'hours', 'days'} -

    Output
    ------
    A dataframe with time duration related columns
    '''

    for k, v in cols.items():
        create_duration_col(df, k, v[0], v[1], time_unit)
    return df


def _ticket_spread(lst):
    '''
    Calculates the price difference between the most and least expensive ticket
    types
    '''
    cost = set()
    for ticket in lst:
        cost.add(ticket['cost'])
    if len(cost) == 0:
        return 0
    spread = max(cost) - min(cost)
    return spread


def _avg_ticket_price(lst):
    '''
    Calculated the average ticket price
    '''
    cost = 0
    num_tickets = 0
    for ticket in lst:
        cost += ticket['cost']*ticket['quantity_total']
        num_tickets += ticket['quantity_total']
    if num_tickets == 0:
        return 0
    return round(sum(cost)/num_tickets, 2)


def _percent_tickets_sold(lst):
    '''
    Returns the percentage of tickets sold out of total available
    '''

    sold = 0
    total = 0
    for ticket in lst:
        sold += ticket['quantity_sold']
        total += ticket['quantity_total']
    if total != 0:
        return (total - sold)/total
    else:
        return 0

def _free_event(ticket_types):
    '''
    Check cost of tickets for free event.
    '''
    free = 0
    for ticket_type in ticket_types:
        if ticket_type['cost'] == 0:
            free = 1
    return free

def extract_ticket_info(df, col='ticket_types', drop_col=False):
    '''
    Extract ticket information such as the spread between ticket price values,
    number of ticket types, and the average ticket price

    Inputs
    ------
    df: pandas DataFrame
    col: STR - (default = 'ticket_types') name of column containing ticket
        information
    drop_col: BOOL - if True, removes 'col' from df

    Output
    ------
    pandas DataFrame
    '''

    df['num_ticket_types'] = df[col].apply(lambda x: len(x))
    df['price_spread'] = df[col].apply(lambda x: _ticket_spread(x))
    df['avg_ticket_price'] = df[col].apply(lambda x: _percent_tickets_sold(x))

    if drop_col == True:
        df.drop(col, axis=1, inplace=True)
    return df

def _extract_text(st):
    '''
    Extract text from <p> tags in html string
    '''

    soup = BeautifulSoup(st,'html.parser')
    text = soup.text
    text = re.sub('\s+',' ', text).strip()

    soup2 = BeautifulSoup(text, 'lxml')
    text = soup2.text

    return text

def _unique_words(text):
    '''
    Creates dictionary of unique words (keys) and their occurances (values)
    in text
    '''

    words = re.sub('[\.,\?:!"\(\)]', '', text.lower()).split()
    word_dic = defaultdict(int)
    for word in words:
        word_dic[word] += 1
    return word_dic

def extract_description_text(df, description_col, unique_words=False,
        drop_col=False):
    '''
    Extracts text data from description_col and engineers new columns related to
    that text.

    Inputs
    ------
    df: pandas DataFrame
    description_col: STR - name of column containing html content is string form
    unique_words: BOOL - when True, creates a new column containing dictionary
        of unique_words and the number of there occurances
    drop_col: BOOL - when True, will remove description_col from DataFrame

    Outputs
    -------
    pandas DataFrame
    '''

    temp = [_extract_text(entry) for entry in df[description_col]]

    df['text'] = temp
    df['text_length'] = df['text'].apply(lambda x: len(x.split()))
    df['nunique_words'] = df['text'].apply(lambda x: len(_unique_words(x)))

    if unique_words:
        df['unique_words'] = df['text'].apply(lambda x: _unique_words(x))
    if drop_col:
        df = df.drop(description_col, axis=1)

    return df


if __name__ == '__main__':
    df = pd.read_pickle('data/clean_data.pkl')
    dur_cols = {'event_duration':['event_start','event_end'], 'user_creation_duration':
            ['user_created','event_created']}
    df = create_duration_cols(df, dur_cols)
    df = extract_ticket_info(df)
    df = extract_description_text(df, 'description')
    # df.to_pickle('data/clean_data3.pkl')
