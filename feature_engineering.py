import pandas as pd
import numpy as np
from datetime import datetime
from bs4 import BeautifulSoup

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
    cost = set()
    for ticket in lst:
        cost.add(ticket['cost'])
    if len(cost) == 0:
        return -1
    spread = max(cost) - min(cost)
    return spread


def _avg_ticket_price(lst):
    cost = []
    num_tickets = []
    price = 0
    for ticket in lst:
        cost.append(ticket['cost'])
        num_tickets.append(ticket['quantity_total'])
    for i in range(len(cost)):
        price += cost[i] * num_tickets[i]/sum(num_tickets)
    return price


def _percent_tickets_sold(lst):
    sold = 0
    total = 0
    for ticket in lst:
        sold += ticket['quantity_sold']
        total += ticket['quantity_total']
    if total != 0:
        return (total - sold)/total
    else:
        return 0


def extract_ticket_info(df, col='ticket_types', drop_col=False):
    df['num_ticket_types'] = df[col].apply(lambda x: len(x))
    df['price_spread'] = df[col].apply(lambda x: _ticket_spread(x))
    df['avg_ticket_price'] = df[col].apply(lambda x: _percent_tickets_sold(x))

    if drop_col == True:
        df.drop(col, axis=1, inplace=True)

def _extract_text(st):
    soup = BeautifulSoup(st,'html.parser')
    p = soup.find_all('p')
    text = ''
    for i in p:
        text += i.text
    return text

def _unique_words(st):
    words = st.split()
    word_dic = {(word, 0) for word in set(words)}
    for word in words:
        word_dic[word] += 1
    return word_dic

def extract_description_text(df, description_col, new_col, unique_words=False,
        return_df=False, drop_col=False):
    temp = [_extract_text(entry) for entry in df[description_col]]
    df[new_col] = temp

    df['word_length'] = df[new_col].apply(lambda x: len(x.split()))
    df['nunique_words'] = df[new_col].apply(lambda x: len(set(x.split())))

    if unique_words:
        df['unique_words'] = df[new_col].apply(lambda x: _unique_words(x.split()))
    if drop_col:
        df.drop(description_col, axis=1, inplace=true)
    if return_df:
        return df


if __name__ == '__main__':
    df = pd.read_pickle('data/clean_data.pkl')
    dur_cols = {'event_duration':['event_start','event_end'], 'user_duration':
            ['user_created','event_created']}
    df = create_duration_cols(df, dur_cols)
    extract_ticket_info(df, drop_col=True)
    df.to_pickle('data/clean_data2.pkl')
