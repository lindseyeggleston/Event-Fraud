import pandas as pd
import numpy as np
from datetime import datetime

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


def _ticket_spread(lst):
    cost = set()
    for ticket in lst:
        cost.add(ticket['cost'])
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
    pass
