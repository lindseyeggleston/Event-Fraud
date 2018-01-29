import pandas as pd
import re
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

def special_chars(df, col, return_count=False):
    '''
    Creates a new column(s) 'special_chars' containing a dictionary of
    non-alphanumeric and non-whitespace characters (aka special characters) as
    keys and the number of occurances in text as the values.

    Input
    -----
    df: pandas DataFrame
    col: STR - column name that contains text data in string form
    return_count: BOOL - if True, returns a new column 'special_char_count'
        with the total number of special characters contained in the text

    Output
    ------
    None
    '''

    df['special_chars'] = df[col].apply(lambda x: _special_chars(x))
    if return_count:
        df['special_char_count'] = df['special_chars'].apply(lambda x:
                sum(x.values()))

def _special_chars(text):
    '''
    Creates a dictionary of non-alphanumeric and non-whitespace charecters from
    text."
    '''
    if text == '':
        return defaultdict(int)
    special_chars = defaultdict(int)
    for char in text:
        if re.match('[^\w\s\.,]', char) is not None:
            special_chars[char] += 1
    return special_chars

def special_char_count(df, char, col_name):
    '''
    Creates new column(s) containing the counts for a specific special
    character(s) (non-alphanumeric and non-whitespace).

    Input
    -----
    df: pandas DataFrame
    char: STR or LIST - special character(s) to search in text
    col_name: STR or LIST - desired name(s) of columns containing special
        character counts

    Output
    ------
    None
    '''
    assert type(col_name)==type(char), 'char and col_name are different types'

    if 'special_chars' not in df.columns and 'text' in df.columns:
        assert 'text' in df.columns, 'Could not find textual data.'
        special_chars = df['text'].apply(lambda x: _special_chars(x))
    else:
        special_chars = df['special_chars']

    if type(char) == str:
        df[col_name] = special_chars.apply(lambda x: x[char])
    elif type(char) == list:
        assert len(col_name)==len(char), 'char and col_name have different lengths'
        for i, c in enumerate(char):
            col = col_name[i]
            df[col] = special_chars.apply(lambda x: x[c])

if __name__ == "__main__":
    # df = pd.read_pickle('data/clean_data3.pkl')
    # special_chars(df, 'text', return_count=True)
    # special_char_count(df,'!','exclam_count')
    # special_char_count(df, ['@','&'], ['comm_at_count','amp_count'])
    # df.to_pickle('data/clean_data4.pkl')

    # print(df.head())

    df = pd.read_pickle('data/clean_data4.pkl')

    word_mat, feat_names = word_matrix(df,'text')
    W = nnmf(word_mat, 20)
    print (W.head())
