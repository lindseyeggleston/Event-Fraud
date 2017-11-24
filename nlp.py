import pandas as pd
from collections import defaultdict

def special_char_count(text, return_dict=False):
    '''
    Counts the number of non-alphanumeric and non-whitespace characters in text.

    Input
    -----
    text: STR - string of text
    return_dict: BOOL - if True, returns a dictionary with special characters as
        keys and counts as values.

    Output
    ------
    integar value (and dictionary, if return_dict = True)
    '''

    special_chars = defaultdict(int)
    for char in text:
        if re.match('[\W\S]', char) != None:
            special_chars[char] += 1
    if return_dict:
        return (sum(special_chars.values()), special_chars)
    else:
        return sum(special_chars.values()



if __name__ == "__main__":
    df = pd.read_pickle('data/clean_data3.pkl')
