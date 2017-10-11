import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def prep_data(filename):
    df = pd.read_json(filename)
    df['fraud'] = df['acct_type'].apply(lambda x: 1 if 'fraud' in x else 0)
    df.drop('acct_type', axis=1, inplace=True)
    return df

if __name__ == '__main__':
    df = prep_data('../data/data.json')
