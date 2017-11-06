import pandas as pd
import numpy as np
from datetime import datetime
import re

def description_nlp(df):
    found = re.search('<p>(.+?)</p>') # doesn't quite work; search broken by '\'
    if found:
        text = found.group(1)
        for st in text:
            re.sub('<.*?>', '', st)
        return text


if __name__ == "__main__":
    df = pd.read_pickle('data/clean_data.pkl')
