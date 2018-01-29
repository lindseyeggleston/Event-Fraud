import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class TopicAnalysis():
    def __init__(self, docs=None, max_feat=5000):
        self.docs = self._test_doc_type(docs)
        self.word_mat = None
        self._tfidf = TfidfVectorizer(max_features=max_feat, stop_words='english',
                ngram_range = (1,3))
        self.feat_names = None

    def _test_doc_type(self, docs, dtype=object):
        '''Tests the data type of docs argument.'''
        if isinstance(docs, pd.core.series.Series):
            return docs.values()
        elif isinstance(docs, np.ndarray):
            return docs
        elif docs == None:
            return np.array([], dtype=dtype)
        else:
            raise ValueError('Inappropriate input type for docs.')

    def add_doc(self, doc):
        '''Adds doc (string type) to docs attribute.'''
        np.append(self.docs, np.array(doc).reshape(1,1), axis=0)

    def create_word_matrix(self):
        self.word_mat = self._tfidf.fit_transform(self.docs).todense()
        self.feat_names = self._tfidf.get_feature_names()
        
