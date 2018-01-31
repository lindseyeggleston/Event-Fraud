import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS

class TopicAnalysis():
    def __init__(self, docs, k, iterations):
        self.docs = docs   # add property decorator to test doc type
        self.k = k
        self.iterations = iterations
        self.V = None
        self.feat_names = None

    # @property
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

    def create_word_matrix(self):
        '''Creates word matrix V from word counts across all each docs.'''
        counts = CountVectorizer(max_features=5000, top_words=stop_words).fit(self.docs)  # ngram_range = (1,3)
        self.V = counts.fit_transform(self.docs).todense()
        self.feat_names = counts.get_feature_names()
        self.W = np.random.rand(self.V.shape[0],self.k)
        self.H = np.random.rand(self.k, self.V.shape[1])

    def nnmf(self):
        nmf = NMF(n_components=self.k, max_iter=self.iterations).fit(self.V)
        W = nmf.transform(self.V)
        topic_matrix = nmf.components_
        pass

    def mse():
        return mean_squared_error(V, self.W.dot(self.H))

    def print_topics(mat, n=5):
        for topic in range(mat.shape[0]):
            indices = mat[topic].argsort()[-1:-n-1:-1]
            top_feat = ', '.join([feat_names[i] for i in indices])
            print ('Top features of {}:'.format(topic), top_feat)

if __name__=='__main__':

    df = pd.read_pickle('data/clean_data4.pkl')
    docs = df['text']

    ## Additional Stop Words
    stop_words = ENGLISH_STOP_WORDS.union({})

    ## Tfidf Version
    tfidf = TfidfVectorizer(max_features=5000, top_words=stop_words, ngram_range = (1,3))
    word_mat = tfidf.fit_transform(docs).todense()

    ## Count Version
    # counts = CountVectorizer(max_features=5000, top_words=stop_words, ngram_range = (1,3))
    # word_mat = counts.fit_transform(docs).todense()
