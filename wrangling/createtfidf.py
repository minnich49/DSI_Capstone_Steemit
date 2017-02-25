
import pymssql
import pandas as pd
import numpy as np
import os
import re
import joblib
import csv
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import word_tokenize 
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


porter =  PorterStemmer()
class PorterTokenizer(object):
    def __init__(self):
        self.porter = porter.stem
    def __call__(self, doc):
        return [self.porter(t) for t in word_tokenize(doc)]

countvect = CountVectorizer(
    encoding = 'utf-8',
    tokenizer = PorterTokenizer(),
    stop_words = stopwords.words('english'),
    lowercase = False
    
)

tfidfvect = TfidfVectorizer(
    encoding = 'utf-8',
    tokenizer = PorterTokenizer(),
    stop_words = stopwords.words('english'),
    lowercase = False
    
)

print 'READ IN DATA'
# set the data path
data_directory = '../data/'


posts_path = os.path.join(data_directory,'posts_raw_cleaned','posts_raw_cleaned.csv') # full data set

df_posts_full = pd.read_csv(posts_path, na_values=[" "])
df_posts = df_posts_full

print 'TFIDF'

# df_posts['body'] = df_posts['body'].dropna() 
df_posts["body"].fillna(' ', inplace=True)
# posts_counts = countvect.fit_transform(df_posts['body'])
posts_tfidf = tfidfvect.fit_transform(df_posts['body'])

# posts_counts_path = os.path.join(data_directory,'posts_counts', 'posts_counts')
posts_tfidf_path = os.path.join(data_directory,'posts_tfidf','posts_tfidf_full.pkl')
joblib.dump(posts_tfidf,posts_tfidf_path)
# joblib.dump(tfidfvect.get_feature_names(),posts_tfidf_path+'_feature_names')

posts_tfidf_desc_path = os.path.join(data_directory,
                                             'posts_tfidf', 
                                             'posts_tfidf_desc.csv')


df_posts.to_csv(posts_tfidf_desc_path,
                              index=False, 
                              quoting=csv.QUOTE_ALL, 
                              encoding='utf-8')