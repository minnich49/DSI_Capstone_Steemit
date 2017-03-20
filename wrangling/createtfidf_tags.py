import json
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
from nltk.stem.wordnet import WordNetLemmatizer


lmtzr = WordNetLemmatizer()
class WordNetLemmatizer(object):
    def __init__(self):
        self.lmtzr = lmtzr.lemmatize    
    def __call__(self, doc):
        return [self.lmtzr(t) for t in word_tokenize(doc)]


tfidfvect = TfidfVectorizer(
    encoding = 'utf-8',
    tokenizer = WordNetLemmatizer(),
    # tokenizer = PorterTokenizer(),
    stop_words = stopwords.words('english'),
    lowercase = True
    
)

types_of_x = set()
def create_tfidf(x):
    types_of_x.add(type(x))
    tags = 'EMPTYDOCUMENT'
    if type(x) == str:
        try:
            j = json.loads(x)
            if 'tags' not in j:
                return 'EMPTYDOCUMENT'
            tags = j['tags']
            tags = ' '.join(tags)
        except Exception as e: 
            print "exception:", e
    return tags


print 'READ IN DATA'
# set the data path
data_directory = '../data'

print os.getcwd()
posts_path = os.path.join(data_directory,
                          'posts_raw_cleaned',
                          'posts_raw_cleaned.csv') # full data set

df_posts_full = pd.read_csv(posts_path, na_values=[" "])
df_posts = df_posts_full

print 'TFIDF'
# create the structure for tfidf for the categories
# df_posts['json_metadata'].fillna(' ', inplace=True)

df_posts['tags'] = df_posts.apply(lambda x: create_tfidf(x['json_metadata']), axis=1)


# print "types_of_x:", types_of_x

print "df_posts['tags']:", df_posts['tags']
# exit()

df_posts['tags'].fillna('EMPTYDOCUMENT', inplace=True)


# print "df_posts['tags']:", df_posts['tags']
# exit()

tags_tfidf = tfidfvect.fit_transform(df_posts['tags'])

tags_tfidf_path = os.path.join(data_directory,'tags_tfidf')

joblib.dump(tags_tfidf,
            os.path.join(tags_tfidf_path,'tags_tfidf.pkl'))

joblib.dump(tfidfvect.get_feature_names(),
            os.path.join(tags_tfidf_path,'tags_tfidf_feature_names'))

tags_tfidf_desc_path = os.path.join(tags_tfidf_path, 'tags_tfidf_desc.csv')


df_posts.to_csv(tags_tfidf_desc_path,
                              index=False, 
                              quoting=csv.QUOTE_ALL, 
                              encoding='utf-8')
