import os
import sys
import csv
import pymssql
import pandas as pd
import numpy as np
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
import HTMLParser

#########################
# LOAD DATA
########################

print 'READ IN DATA'
# set the data path
data_directory = '../data/'
posts_path = os.path.join(data_directory,'sample_29k_pos_values.csv') # full data set
df_posts_full = pd.read_csv(posts_path)
df_posts = df_posts_full

#########################
# Combined data
########################
print 'CREATING A NEW CLEAN DATASET'

# Combine multiple updates to articles to get one body per post
combined_body = df_posts.groupby(['author','permlink']).agg(lambda x: ''.join(set(x))).reset_index()
combined_body = combined_body.ix[:,['body','author','permlink']]

# Remove Duplicates
idx_not_duplicates = ~df_posts.duplicated(['author','permlink'])
df_posts = df_posts.ix[idx_not_duplicates,:]
df_posts.drop('body',axis = 1,inplace=True)


df_posts = pd.merge(df_posts,combined_body,on=['author','permlink'])

df_posts.sort_values(by='total_payout_value',ascending=False,inplace=True)


#################################
# Create new features
#################################
print 'CREATING NEW FEATURES'
expression = r'http\S+'
# Extract all Links
df_posts['body urls'] = df_posts['body'].str.findall(expression)
df_posts['number of body urls'] = df_posts['body urls'].apply(len)
df_posts['number of youtube urls'] = (df_posts.ix[:,'body urls']
                                      .str.join(' ')
                                      .str.replace('\.','')
                                      .str.count('youtube'))

df_posts['number of image urls'] = (df_posts.ix[:,'body urls']
                                    .str.join(' ')
                                    .str.count('jpg|png|gif|jpeg'))


expression = '#(\S+)'
# Extract all Hash Tages
df_posts['body tags'] = df_posts['body'].str.findall(expression)
df_posts['number of body tags'] = df_posts['body tags'].apply(len)

expression = '@(\S+)'
# Extract all Hash Tages
df_posts['body mentions'] = df_posts['body'].str.findall(expression)
df_posts['number of body mentions'] = df_posts['body mentions'].apply(len)


############################
# CLEAN DATA
############################
print 'CLEANING DATA'
#START CLEANING TEXT -- 
# 1. html removal
# 2. words, and numbers
# 3. symbols ',!
# 4. decode, encode?

# Remove Links
expression = r'http\S+'
df_posts['body'] = df_posts['body'].str.replace(expression,'')

# Remove all Hash Tags
expression = '#(\S+)'
df_posts['body'] = df_posts['body'].str.replace(expression,'')

# Remove all Tags
expression = '@(\S+)'
df_posts['body'] = df_posts['body'].str.replace(expression,'')

def removesymbols(x):
    x = re.sub("[\W\d]+"," ", x.strip())
    x = str(x)
    return x

# Keep only letters and numbers
# df_posts['body'] = df_posts['body'].apply(lambda x: removesymbols(x), axis=1)
df_posts['body'] = df_posts.apply(lambda x: removesymbols(x['body']), axis=1)

# Remove ascii stuff
df_posts['body'] = df_posts['body'].str.decode('unicode_escape').str.encode('ascii', 'ignore')
# df_posts['body'] = df_posts.apply(lambda row: row['body'].decode('unicode_escape').encode('ascii', 'ignore'), axis=1)


# Remove all periods
# expression = '\.'
# df_posts['body'] = df_posts['body'].str.replace(expression,' ')

# # # Remove all new lines
# expression = r'\n'
# df_posts['body'] = df_posts['body'].str.replace(expression,' ')


# # Remove Any Capital Letter by themselves A, B, C, D etc
# expression = r'\b[A-Z]\b'
# df_posts['body'] = df_posts['body'].str.replace(expression,'')

# # Remove double spaces
# expression = ' +'
# df_posts['body'] = df_posts['body'].str.replace(expression,' ')

# # Remove pure numerical values that have greater than 5 digits
# expression = r'\b[0-9]{5,100}\b'
# df_posts['body'] = df_posts['body'].str.replace(expression,'')

# # Remove all non alpha numeric
# expression = '[^A-Za-z0-9 ]+'
# df_posts['body'] = df_posts['body'].str.replace(expression,'')

# expression = '0A0A'
# df_posts['body'] = df_posts['body'].str.replace(expression,' ')


posts_raw_cleaned = os.path.join(data_directory,
                                             'posts_raw_cleaned', 
                                             'posts_raw_cleaned.csv')
print 'READING TO FILE'
df_posts.to_csv(posts_raw_cleaned,
                              index=False, 
                              quoting=csv.QUOTE_ALL, 
                              encoding='utf-8')