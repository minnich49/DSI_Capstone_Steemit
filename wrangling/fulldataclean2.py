import os
import sys
import csv
import pandas as pd
import re
import joblib
import nltk
from nltk.tokenize import word_tokenize
import HTMLParser
from bs4 import BeautifulSoup
from markdown import markdown
import urllib2
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

#########################
# LOAD DATA
########################

print 'READ IN DATA'
# set the data path
data_directory = '../data/'

sample_data = False

if sample_data:
    posts_path = os.path.join(data_directory,'sample_data.csv')
else:
    posts_path = os.path.join(data_directory,'all_posts0.csv')
df_posts = pd.read_csv(posts_path)
# Remove blank articles
df_posts =  df_posts[df_posts['body'].notnull()]

#########################
# Combined data
########################
print 'CREATING A NEW CLEAN DATASET'

# Combine multiple updates to articles to get one body per post
combined_body = (df_posts.groupby(['author','permlink'])
                 .agg(lambda x: ''.join(set(x)))
                 .reset_index()
                 )
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

#Encode to unicode so that Beautiful Soup can work properly
df_posts['body'] = df_posts['body'].str.decode('utf-8')

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

# Removes html,markdown, percent encoding
def remove_html_markdown(x):
    # Markdown code is extremely slow and cannot be used on hte large dataset
#     html = markdown(x)
    raw = BeautifulSoup(x,'lxml').get_text()
#     text = ' '.join(BeautifulSoup(html).findAll(text=True))
    output = urllib2.unquote(raw)
    return output

df_posts['body'] = df_posts.apply(lambda x: remove_html_markdown(x['body']), axis=1)

# # # Remove all new lines
expression = r'\n'
df_posts['body'] = df_posts['body'].str.replace(expression,' ')

# Remove unicode junk
df_posts['body'] = (df_posts['body']
                    .str.encode('ascii', 'ignore'))

df_posts['body tags'] = df_posts['body']

# Replace in order to properly identify mentions
df_posts['body'] = df_posts['body'].str.replace(r'@@','@')

expression = '@(\S+)'
# Extract all Hash Tages
df_posts['body mentions'] = df_posts['body'].str.findall(expression)
df_posts['number of body mentions'] = df_posts['body mentions'].apply(len)

############################
# CLEAN DATA
############################
print 'CLEANING DATA'
# START CLEANING TEXT --
# 1. html removal
# 2. words, and numbers
# 3. symbols ',!
# 4. decode, encode?

# Remove Links
expression = r'http\S+'
df_posts['body'] = df_posts['body'].str.replace(expression,' ')

# Remove all Hash Tags
expression = '#(\S+)'
df_posts['body'] = df_posts['body'].str.replace(expression,' ')

# Remove all Tags
expression = '@(\S+)'
df_posts['body'] = df_posts['body'].str.replace(expression,' ')

def removesymbols(x):
    x = re.sub("[\W\d]+"," ", x.strip())
    x = str(x)
    return x

# Keep only letters and numbers
# df_posts['body'] = df_posts['body'].apply(lambda x: removesymbols(x), axis=1)
df_posts['body'] = df_posts.apply(lambda x: removesymbols(x['body']), axis=1)


# Remove Any Capital Letter by themselves A, B, C, D etc
# These are usually removed anyways during stemming but seems to be some
# residuals left over
expression = r'\b[A-Z]\b|\b[a-z]\b'
df_posts['body'] = df_posts['body'].str.replace(expression,' ')

posts_raw_cleaned = os.path.join(data_directory,
                                             'posts_raw_cleaned0',
                                             'posts_raw_cleaned0.csv')
print 'WRITING TO FILE'
df_posts.to_csv(posts_raw_cleaned,
                              index=False,
                              quoting=csv.QUOTE_ALL,
                              encoding='utf-8')