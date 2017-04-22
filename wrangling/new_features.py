import pandas as pd
import numpy as np
import os
import csv
import json
import sys
import joblib

import csv
import json
import sys
print 'READ IN DATA'
# set the data path
data_directory = '../data/'
data_directory = os.path.join('..' ,'data')

authors = pd.read_csv(os.path.join(data_directory,'accounts.csv'))

print os.path.join(data_directory,'accounts.csv')
authors = pd.read_csv(os.path.join(data_directory,'accounts.csv'))
posts_raw_cleaned = pd.read_csv(os.path.join(data_directory,
                                             'posts_raw_cleaned',
                                             'posts_raw_cleaned.csv'))
# Output File
output_file = os.path.join(data_directory,
                           'posts_cleaned_features',
                           'posts_cleaned_features.csv')

# Add number of times steem is found within the body
steem_counts = posts_raw_cleaned['body'].str.lower().str.count('steem')
posts_raw_cleaned['number of steem counts'] = steem_counts

# A flag authors that are in the top 5%
percent = np.percentile(authors['reputation'],95)
top_95_idx = (authors['reputation'] > percent)
whale_list = authors.ix[top_95_idx,'name'].values
posts_raw_cleaned['whale'] = posts_raw_cleaned['author'].isin(whale_list).astype(int)


# Flag where whale is the author
posts_raw_cleaned['whale'] = posts_raw_cleaned['author'].isin(whale_list).astype(int)

# Flag where whales are mentioned
whale_mention_counts = np.zeros(posts_raw_cleaned['body mentions'].shape[0])
whale_mention_counts = np.zeros(posts_raw_cleaned['body mentions'].shape[0])
for whale in whale_list:
    whale_mention_counts += posts_raw_cleaned['body mentions'].str.count(whale).values
posts_raw_cleaned['body whale mentions'] =whale_mention_counts

# Add Language
languages = []
for language in posts_raw_cleaned['body_language']:
    if (language != '[]') & pd.notnull(language):
        languages.append(json.loads(language)[0]['language'])
    else:
        languages.append('unknown')

posts_raw_cleaned['language'] = languages

# Scale Author Reputation to put it a more reasonable scale
posts_raw_cleaned['author_reputation_scaled'] = (posts_raw_cleaned[
                                                     'author_reputation'] + 0.0) / (
                                                10 ** 14)

# Add Centrality Measures
print('Loading Centrality')
input_directory = 'networkx_votes'

def load_joblib(filename):
    return joblib.load(os.path.join(input_directory,filename))


hubs, authorities = load_joblib('hits')
cluster = load_joblib('parts')
pagerank = load_joblib('prank')
eig_cent = load_joblib('eig_cent')
core_k = load_joblib('core_k')

posts_raw_cleaned['Cluster'] = posts_raw_cleaned['author'].map(cluster)
posts_raw_cleaned.loc[:, 'Cluster'] = posts_raw_cleaned['Cluster']
posts_raw_cleaned.loc[~posts_raw_cleaned['Cluster'].isin([1, 3, 0, 2, 5, 4]), 'Cluster Condense'] = 'Other'
posts_raw_cleaned['Hubs'] = posts_raw_cleaned['author'].map(hubs) * 10000
posts_raw_cleaned['Authorities'] = posts_raw_cleaned['author'].map(authorities) * 10000
posts_raw_cleaned['Page Rank'] = posts_raw_cleaned['author'].map(pagerank) * 10000
posts_raw_cleaned['Eigen Centrality'] = posts_raw_cleaned['author'].map(eig_cent) * 10000
posts_raw_cleaned['Core K'] = posts_raw_cleaned['author'].map(core_k) * 10000

# Add time series feature

timeseries = pd.read_csv('timeseries/timeseries.csv', header=None)
posts_raw_cleaned['trending'] = timeseries


########################################
print 'Calculating Gradients'
# Add Gradients
category = posts_raw_cleaned.ix[:, ['created', 'category']]
category = category.set_index('created')
category = pd.get_dummies(category, prefix='', prefix_sep='')
category.index = pd.DatetimeIndex(category.index)

# Number of days to group by
number_of_days = 3
category = category.resample(str(number_of_days) + 'D').sum()
x_array = range(category.shape[0] * number_of_days)
x_array = x_array[0::number_of_days]

for cat in category.columns:
    y = category[cat].values
    slopes = np.array([0] + list(np.diff(y[::1]) / np.diff(x_array[::1])))
    category[cat] = slopes

# Get current row and category and date

for row in posts_raw_cleaned.index:

    row_category = posts_raw_cleaned.ix[row, 'category']
    if pd.notnull(row_category):
        row_date = posts_raw_cleaned.ix[row, 'created']
        # From post data, find closest date
        try:
            closest_date = category.index.get_loc(pd.to_datetime(row_date),
                                                  method='ffill')
            closest_gradient = category.ix[closest_date, row_category]
            posts_raw_cleaned.loc[row, 'Gradient'] = closest_gradient
        except:
            posts_raw_cleaned.loc[row, 'Gradient'] = 0
            # print 'Did not work for ', row_date, row_category

posts_raw_cleaned['Gradient'] = posts_raw_cleaned['Gradient'].fillna(0)
########################################

posts_raw_cleaned.to_csv(output_file,
                              index=False, 
                              quoting=csv.QUOTE_ALL, 
                              encoding='utf-8')