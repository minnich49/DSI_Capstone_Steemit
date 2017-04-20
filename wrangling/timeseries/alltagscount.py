import pandas as pd
import numpy as np
import os

import csv
import json
import sys

from collections import Counter

import matplotlib.pyplot as plt

print 'READ IN DATA'
# set the data path
data_directory = '../data/'

data_directory = '../data/'
data_directory = os.path.join('..' ,'data')

authors = pd.read_csv(os.path.join(data_directory,'accounts.csv'))

print os.path.join(data_directory,'accounts.csv')
authors = pd.read_csv(os.path.join(data_directory,'accounts.csv'))
posts_raw_cleaned = pd.read_csv(os.path.join(data_directory,
                                             'posts_raw_cleaned',
                                             'posts_raw_cleaned.csv'))


df_timeseries = posts_raw_cleaned

# create counts for all tags
def createtag_count(x):
    count = 0
    if type(x) == str:
        try:
            j = json.loads(x)
            if 'tags' not in j:
                return 0
            tags = j['tags']
            count = len(tags)
        except Exception as e: 
            print "exception:", e
    return count

# create a column that has a list for each one
def tags(x):
    tags = 'EMPTY'
    if type(x) == str:
        try:
            j = json.loads(x)
            if 'tags' not in j:
                return 'EMPTY'
            tags = j['tags']
        except Exception as e: 
            print "exception:", e
    return tags 


df_timeseries['tags_count'] = posts_raw_cleaned.apply(lambda x: createtag_count(x['json_metadata']), axis=1)
df_timeseries['tags'] = posts_raw_cleaned.apply(lambda x: tags(x['json_metadata']), axis=1)

# change the date columns to be a timeseries object
df_timeseries['created'] = pd.to_datetime(df_timeseries['created'], format='%Y-%m-%d')


# return only the date, chopping off the time
df_timeseries['date_final'] = posts_raw_cleaned.apply(lambda row: row['created'].date(), axis=1)
# creating month year columns
df_timeseries['yearmonth_final'] = df_timeseries['date_final'].map(lambda x: str(x.year) + '-'+ str(x.month))

path = '/Users/laurenmccarthy/Documents/Columbia/Capstone/DSI_Capstone_Steemit/data/timeseries/'

with open(path+'trending.json', 'r') as fp:
    trending = json.load(fp)


from datetime import timedelta
from datetime import datetime
# outside loop for all the tags

all_tags_count = {}
for key in trending.keys():
    print key
    date2idx = {}
    idx = 0
    t=df_timeseries['date_final'].min()
    while t < df_timeseries['date_final'].max():
        s = datetime.strftime(t, "%Y-%m-%d")
        date2idx[s] = idx
        t += timedelta(days=1)
        idx += 1
    s = datetime.strftime(t, "%Y-%m-%d")
    date2idx[s] = idx

    current_tags_count = np.zeros(idx+1)
    #print current_tags_count.shape
    for s, count in trending[key].iteritems():
        #print s, count
        idx = date2idx[s]
        #print idx
        current_tags_count[idx] = count
    all_tags_count[key] = current_tags_count

path = '/Users/laurenmccarthy/Documents/Columbia/Capstone/DSI_Capstone_Steemit/data/timeseries/'

with open(path+'all_tags_count.json', 'w') as fp:
    json.dump(all_tags_count, fp)

