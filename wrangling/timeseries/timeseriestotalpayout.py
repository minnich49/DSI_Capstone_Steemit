import pandas as pd
import numpy as np
import os

import csv
import json
import sys
import time
import json

import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter

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

# Y = df_timeseries.sort('total_payout_value')
Y = df_timeseries
# top25payout = Y.head(25)

datetotalpayout = {}
tagstotalpayout = {}
for index, row in Y.iterrows():
    date = row['yearmonth_final']
    tag = row['tags']
    if tag == 'EMPTY':
        pass
    else:
        if date not in datetotalpayout:
            print 'adding date', date
            datetotalpayout[date] = {}
        for topic in tag:
            # print 'topic', topic
            if topic not in datetotalpayout[date]:
                datetotalpayout[date][topic] = row['total_payout_value']
                tagstotalpayout[topic] = row['total_payout_value']
            else:
                datetotalpayout[date][topic] += row['total_payout_value']
                tagstotalpayout[topic] += row['total_payout_value']

print 'tagstotalpayout', tagstotalpayout
print 'datetotalpayout', datetotalpayout
# plt.plot()


# plt.bar(range(len(top25payout)), top25payout['total_payout_value']
# plt.xticks(range(len(top25payout)), top25payout['tags'], rotation='vertical')
# plt.tight_layout()
# plt.show()

top_25_payout = dict(Counter(tagstotalpayout).most_common(25))
print top_25_payout

for date in dates:
    print date
    date_list_totalpayout = datetotalpayout[date]
    top_five_dict = dict(Counter(date_list_totalpayout).most_common(5))
    print top_five_dict
    ax = plt.subplot(111)
    ax.bar(range(len(top_five_dict)), top_five_dict.values(),width=0.2,color='b')
    plt.xticks(range(len(top_five_dict)), top_five_dict.keys(), rotation='vertical')
    plt.tight_layout()
    plt.show()

plt.bar(range(len(top_25_payout)), top_25_payout.values())
plt.xticks(range(len(top_25_payout)), top_25_payout.keys(), rotation='vertical')
plt.tight_layout()
plt.show()