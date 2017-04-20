from datetime import timedelta
from datetime import datetime
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

# change the date columns to be a timeseries object
df_timeseries['created'] = pd.to_datetime(df_timeseries['created'], format='%Y-%m-%d')
# return only the date, chopping off the time
df_timeseries['date_final'] = posts_raw_cleaned.apply(lambda row: row['created'].date(), axis=1)
# creating month year columns
df_timeseries['yearmonth_final'] = df_timeseries['date_final'].map(lambda x: str(x.year) + '-'+ str(x.month))


path = '/Users/laurenmccarthy/Documents/Columbia/Capstone/DSI_Capstone_Steemit/data/timeseries/'


with open(path+'trendingtags.json', 'r') as fp:
    trending = json.load(fp)

with open(path+'all_tags_count.json', 'r') as fp:
    all_tags_count = json.load(fp)

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


# plt.plot(all_tags_count['brexit'])
# dates = df_timeseries['date_final']
# plt.xticks(range(len(dates)), dates, size='small')

# d = "2016-06-30"
# i = date2idx[d]


# trending_score = []
# for row in df_timeseries.interitems():
#     d = row['date_final']
#     i = date2idx[d]
#     tag = row['category']
#     val = all_tags_count[tag][i-3:i].mean()
#     trending_score.append(val)    

trending_score = []
for index, row in df_timeseries.iterrows():
    d = str(row['date_final'])
    print d
    i = date2idx[d]
    tag = row['category']
    print tag
    try:
        val = all_tags_count[tag][i-3:i].mean()
        print val
    except:
        val = 0
    trending_score.append(val) 

