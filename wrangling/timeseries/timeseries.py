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

# posts_raw_cleaned.apply(lambda row: row['created'].date(), axis=1)

# df['YearMonth'] = df['ArrivalDate'].map(lambda x: 1000*x.year + x.month)
# df_timeseries['date_final'] = posts_raw_cleaned.apply(lambda row: row['created'].date(), axis=1)
 

# create a sorted timeseries column called X
X = df_timeseries.sort('date_final')

# X['date_final'] = X.apply(lambda row: str(row['date_final']), axis=1)

# make the time series object a time stamp
# X['ts'] = X.apply(lambda row: time.mktime(row['date_final'].timetuple()), axis=1)
# X['ts'] = X.apply(lambda row: time.mktime(row['date_final'].timetuple()), axis=1)

plt.plot(X['date_final'], X['tags_count']) # plot all points as a scatter plot
plt.gcf().autofmt_xdate()
plt.show()

#Look into making this work with dt
#X['dt'] = X.apply(lambda row: datetime.combine(row['date_final'], datetime.min.time()), axis=1)


Y = X.groupby(['date_final'])['tags_count'].mean()
std = X.groupby(['date_final'])['tags_count'].std()
plt.plot_date(Y.index, Y, '-')
plt.fill_between(Y.index, (Y - std).tolist(), (Y + std).tolist(), alpha=0.5, color='b')
plt.gcf().autofmt_xdate()
plt.show()


#average number of tags for past x days

#populatrity of tags over the past few days
# 100 times in past 3 days
# per post...average of the sum of how many times that tag was used in the past x days


# { date: { tag1: count, tag2: count, ... }}

print 'datecount', 'tagscountdict'
datecount = {}
tagscountdict = {}
for index, row in X.iterrows():
    date = row['yearmonth_final']
    tag = row['tags']
    if tag == 'EMPTY':
        pass
    else:
        if date not in datecount:
            print 'adding date', date
            datecount[date] = {}
        for topic in tag:
            if topic not in datecount[date]:
                datecount[date][topic] = 1
                tagscountdict[topic] = 1
            else:
                datecount[date][topic] += 1
                tagscountdict[topic] += 1

print 'tending'
trending = {}
for index, row in X.iterrows():
    tag = row['tags']
    dates = row['date_final']
    if tag == 'EMPTY':
        pass
    else:
        if tag not in trending:
            trending[tag] = {}
        for date in dates:
            if date not in trending[tag]:
                trending[tag][date] = 1
            else:
                trending[tag][date] += 1



path = '/Users/laurenmccarthy/Documents/Columbia/Capstone/DSI_Capstone_Steemit/data/timeseries/'

with open(path+'datecount.json', 'w') as fp:
    json.dump(datecount, fp)

with open(path+'tagscountdict.json', 'w') as fp:
    json.dump(tagscountdict, fp)

with open(path+'trendingtags.json', 'w') as fp:
    json.dump(trendings, fp)




