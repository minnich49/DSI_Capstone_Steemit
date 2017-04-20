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

path = '/Users/laurenmccarthy/Documents/Columbia/Capstone/DSI_Capstone_Steemit/data/timeseries/'

with open(path+'datecount.json', 'r') as fp:
    datecount = json.load(fp)

with open(path+'tagscountdict.json', 'r') as fp:
    tagscountdict = json.load(fp)

top_25_tags = dict(Counter(tagscountdict).most_common(25))
print 'top_25_tags', top_25_tags
top_25_tags

dates = datecount.keys() 

for date in dates:
	print date
	date_list = datecount[date]
	top_five = sorted(date_list, key=date_list.get, reverse=True)[:5]
	print top_five

	# print max(datecount[date].values())
	# print max(datecount[date].keys())



for date in dates:
	print date
	date_list = datecount[date]
	top_five_dict = dict(Counter(date_list).most_common(1))
	print top_five_dict
	ax = plt.subplot(111)
	ax.bar(range(len(top_five_dict)), top_five_dict.values(),width=0.2,color='b')
	plt.xticks(range(len(top_five_dict)), top_five_dict.keys(), rotation='vertical')
	plt.tight_layout()
	plt.show()
	# ax.bar(dates, top_five_dict,width=0.2,color='g',align='center')
	# ax.bar(dates+0.2, top_five_dict,width=0.2,color='r',align='center')
	# ax.xaxis_date()

# plt.show()

plt.bar(range(len(top_25_tags)), top_25_tags.values())
plt.xticks(range(len(top_25_tags)), top_25_tags.keys(), rotation='vertical')
plt.tight_layout()
plt.show()





# top 25 posts waht tags did they use:
