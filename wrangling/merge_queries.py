import joblib
import pandas as pd
import os
import joblib
import csv
all_files = []
dir = '../data/large_data_store/'
for i, filename in enumerate(os.listdir(dir)):
    if 'join' in filename:
        all_files.extend(joblib.load(filename=os.path.join(dir, filename)))
column_names = joblib.load(os.path.join(dir,'column_names'))

all_posts = pd.DataFrame(all_files)
all_posts.columns = column_names
all_posts.to_csv('../data/all_posts.csv',
                 encoding='utf-8',
                 quoting=csv.QUOTE_ALL,
                 index = False)