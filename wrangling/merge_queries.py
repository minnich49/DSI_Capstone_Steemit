import joblib
import pandas as pd
import os
import joblib
import csv
all_files = []
dir = '../data/large_data_store/'
dir0 = '../data/large_data_store0/'

output_filename = ['../data/all_posts.csv','../data/all_posts0.csv']
dir_list = [dir,dir0]
for j,directory in enumerate(dir_list):
    for i, filename in enumerate(os.listdir(directory)):
        if 'join' in filename:
            print 'File Number:', i
            all_files.extend(joblib.load(filename=os.path.join(directory, filename)))
    column_names = joblib.load(os.path.join(directory,'column_names'))

    print 'Saving file to csv'
    all_posts = pd.DataFrame(all_files)
    all_posts.columns = column_names
    all_posts.to_csv(output_filename[j],
                     encoding='utf-8',
                     quoting=csv.QUOTE_ALL,
                     index = False)