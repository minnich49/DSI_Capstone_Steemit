import pymssql
import pandas as pd
import numpy as np
import os
import re
import joblib
import csv


print 'READ IN DATA'
# set the data path
data_directory = '../data/'

new_feature_data_path = '' # add where we will save this


# change change this to load different types of data
data,feature_names,data_desc = load_data_and_description(data_type='posts_tfidf')

# define which values you want to use
featurestoadd = ['author_reputation','number of body urls', u'number of youtube urls','number of image urls', 'number of body tags', 'number of body mentions']

################################################
# CREATE SECTION TO CLEAN THE DIFFERENT COLUMNS
################################################

# this can be somewhat static since the original features will not change
new_features = data_desc[featurestoadd].as_matrix()

new_data = np.column_stack((data, new_features))

new_data.to_csv(new_feature_data_path,
                              index=False, 
                              quoting=csv.QUOTE_ALL, 
                              encoding='utf-8')