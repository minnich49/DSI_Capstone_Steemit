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

