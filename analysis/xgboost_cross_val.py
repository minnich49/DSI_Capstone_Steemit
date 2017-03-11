
import xgboost as xgb
import numpy as np
import pandas as pd
import pymssql
import pandas as pd
import numpy as np
import os
import re
import joblib
import seaborn as sns
import matplotlib.pylab as plt
import scipy.sparse as ssp

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from DSI_Capstone_Steemit.utils.utils import(
    load_data_and_description,
)
from sklearn.model_selection import train_test_split

data,feature_names,data_desc = load_data_and_description(data_type='posts_tfidf')
data_desc['log total_payout_value'] = np.log(data_desc['total_payout_value'])



# Remove middle value articles

idx1 = data_desc['log total_payout_value'] < 1.2
idx2 = data_desc['log total_payout_value'] >2.5

idx_not = (~idx1) & (~idx2)

data_desc = data_desc[~idx_not]
data = data[~idx_not.values,:]
y = data_desc['log total_payout_value'] >2.5

value_counts = data_desc['category'].value_counts()
top_categories = value_counts.index[value_counts > np.percentile(data_desc['category'].value_counts(),97)]
idx = data_desc['category'].isin(top_categories)
data_desc['top category'] = idx.astype(int)

data_desc['top category listed'] = data_desc.ix[data_desc['top category'].values.astype(bool) ,'category']

data_desc['top category listed'] = data_desc['top category listed'].fillna('Other')

train_features = data_desc.ix[:,['number of body tags',
                                   'number of body urls',
                                   'number of image urls',
                                   'number of body mentions',
                                   'number of image urls',
                                   'number of youtube urls',
                                   'language',
                                   'author_reputation_scaled',
                                   'number of steem counts',
                                'top category']]


train = pd.get_dummies(train_features)

num_image_urls = train['number of image urls'].values[:,0]
train.drop('number of image urls',axis = 1, inplace=True)

train['number of image urls'] = num_image_urls

training_names = train.columns

train_sparse = ssp.csr_matrix(train)
new_data = ssp.hstack([data,train_sparse])
train = new_data.tocsr()

# All samples
number_of_samples = train.shape[0]

X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.33, random_state=42)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


cv_params = {'max_depth': [1,3,5,7],
             'min_child_weight': [1,3,7],
             'colsample_bytree': [0.5,0.8,1],
             'reg_alpha':[0.01,0.1,0.2],
             'reg_lambda':[0.01,0.1,0.2]}

# cv_params = {'max_depth': [1,2]}



ind_params = {'learning_rate': 0.05, 'n_estimators': 100, 'seed':0,
             'objective': 'binary:logistic'}
model = xgb.XGBClassifier(**ind_params)
optimized_GBM = GridSearchCV(model,
                            cv_params,
                             scoring ='accuracy',
                             cv = 5,
                             n_jobs = -1,
                             verbose = False)

optimized_GBM.fit(X_train, y_train.values)

results = pd.DataFrame(optimized_GBM.cv_results_)
results.to_csv('../data/' + 'results.csv')