import pandas as pd
import numpy as np
import scipy.sparse as ssp

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

xgtrain = xgb.DMatrix(X_train.toarray(), label=y_train)

def xgb_evaluate(min_child_weight,
                 colsample_bytree,
                 max_depth,
                 subsample,
                 gamma,
                 alpha):

    params['min_child_weight'] = int(min_child_weight)
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(max_depth)
    params['subsample'] = max(min(subsample, 1), 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)


    cv_result = xgb.cv(params, xgtrain, num_boost_round=num_rounds, nfold=5,
             seed=random_state,
             callbacks=[xgb.callback.early_stop(50)])

    return -cv_result['test-mae-mean'].values[-1]

num_rounds = 10
random_state = 2016
num_iter = 5
init_points = 5
params = {
    'eta': 0.05,
    'silent': 1,
    'eval_metric': 'mae',
    'verbose_eval': True,
    'seed': random_state
}

import xgboost as xgb
from bayes_opt import BayesianOptimization

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (1, 20),
                                            'colsample_bytree': (0.1, 1),
                                            'max_depth': (1, 15),
                                            'subsample': (0.5, 1),
                                            'gamma': (0, 10),
                                            'alpha': (0, 10),
                                            })
xgbBO.maximize(init_points=init_points, n_iter=num_iter)

print(xgbBO.res['max'])
print(xgbBO.res['all'])

xgbBO.points_to_csv('bayesian_results.csv')
import json
with open('bayesian_results.json', 'w') as fp:
    json.dump(xgbBO.res, fp)
