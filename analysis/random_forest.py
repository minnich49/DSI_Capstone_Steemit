import pandas as pd
import numpy as np
import scipy.sparse as ssp
from lauren_function import *
from sklearn.model_selection import train_test_split
from bayes_opt import BayesianOptimization
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product
from bayes_opt import BayesianOptimization

data_directory = '../data/'

data,feature_names,data_desc = load_data_and_description(data_type='tags_tfidf')
# print 'median', np.median(np.log(data_desc['total_payout_value']))

data_desc['log total_payout_value'] = np.log(data_desc['total_payout_value'])
idx1 = data_desc['log total_payout_value'] < 1.2
idx2 = data_desc['log total_payout_value'] >2.5

idx_not = (~idx1) & (~idx2)

data_desc = data_desc[~idx_not]
data = data[~idx_not.values,:]

y = data_desc['log total_payout_value'] >2.5


# model = RFC()


# creating a label for if it is in the top category
value_counts = data_desc['category'].value_counts()
top_categories = value_counts.index[value_counts > np.percentile(data_desc['category'].value_counts(),97)]
idx = data_desc['category'].isin(top_categories)
data_desc['top category'] = idx.astype(int)
data_desc['top category listed'] = data_desc.ix[data_desc['top category'].values.astype(bool) ,'category']
data_desc['top category listed'] = data_desc['top category listed'].fillna('Other')

# some of the values were null -- filled them in with 0
data_desc['number of steem counts'] = data_desc['number of steem counts'].fillna(0)


new_featurenames = ['number of body tags',
  'number of body urls',
  'number of image urls',
  'number of body mentions',
  'number of image urls',
  'number of youtube urls',
  'language',
  'author_reputation_scaled',
  'number of steem counts',
  'top category listed']
train_features = data_desc[new_featurenames]

# #['number of body tags',
#                                    'number of body urls',
#                                    'number of image urls',
#                                    'number of body mentions',
#                                    'number of image urls',
#                                    'number of youtube urls',
#                                    'language',
#                                    'author_reputation_scaled',
#                                    'number of steem counts']]


train = pd.get_dummies(train_features)

num_image_urls = train['number of image urls'].values[:,0]
train.drop('number of image urls',axis = 1, inplace=True)

train['number of image urls'] = num_image_urls

training_names = train.columns

train_sparse = ssp.csr_matrix(train)
new_data = ssp.hstack([data,train_sparse])
# print 'column names', new_data.columns
train = new_data.tocsr()

# All samples
number_of_samples = train.shape[0]

X_train, X_test, y_train, y_test = train_test_split(
    train, y, test_size=0.10, random_state=42)

#########################################################
############ SET UP BAYESIAN OPTIMIZATION ###############
#########################################################
# print 'starting bayesian optimization'

# gp_params = {"alpha": 1e-5}

# def rfccv(n_estimators, min_samples_split, max_features):
#     val = cross_val_score(
#         RFC(n_estimators=int(n_estimators),
#             min_samples_split=int(min_samples_split),
#             max_features=min(max_features, 0.999),
#             random_state=2
#         ),
#         X_train, y_train, 'f1', cv=2
#     ).mean()
#     return val

# rfcBO = BayesianOptimization(
#         rfccv,
#         {'n_estimators': (10, 250),
#         'min_samples_split': (2, 25),
#         'max_features': (0.1, 0.999)}
#     )

# rfcBO.explore({'n_estimators': [10, 50], 'min_samples_split': [2, 5], 'max_features': [.1, .5]})

# print 'optimizing'
# rfcBO.maximize(n_iter=10)

# params = rfcBO.res['max']['max_val']

# params['n_estimators'] = int(params['n_estimators'])
# params['min_samples_split'] = int(params['min_samples_split'])
# params['max_features'] = float(params['max_features'])

# model = RFC(**params)
model = RFC(n_estimators=134, min_samples_split=22, max_features=.1)
print 'model with optimized parameters', model


print 'Fitting model'
model.fit(X_train, y_train)


print 'score train:', model.score(X_train, y_train)
print 'score test:', model.score(X_test, y_test)

columnnames = feature_names + new_featurenames


sorted_fi, sorted_cn = zip(
  *sorted(
    zip(model.feature_importances_, columnnames),
    reverse = True
  )
)


print 'creating top features'
top_features = sorted_fi[:25]
top_names = sorted_cn[:25]

print 'top_features', top_features
print 'top_names', top_names

indicies = list(range(25))
print 'indicies', indicies


# Print the feature ranking
print("Feature ranking:")

for f in range(25):
    print("%d. feature %d (%f)" % (f + 1, indicies[f], top_features[f]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(25), top_features,
       color="r", align="center")
plt.xticks(range(25), top_names, rotation=90)
plt.xlim([-1, 25])
plt.tight_layout()
plt.show()




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, model.predict(X_test))
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,0],
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=[1,0], normalize=True,
                      title='Normalized confusion matrix')
plt.tight_layout()
plt.show()

