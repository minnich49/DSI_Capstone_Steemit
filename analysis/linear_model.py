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
import matplotlib.pyplot as plt

data_directory = '../data/'

data,feature_names,data_desc = load_data_and_description(data_type='tags_tfidf')

# Keep the targets as continuous
# y = data_desc['total_payout_value']
y = np.log(data_desc['total_payout_value'])

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

# train_features = data_desc[['number of body tags',
                                   # 'number of body urls',
                                   # 'number of image urls',
                                   # 'number of body mentions',
                                   # 'number of image urls',
                                   # 'number of youtube urls',
                                   # 'language',
                                   # 'author_reputation_scaled',
                                   # 'number of steem counts']]


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
    train, y, test_size=0.10, random_state=42)


model = linear_model.LinearRegression(fit_intercept=True, normalize=True)


print 'Fitting model'
model.fit(X_train, y_train)

# columnnames = feature_names + new_featurenames


# The coefficients
print('Coefficients: \n', model.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % np.mean((model.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % model.score(X_test, y_test))

print('R2_score: %.2f' % r2_score(y_test, model.predict(X_test)))


##############################################
# CREATING FEATURE PLOTS

columnnames = feature_names + new_featurenames

sorted_fi, sorted_cn = zip(
  *sorted(
    zip(model.coef_, columnnames),
    reverse = True
  )
)

sorted_fi2, sorted_cn2 = zip(
  *sorted(
    zip(model.coef_, columnnames),
    reverse = False
  )
)

print 'creating top features'
top_features = sorted_fi[:12]
top_names = sorted_cn[:12]

print 'top_features', top_features
print 'top_names', top_names

indicies = list(range(24))
print 'indicies', indicies

print 'creating bottom features'
bottom_features = sorted_fi2[:12]
bottom_names = sorted_cn2[:12]

print 'top_features', bottom_features
print 'top_names', bottom_names

all_features = top_features + bottom_features
all_names = top_names + bottom_names



# indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking")

for f in range(24):
    print("%d. feature %d (%f)" % (f + 1, indicies[f], all_features[f]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(24), all_features,
       color="r", align="center")
plt.xticks(range(24), all_names, rotation=90)
# plt.xticks(rotation=90)
plt.xlim([-1, 24])
# plt.xlabel(top_names, rotation='vertical')
plt.tight_layout()
plt.show()

# print 'creating bottom features'
# bottom_features = sorted_fi2[:25]
# bottom_names = sorted_cn2[:25]

# print 'top_features', bottom_features
# print 'top_names', bottom_names

# indicies = list(range(25))
# print 'indicies', indicies

# indices = np.argsort(importances)[::-1]

# # Print the feature ranking
# print("Feature ranking")

# for f in range(25):
#     print("%d. feature %d (%f)" % (f + 1, indicies[f], bottom_features[f]))

# # Plot the feature importances of the forest
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(25), bottom_features,
#        color="r", align="center")
# plt.xticks(range(25), bottom_names, rotation=90)
# # plt.xticks(rotation=90)
# plt.xlim([-1, 25])
# # plt.xlabel(top_names, rotation='vertical')
# plt.show()





# # print 'zipped_feature_importances_columns', zipped_feature_importances_columns


# i = 0
# for feature_importance_top, column_name_top in zip(sorted_fi_top, sorted_cn_top): #zipped_feature_importances_columns:
#   print 'feature_importance_top, column_name_top', feature_importance_top, column_name_top
#   i += 1
#   if i > 25:
#     break

# i = 0
# for feature_importance_bottom, column_name_bottom in zip(sorted_fi_top, sorted_cn_top): #zipped_feature_importances_columns:
#   print 'feature_importance_bottom, column_name_bottom', feature_importance_bottom, column_name_bottom
#   i += 1
#   if i > 25:
#     break

# plt.bar(column_name, feature_importance)
# plt.show()



