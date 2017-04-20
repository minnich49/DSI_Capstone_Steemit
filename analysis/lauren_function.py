import os
import sys
import pandas as pd
import joblib

data_directory = '../data/'

def load_data_and_description(data_type = 'tfidf'):
    # check_for_dir_python_path()

    if data_type == 'tfidf':
        directory = 'posts_tfidf'
        file_name = 'posts_tfidf.pkl'
        feature_names = 'posts_tfidf_feature_names'

    elif data_type == 'word2vec':
        file_name = 'word2vec_doc_matrix_avg'
        feature_names = 'word2vec_doc_matrix_avg_feature_names'

    elif data_type == 'tags_tfidf':
        directory = 'tags_tfidf'
        file_name = 'tags_tfidf.pkl'
        feature_names = 'tags_tfidf_feature_names'

    else:
        directory = 'posts_counts'
        file_name = 'posts_counts.pkl'
        feature_names = 'posts_counts_feature_names'


    desc_file = os.path.join(data_directory,
                             'posts_cleaned_features',
                             'posts_cleaned_features.csv')


    data_path = os.path.join(data_directory,directory,file_name)
    feature_path = os.path.join(data_directory,directory,feature_names)

    data = joblib.load(data_path)
    data_desc = pd.read_csv(desc_file)

    # Do not have feature names for word2vec matrices
    if data_type == 'word2vec':
        feature_names = None
    else:
        feature_names = joblib.load(feature_path)

    return data,feature_names,data_desc