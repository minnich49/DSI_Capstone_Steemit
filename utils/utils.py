# Check for directory and if not there, create one
import os
import sys
import pandas as pd
import joblib

def ensure_directory(directory):
    directory = '../data/' + directory
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check for each directory in the directory list
def ensure_directories(dir_list):
    for directory in dir_list:
        ensure_directory(directory)


data_directory = '../data/'


def load_data_and_description(data_type = 'tfidf'):
    if data_type == 'tfidf':
        file_name = 'posts_tfidf'
        feature_names = 'posts_tfidf_feature_names'
        desc_file = 'posts_tfidf_desc.csv'

    elif data_type == 'word2vec':
        file_name = 'word2vec_doc_matrix_avg'
        feature_names = 'word2vec_doc_matrix_avg_feature_names'
        desc_file = 'word2vec_doc_matrix_avg_desc.csv'
    else:
        file_name = 'posts_counts'
        feature_names = 'posts_counts_feature_names'
        desc_file = 'posts_counts_desc.csv'

    data_path = os.path.join(data_directory,file_name,file_name)
    feature_path = os.path.join(data_directory,file_name,feature_names)
    desc_path = os.path.join(data_directory, file_name,desc_file)

    data = joblib.load(data_path)
    data_desc = pd.read_csv(desc_path)

    # Do not have feature names for word2vec matrices
    if data_type == 'word2vec':
        feature_names = None
    else:
        feature_names = joblib.load(feature_path)

    return data,feature_names,data_desc


