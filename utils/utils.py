# Check for directory and if not there, create one
import os
import sys
import pandas as pd
import joblib

data_directory = '../data/'

def ensure_directory(directory):
    directory = '../data/' + directory
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check for each directory in the directory list
def ensure_directories(dir_list):
    for directory in dir_list:
        ensure_directory(directory)

def check_for_dir_python_path():
    module_directory = os.path.join(
        os.getcwd().split('DSI_Capstone_Steemit')[0])
    if module_directory not in sys.path:
        sys.path.insert(1, module_directory)

def load_raw():
    posts_path = os.path.join(data_directory, 'posts_raw_cleaned',
                              'posts_raw_cleaned.csv')
    df_posts = pd.read_csv(posts_path)
    return df_posts
def load_data_and_description(data_type = 'tfidf'):
    # check_for_dir_python_path()

    if data_type == 'tfidf':
        directory = 'posts_tfidf'
        file_name = 'posts_tfidf.pkl'
        feature_names = 'posts_tfidf_feature_names'
        desc_file = 'posts_tfidf_desc.csv'

    elif data_type == 'word2vec':
        file_name = 'word2vec_doc_matrix_avg'
        feature_names = 'word2vec_doc_matrix_avg_feature_names'
        desc_file = 'word2vec_doc_matrix_avg_desc.csv'
    else:
        directory = 'posts_counts'
        file_name = 'posts_counts.pkl'
        feature_names = 'posts_counts_feature_names'
        desc_file = 'posts_counts_desc.csv'

    data_path = os.path.join(data_directory,directory,file_name)
    feature_path = os.path.join(data_directory,directory,feature_names)
    desc_path = os.path.join(data_directory, directory,desc_file)

    data = joblib.load(data_path)
    data_desc = pd.read_csv(desc_path)

    # Do not have feature names for word2vec matrices
    if data_type == 'word2vec':
        feature_names = None
    else:
        feature_names = joblib.load(feature_path)

    return data,feature_names,data_desc



