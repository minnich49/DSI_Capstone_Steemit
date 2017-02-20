# Check for directory and if not there, create one
import os
import sys
def ensure_directory(directory):
    directory = '../data/' + directory
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)

# Check for each directory in the directory list
def ensure_directories(dir_list):
    for directory in dir_list:
        ensure_directory(directory)

