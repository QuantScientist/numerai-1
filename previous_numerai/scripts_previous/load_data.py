"""
Functions to load the dataset.
"""

import numpy as np
import pandas as pd

def read_data(file_name):
    """This function is taken from:
    https://github.com/benhamner/BioResponse/blob/master/Benchmarks/csv_io.py
    """
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [x for x in line]
        samples.append(sample)
    return samples

def load():
    """Conveninence function to load all data as numpy arrays.
    """
    print "Loading data..."
    filename_train = 'numerai_datasets/numerai_training_data.csv'
    filename_test = 'numerai_datasets/numerai_tournament_data.csv'

    train = pd.read_csv("numerai_datasets/numerai_training_data.csv")
    y_train = np.array([x[0] for x in train])
    X_train = np.array([x[1:] for x in train])
    X_test = np.array(read_data("numerai_datasets/numerai_tournament_data.csv"))
    return X_train, y_train, X_test

if __name__ == '__main__':

    X_train, y_train, X_test = load()
