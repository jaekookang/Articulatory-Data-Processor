'''Utilities

2021-01-11 first created
'''

import os
import re
import textgrid
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from numpy.linalg import svd
from time import strftime, gmtime
import shutil
from ruamel.yaml import YAML


def safe_mkdir(folder):
    '''Make directory if not exists'''
    if not os.path.exists(folder):
        # os.mkdir(folder)
        os.makedirs(folder)


def safe_rmdir(folder):
    '''Remove directory if exists'''
    if os.path.exists(folder):
        shutil.rmtree(folder)


def get_time():
    '''Get current date & time'''
    return strftime('%Y-%m-%d_%H_%M_%S', gmtime())


def get_date():
    '''Get current date'''
    return strftime('%Y-%m-%d', gmtime())


def find_elements(pattern, my_list):
    '''Find elements in a list'''
    elements = []
    index = []
    for i, l in enumerate(my_list):
        if re.search(pattern, l):
            elements.append(my_list[i])
            index.append(i)
    return index, elements


def hz_to_mel(freq_hz):
    return 2595 * np.log10(1 + freq_hz/700)


def mel_to_hz(freq_mel):
    return (10**(freq_mel/2595) - 1) * 700


def drop_na(df):
    '''Drop NaNs from dataframe (IEEE)
    Returns
    - df: original or nan-removed dataframe
    - nans: list of nan indices
    '''
    # Check if na exists
    nans = df.isna().sum(axis=1).values
    nans = np.where(nans)[0]

    if len(nans) == 0:
        print('No NaNs were found')
        return df, nans
    else:
        print(f' {len(nans)} NaNs were found')
        df = df.drop(nans).reset_index(drop=True)
        return df, nans


def add_unique_token_id(df):
    '''Add unique token id to dataframe (IEEE)'''
    assert 'Token' not in df.columns.to_list(), '"Token" column already exists!'
    dfs = []
    for t in df.TimeAt.unique():
        d = df.loc[df.TimeAt == t].reset_index(drop=True)
        d['Token'] = d.index
        dfs.append(d)
    DC = pd.concat(dfs)
    DC.sort_values(by=['Token', 'TimeAt'], inplace=True)
    # Test if the original dataframe and indexed dataframe are the same or not; TRUE is expected.
    truth = df.equals(DC[df.columns].reset_index(drop=True))
    if truth:
        df = DC[['Token'] + df.columns.tolist()].reset_index(drop=True)
        df.Token = df.Token.astype('int')
        print('Index ("Token") column is added to the data')
        return df
    else:
        raise Exception('Check again when adding index to the data!!')


class Scaler:
    def __init__(self, which_spkr, Z):
        self.spkr = which_spkr
        self.Z = Z
        self.dict = self.Z[self.spkr]

    def transform(self, data, transform_type):
        mu = list(self.dict[transform_type]['mean'].values())
        mu = np.array(mu).reshape(1, -1)
        sd = list(self.dict[transform_type]['std'].values())
        sd = np.array(sd).reshape(1, -1)
        zdata = (data - mu)/sd
        return zdata

    def inverse_transform(self, zdata, transform_type):
        mu = list(self.dict[transform_type]['mean'].values())
        mu = np.array(mu).reshape(1, -1)
        sd = list(self.dict[transform_type]['std'].values())
        sd = np.array(sd).reshape(1, -1)
        data = zdata * sd + mu
        return data


class MY_PCA:
    def __init__(self, which_spkr, PC):
        self.spkr = which_spkr
        if isinstance(PC[self.spkr], list):
            self.V = np.array(PC[self.spkr])  # n_comp x n_dim
        else:
            self.V = np.array(PC[self.spkr]['components'])  # n_comp x n_dim

    def transform(self, data):
        data_pca = data.dot(self.V.T)  # (N,n_dim) x (n_dim,n_comp)
        return data_pca

    def inverse_transform(self, data_pca):
        data = data_pca.dot(self.V)  # (N,n_comp) x (n_comp,n_dim)
        return data


def remove_outlier_zscore(data, columns, threshold, log=False):
    # remove outliers based on zscored data using threshold
    data_ = data[columns]
    data_before = (data_ - data_.mean(axis=0))/data_.std(axis=0)
    idx = ((data_before < threshold) & (data_before > -threshold)).all(axis=1)
    data_after = data[idx].reset_index(drop=True)
    if log:
        before = data_before.shape[0]
        after = data_after.shape[0]
        txt = f'{before} to {after} ({before-after} removed) => {(after)/(before)*100:.2f}% remained'
        return data_after, txt
    else:
        return data_after  # zscored -> original


def MSE(y, yhat):
    # grand mean squared error given y and yhat
    # => This function calculate over entire values
    #   y, yhat: N x dim
    return np.square(y - yhat).mean()


def RMSE(y, yhat):
    # grand root mean squared error given y and yhat
    # => This function calculate over entire values
    #   y, yhat: N x dim
    return np.sqrt(np.square(y - yhat).mean())


class Map(dict):
    '''Dot dictionary
    Example:
      d = Map({'layer': 3, 'batch': 100, 'pooling': 'average'})
    Retrieved and edited from: https://stackoverflow.com/a/32107024
    '''

    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


class HParams(Map):
    '''Hyperparameter initializer
    '''

    def __init__(self, yaml_file, config_name=None):
        super().__init__()
        if isinstance(yaml_file, dict):
            # Read dictionary directly
            dictionary = yaml_file
        else:
            # Read yaml file
            with open(yaml_file) as f:
                if config_name is not None:
                    dictionary = YAML().load(f)[config_name]
                else:
                    dictionary = YAML().load(f)

        # Make dotted dictionary
        stack = [(k, v) for k, v in dictionary.items()]
        while stack:  # recursion
            key, val = stack.pop()
            if isinstance(val, dict):
                self.__setattr__(key, Map(dict(val)))
                stack.extend([(k, v) for k, v in val.items()])
            else:
                self.__setattr__(key, val)


if __name__ == '__main__':
    # Load hparams
    hp = HParams('hparams.yaml')

    # Save hparams
    with open('hparams_new.yaml', 'w') as f:
        YAML().dump(dict(hp), f)
