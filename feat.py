#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
originalデータのカラム名を変更
target => target
id => t_id
'''






import pandas as pd
import numpy as np
import os
os.chdir('/Users/IkkiTanaka/numerai2/')
import bloscpack
#INPUT_PATH
from base import INPUT_PATH


from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, MinMaxScaler

train = pd.read_csv('data/input/numerai_training_data.csv')
test = pd.read_csv('data/input/numerai_tournament_data.csv')

target = train.target
t_id = test.t_id

del train['target'], test['t_id']
ori_columns = train.columns


def MyPreprocessing(train, test,pipe=None):
    '''
    Wrapper of sklearn.preprocessing
    Saving the preprocessed data into "INPUT_PATH/{processing_name}_train(test).csv"

    Input:
    train, test

    return:
    None
    '''
    pipeline = pipe
    pipe_neme = str(pipe).split('(')[0]
    train = pipeline.fit_transform( train )
    test = pipeline.transform( test )
    train = pd.DataFrame(train, columns=map(str,range(train.shape[1])))
    test = pd.DataFrame(test, columns=map(str,range(test.shape[1])))
    train.columns += '_{}'.format(pipe_neme)
    test.columns += '_{}'.format(pipe_neme)

    #all same columns check
    for i in train.columns:
        val_cnt = train[i].value_counts()
        if len(val_cnt) <= 1:
            del train[i], test[i]
    assert((False in train.columns == test.columns) == False)
    #bloscpack.pack_ndarray_file(train,INPUT_PATH+'MinMaxScaler_train.csv')
    train.to_csv(INPUT_PATH+'{}_train.csv'.format(pipe_neme),index=False)
    test.to_csv(INPUT_PATH+'{}_test.csv'.format(pipe_neme),index=False)
    return None

def MyTSNE(train,test):
    #MyTSNE(train.iloc[:100,:],test.iloc[:20,:])
    model = TSNE(n_components=2, random_state=0)
    a = np.vstack(
            [train.values,
            test.values]
            )
    model.fit_transform(a)
    return

def MyRoundFloat(train, test):
    train = train.applymap(lambda x: round(x,ndigits=2))
    test = test.applymap(lambda x: round(x,ndigits=2))
    train.columns += '_{}'.format('RoundFloat')
    test.columns += '_{}'.format('RoundFloat')

    #all same columns check
    for i in train.columns:
        val_cnt = train[i].value_counts()
        if len(val_cnt) <= 1:
            del train[i], test[i]
    assert((False in train.columns == test.columns) == False)
    train.to_csv(INPUT_PATH+'{}_train.csv'.format('RoundFloat'),index=False)
    test.to_csv(INPUT_PATH+'{}_test.csv'.format('RoundFloat'),index=False)
    return

def MyAddFeatures(train, test):
    columns = train.columns
    add_train = pd.DataFrame()
    add_test = pd.DataFrame()
    for i in xrange(train.shape[1]):
        for j in xrange(train.shape[1]):
            if i < j:
                new_column = train.columns[i]+'+'+train.columns[j]
                add_train = pd.concat([add_train, pd.DataFrame((train.iloc[:,i] + train.iloc[:,j]).values, columns = [new_column])],axis=1)
                add_test = pd.concat([add_test, pd.DataFrame((test.iloc[:,i] + test.iloc[:,j]).values, columns = [new_column])],axis=1)
    #add_train.columns += '_{}'.format('AddFeatures')
    #add_test.columns += '_{}'.format('AddFeatures')
    assert((False in add_train.columns == add_test.columns) == False)
    add_train.to_csv(INPUT_PATH+'{}_train.csv'.format('AddFeatures'),index=False)
    add_test.to_csv(INPUT_PATH+'{}_test.csv'.format('AddFeatures'),index=False)
    return

def MySubFeatures(train, test):
    columns = train.columns
    add_train = pd.DataFrame()
    add_test = pd.DataFrame()
    for i in xrange(train.shape[1]):
        for j in xrange(train.shape[1]):
            if i < j:
                new_column = train.columns[i]+'-'+train.columns[j]
                add_train = pd.concat([add_train, pd.DataFrame((train.iloc[:,i] - train.iloc[:,j]).values, columns = [new_column])],axis=1)
                add_test = pd.concat([add_test, pd.DataFrame((test.iloc[:,i] - test.iloc[:,j]).values, columns = [new_column])],axis=1)
    #add_train.columns += '_{}'.format('AddFeatures')
    #add_test.columns += '_{}'.format('AddFeatures')
    assert((False in add_train.columns == add_test.columns) == False)
    add_train.to_csv(INPUT_PATH+'{}_train.csv'.format('SubFeatures'),index=False)
    add_test.to_csv(INPUT_PATH+'{}_test.csv'.format('SubFeatures'),index=False)
    return

def MyLogTransform(train, test):
    train = train.applymap(lambda x: np.log(x+1))
    test = test.applymap(lambda x: np.log(x+1))
    train.columns += '_{}'.format('Log')
    test.columns += '_{}'.format('Log')

    #all same columns check
    for i in train.columns:
        val_cnt = train[i].value_counts()
        if len(val_cnt) <= 1:
            del train[i], test[i]
    assert((False in train.columns == test.columns) == False)
    train.to_csv(INPUT_PATH+'{}_train.csv'.format('LogTransform'),index=False)
    test.to_csv(INPUT_PATH+'{}_test.csv'.format('LogTransform'),index=False)
    return


if __name__ == '__main__':
    MyPreprocessing(train, test, pipe=MinMaxScaler())
    MyPreprocessing(train, test, pipe=PolynomialFeatures(interaction_only=True))
    
    MyRoundFloat(train, test)

    MyAddFeatures(train, test)
    MySubFeatures(train, test)

    MyLogTransform(train, test)
    #pd.read_csv(INPUT_PATH + 'LogTransform_train.csv')
    #pd.read_csv(INPUT_PATH + 'SubFeatures_train.csv')

