#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
より一般的にするためにbaseに全てのパスを設定すれば済むように変更
data/input
data/output

VWを追加する！！

全データで予測できるようにする
s => Stacking
t => 全データで学習、予測
st => Stacking後、全データで学習予測
'''
import os
os.chdir('/Users/IkkiTanaka/numerai2/')

from base import INPUT_PATH, OUTPUT_PATH

import pandas as pd
PATH = '/Users/IkkiTanaka/numerai2/'

import numpy as np
np.random.seed(407)

#keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

#base utils
from base import load_data, save_pred_as_submit_format


#classifiers
from base import BaseModel, XGBClassifier, KerasClassifier, VWClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier

FEATURE_LIST_stage1 = {
                'train':('data/input/numerai_training_data.csv',
                         'data/input/MinMaxScaler_train.csv',
                         #'data/input/PolynomialFeatures_train.csv',
                         'data/input/RoundFloat_train.csv',
                         'data/input/AddFeatures_train.csv',
                         'data/input/SubFeatures_train.csv',
                         'data/input/LogTransform_train.csv',
                         ),#targetはここに含まれる
                'test':('data/input/numerai_tournament_data.csv',
                        'data/input/MinMaxScaler_test.csv',
                        #'data/input/PolynomialFeatures_test.csv',
                        'data/input/RoundFloat_test.csv',
                        'data/input/AddFeatures_test.csv',
                        'data/input/SubFeatures_test.csv',
                        'data/input/LogTransform_test.csv',
                        ),
                }


X,y,test  = load_data(flist=FEATURE_LIST_stage1)
assert((False in X.columns == test.columns) == False)
nn_input_dim = X.shape[1]
del X, y, test


PARAMS_V1 = {
            'colsample_bytree':0.9,
            'learning_rate':0.01,
            'max_depth':5, 'min_child_weight':1,
            'n_estimators':300, 'nthread':-1,
            'objective':'binary:logistic', 'seed':407,
            'silent':True, 'subsample':0.8
         }

class ModelV1(BaseModel):
        def build_model(self):
            return XGBClassifier(**self.params)

PARAMS_V2 = {
            'colsample_bytree':0.5,
            'learning_rate':0.01,
            'max_depth':6, 'min_child_weight':1,
            'n_estimators':300, 'nthread':-1,
            'objective':'binary:logistic', 'seed':407,
            'silent':True, 'subsample':0.7,'colsample_bylevel':0.9,
            'gamma':5,'reg_lambda':7,'reg_alpha':1,
         }

class ModelV2(BaseModel):
        def build_model(self):
            return XGBClassifier(**self.params)

PARAMS_V3 = {
            'colsample_bytree':0.8,
            'learning_rate':0.01,
            'max_depth':7, 'min_child_weight':1,
            'n_estimators':300, 'nthread':-1,
            'objective':'binary:logistic', 'seed':407,
            'silent':True, 'subsample':0.95,'colsample_bylevel':1.0,
            'gamma':2,'reg_lambda':0,'reg_alpha':1,
         }

class ModelV3(BaseModel):
        def build_model(self):
            return XGBClassifier(**self.params)


PARAMS_V4 = {
            'batch_size':128,
            'nb_epoch':50,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV4(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dense(input_dim=nn_input_dim, output_dim=150, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=150,output_dim=60, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=60,output_dim=2, init='uniform', activation='relu'))
            model.add(Activation('softmax'))
            #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer='sgd', loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)


PARAMS_V5 = {}
class ModelV5(BaseModel):
        def build_model(self):
            return LR()
        


PARAMS_V6 = {
    'trainCommand' : ("vw --loss_function logistic --l2 0.001 --learning_rate 0.01 --link=logistic --passes 20 --decay_learning_rate 0.97 --power_t 0 -d {}train_vw.data --cache_file vw.cache -f {}vw.model -b 28 --holdout_off --total 4".format(OUTPUT_PATH, OUTPUT_PATH)).split(' '), \

    'predictCommand': ("vw -t -d {}test_vw.data -i {}vw.model -p {}vw.predict".format(OUTPUT_PATH,OUTPUT_PATH,OUTPUT_PATH)).split(' ')
            }

class ModelV6(BaseModel):
        def build_model(self):
            return VWClassifier(**self.params)

PARAMS_V7 = {}
class ModelV7(BaseModel):
        def build_model(self):
            return KNeighborsClassifier()




#Final stageのモデル

PARAMS_V1_final = {
                'colsample_bytree':0.5,
                'learning_rate':0.01,
                'max_depth':6, 'min_child_weight':1,
                'n_estimators':200, 'nthread':-1,
                'objective':'binary:logistic', 'seed':407,
                'silent':True, 'subsample':0.7,'colsample_bylevel':0.9,
                'gamma':5,'reg_lambda':7,'reg_alpha':1,
                }

class ModelV1_final(BaseModel):
    def build_model(self):
        return XGBClassifier(**self.params)


PARAMS_V2_final = {
            'batch_size':128,
            'nb_epoch':130,
            'verbose':1, 
            'callbacks':[],
            'validation_split':0.,
            'validation_data':None,
            'shuffle':True,
            'show_accuracy':True,
            'class_weight':None,
            'sample_weight':None,
            'normalize':True,
            'categorize_y':True
            }

class ModelV2_final(BaseModel):
        def build_model(self):
            model = Sequential()
            model.add(Dense(input_dim=nn_input_dim2, output_dim=150, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=250,output_dim=160, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=160,output_dim=80, init='uniform', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(input_dim=80,output_dim=2, init='uniform', activation='relu'))
            model.add(Activation('softmax'))
            #sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

            model.compile(optimizer='sgd', loss='binary_crossentropy',class_mode='binary')

            return KerasClassifier(nn=model,**self.params)



if __name__ == "__main__":
    m = ModelV1(name="v1_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V1)
    m.run()

    m = ModelV2(name="v2_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V2)
    m.run()

    m = ModelV3(name="v3_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V3)
    m.run()

    m = ModelV4(name="v4_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V4)
    m.run()

    m = ModelV5(name="v5_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V5)
    m.run()

    m = ModelV6(name="v6_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V6)
    m.run()

    m = ModelV7(name="v7_stage1",
                flist=FEATURE_LIST_stage1,
                params = PARAMS_V7)
    m.run()

    ######## Final Model #########
    output_file = 'final.csv'
    FEATURE_LIST_stage2 = {
                'train':('data/input/numerai_training_data.csv',
                         'data/input/MinMaxScaler_train.csv',
                         #'data/input/PolynomialFeatures_train.csv',
                         'data/input/AddFeatures_train.csv',
                         'data/input/SubFeatures_train.csv',
                         'data/input/LogTransform_train.csv',
                         'data/output/v1_stage1_all_fold.csv',
                         'data/output/v2_stage1_all_fold.csv',
                         'data/output/v3_stage1_all_fold.csv',
                         'data/output/v4_stage1_all_fold.csv',
                         'data/output/v5_stage1_all_fold.csv',
                         'data/output/v6_stage1_all_fold.csv',
                         'data/output/v7_stage1_all_fold.csv',
                         ),#targetはここに含まれる
                'test':('data/input/numerai_tournament_data.csv',
                        'data/input/MinMaxScaler_test.csv',
                        #'data/input/PolynomialFeatures_test.csv',
                        'data/input/AddFeatures_test.csv',
                        'data/input/SubFeatures_test.csv',
                        'data/input/LogTransform_test.csv',
                        'data/output/v1_stage1_test.csv',
                        'data/output/v2_stage1_test.csv',
                        'data/output/v3_stage1_test.csv',
                        'data/output/v4_stage1_test.csv',
                        'data/output/v5_stage1_test.csv',
                        'data/output/v6_stage1_test.csv',
                        'data/output/v7_stage1_test.csv',
                        )
                      }

    X,y,test  = load_data(flist=FEATURE_LIST_stage2)
    assert((False in X.columns == test.columns) == False)
    nn_input_dim2 = X.shape[1]
    del X, y, test
    
    #X, y, test = load_data(flist=FEATURE_LIST_stage2)
    

    m = ModelV1_final(name="v1_final",
                     flist=FEATURE_LIST_stage2,
                     params = PARAMS_V1_final,
                     kind = 't',
                    )
    m.run()

    
    #m = ModelV2_final(name="v2_final",
    #                  flist=FEATURE_LIST_stage2,
    #                  params = PARAMS_V2_final,
    #                  kind = 'st',
    #                  )
    #m.run()

    save_pred_as_submit_format(OUTPUT_PATH+'v1_final_TestInAllTrainingData.csv', OUTPUT_PATH+output_file)
    #a.fit(X, y, eval_metric='logloss',eval_set=[(X, y),])
    '''
    #pred_random
    import random
    pred = pd.read_csv(OUTPUT_PATH+'v1_final_TestInAllTrainingData.csv').values
    submission = pd.read_csv(INPUT_PATH+'numerai_example_predictions.csv')
    submission['probability'] = pred

    for i in xrange(1000):
        random.seed(i)
        sample_num = random.randint(0,len(submission)/10)
        random.seed(i)
        random_index = random.sample(submission.index.values, sample_num)
        submission.loc[random_index, 'probability'] = 1 - submission.loc[random_index, 'probability']
    submission.to_csv( OUTPUT_PATH+output_file, columns = ( 't_id', 'probability' ), index = None )





    import xgboost as xgb

    dtrain = xgb.DMatrix(X,label=y)
    
    PARAMS_V1 = {
            'colsample_bytree':0.8,
            'learning_rate':0.01,
            'max_depth':7, 'min_child_weight':1,
            'nthread':6,'gamma':2,'reg_lambda':1,'reg_alpha':5,
            'objective':'binary:logistic',
            'silent':1, 'subsample':0.7
    }
    
    xgb.cv(PARAMS_V1,dtrain,num_boost_round=300, nfold=5,metrics={'logloss'},seed=407, show_stdv = False,show_progress=True,)


    import xgboost as xgb
    import matplotlib.pyplot as plt
    xgb.plot_importance(a.clf)
    plt.show()

    import matplotlib.pyplot as plt
    plt.scatter(X[u'feature1'],X[u'feature2'])
    plt.show()

    pred = pd.read_csv(OUTPUT_PATH+'v2_final_TestInAllTrainingData.csv').values
    submission = pd.read_csv(INPUT_PATH+'numerai_example_predictions.csv')
    
    rand_pred = []
    submission['probability'] = (((test.mean(1) - test.mean(1).mean())/test.mean(1).std()/100. + 0.5).values * 0.8+ pred * 1.2 )/2.0
    submission.to_csv( OUTPUT_PATH+output_file, columns = ( 't_id', 'probability' ), index = None )

    '''

    
    
    



    
