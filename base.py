#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
より一般的にするためにbaseに全てのパスを設定すれば済むように変更
data/input
data/output

VWを追加する！！

全データで予測できるようにする()
s => Stacking
t => 全データで学習、予測
st => Stacking後、全データで学習予測
'''

######### General #########
import numpy as np
import pandas as pd
import os, sys

######### Problem Type #########
eval_type = 'logloss' #'logloss', 'rmse'

problem_type = 'classification' #or 'regression'

classification_type = 'binary'# or 'multi-class'




######### PATH #########
os.chdir('/Users/IkkiTanaka/numerai2/')
PATH = '/Users/IkkiTanaka/numerai2/'

INPUT_PATH = '/Users/IkkiTanaka/numerai2/data/input/'
OUTPUT_PATH = '/Users/IkkiTanaka/numerai2/data/output/'

# for creating fold index
ORIGINAL_TRAIN_FORMAT = 'numerai_training_data.csv'

# for saving the submitted format file
SUBMIT_FORMAT = 'numerai_example_predictions.csv'

######### BaseEstimator ##########
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin

######### Keras #########
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils

######### XGBoost #########
import xgboost as xgb

######### Evaluation ##########
from sklearn.metrics import log_loss as ll
from sklearn.metrics import roc_auc_score as AUC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold, StratifiedKFold

######### Vowpal Wabbit ##########
#import wabbit_wappa as ww
import os
from time import asctime, time
import subprocess
import csv

######### CV index #########
n_folds = 5
print 'n_fold = ', n_folds

if os.path.isfile(INPUT_PATH + 'cv_id.csv') == False:
    train = pd.read_csv(INPUT_PATH + ORIGINAL_TRAIN_FORMAT)
    a = StratifiedKFold(train.target,n_folds=n_folds, shuffle=False, random_state=407)
    cv_id = {}
    fold_index = 0
    for train, test in a:
        print train,test,fold_index
        cv_id[fold_index] = {}
        cv_id[fold_index]['train'] = train
        cv_id[fold_index]['test'] = test
        fold_index += 1
    b = pd.DataFrame(cv_id).stack().T
    b.to_hdf(INPUT_PATH + 'cv_id.dump')
    
######### Utils #########

#feature listを渡してデータを作成するutil関数
def load_data(flist):
    '''
    flistにシリアライゼーションを渡すことでより効率的に
    data構造をここで考慮
    '''
    flist_len = len(flist['train'])
    X_train = pd.DataFrame()
    test = pd.DataFrame()
    for i in xrange(flist_len):
        X_train = pd.concat([X_train,pd.read_csv(PATH+flist['train'][i])],axis=1)
        test = pd.concat([test,pd.read_csv(PATH+flist['test'][i])],axis=1)

    y_train = X_train['target']
    del X_train['target']
    del test['t_id']
    print X_train.columns
    print test.columns
    assert( (False in X_train.columns == test.columns) == False)
    return X_train, y_train, test 

#最終予測結果を提出フォーマットで保存する
def save_pred_as_submit_format(pred_path, output_file):
    print 'read prediction <{}>'.format(pred_path)
    pred = pd.read_csv(pred_path).values
    #(((test.mean(1) - test.mean(1).mean())/test.mean(1).std()/100. + 0.5).values + pred)/2.0
    submission = pd.read_csv(INPUT_PATH+SUBMIT_FORMAT)
    submission['probability'] = pred
    submission.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )
    print 'done writing'
    return

#evalation function
def eval_pred( y_true, y_pred, eval_type=eval_type):
    if eval_type == 'logloss':#eval_typeはここに追加
        print "logloss: ", ll( y_true, y_pred )
        return ll( y_true, y_pred )             
    elif eval_type == 'auc':
        print "AUC: ", AUC( y_true, y_pred )
        return AUC( y_true, y_pred )             
    elif eval_type == 'rmse':
        print "rmse: ", rmse( y_true, y_pred )
        return rmse( y_true, y_pred )



######### BaseModel Class #########

class BaseModel(BaseEstimator):
    """
    Parameters of fit
    ----------
    FEATURE_LIST = {
                    'train':('flist_train.csv'),#targetはここに含まれる
                    'test':('flist_test.csv'),
                    }

    Note
    ----
    init: compiled model
    

    
    (Example)
    from base import BaseModel, XGBClassifier
    FEATURE_LIST = ["feat.group1.blp"]
    PARAMS = {
            'n_estimator':700,
            'sub_sample': 0.8,
            'seed': 71
        }
    class ModelV1(BaseModel):
         def build_model(self):
         return XGBClassifier(**self.params)


    if __name__ == "__main__":
        m = ModelV1(name="v1",
                    flist=FEATURE_LIST,
                    params=PARAMS,
                    kind='s')
        m.run()
   
    """
    def __init__(self, name="", flist={}, params={}, kind='s'):
        '''
        name: Model name
        flist: Feature list
        params: Parameters
        kind: Kind of run() 
        {'s': Stacking Only
         't': Training all data and predict test
         'st': Stacking and then training all data and predict test
               Using save final model with cross-validation
         'cv': Only cross validation without saving the prediction

        '''
        self.name = name
        self.flist = flist
        self.params = params
        self.kind = kind
        assert(self.kind in ['s', 't', 'st', 'cv'])
        

    def build_model(self):
        return None


    def run(self):
        
        X, y, test = self.load_data()
        
        if self.kind == 't':
            clf = self.build_model()
            clf.fit(X, y)
            y_submission = clf.predict_proba(test)#[:,1]#multi-class => 消す #コード変更
            y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
            y_submission.to_csv(OUTPUT_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)
            return 0 #保存して終了
        
        skf = pd.read_pickle(INPUT_PATH+'cv_id.dump')
        clf = self.build_model()
        print "Creating train and test sets for blending."
        #print "\nLevel 0"

        ############# for binary #############
        if problem_type == 'regression' or classification_type == 'binary':
            dataset_blend_train = np.zeros(X.shape[0]) #trainの予測結果の保存
            dataset_blend_test = np.zeros(test.shape[0]) #testの予測結果の保存
    
            #stacked_data_columns = X.columns.tolist()
            dataset_blend_test_j = np.zeros((test.shape[0], n_folds))
        
        ############# for multi-class #############
        elif classification_type == 'multi-class':
            #TODO
            pass


        evals = []
        for i in xrange(n_folds):# # of n_fold
            train_fold = skf['train'][i]
            test_fold = skf['test'][i]
            print "Fold", i
            #print X
            #print train_fold
            X_train = X.ix[train_fold]
            y_train = y.ix[train_fold]
            X_test = X.ix[test_fold]
            y_test = y.ix[test_fold]

            clf.fit(X_train, y_train)

            if problem_type == 'classification' and classification_type == 'binary':            
                #if using the mean of the prediction of each n_fold
                #print str(type(clf))
                if 'sklearn' in str(type(clf)):
                    y_submission = clf.predict_proba(X_test)[:,1]
                else:
                    y_submission = clf.predict_proba(X_test)

            elif problem_type == 'regression':      
                y_submission = clf.predict(X_test)

            dataset_blend_train[test_fold] = y_submission
            
            #外に持ってく
            evals.append(eval_pred(y_test, y_submission, eval_type))

            ############ binary classification ############
            if problem_type == 'classification' and classification_type == 'binary':            
                #if using the mean of the prediction of each n_fold
                if 'sklearn' in str(type(clf)):
                    dataset_blend_test_j[:, i] = clf.predict_proba(test)[:,1]
                else:
                    dataset_blend_test_j[:, i] = clf.predict_proba(test)

            ############ multi-class classification ############
            elif problem_type == 'classification' and classification_type == 'multi-class':            
                #TODO
                #if using the mean of the prediction of each n_fold
                #dataset_blend_test_j += clf.predict_proba(test)
                #dataset_blend_test_j /= n_folds
                pass

            ############ regression ############
            elif problem_type == 'regression':      
                #if using the mean of the prediction of each n_fold
                dataset_blend_test_j[:, i] = clf.predict(test)


        dataset_blend_test = dataset_blend_test_j.mean(1)
        
        for i in xrange(n_folds):
            print 'Fold{}: {}'.format(i+1, evals[i])
        print '{} Mean: '.format(eval_type), np.mean(evals), ' Std: ', np.std(evals)

        #Saving 上でモデルの保存も追加できる
        if self.kind != 'cv':
            print 'Saving results'
            dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=['{}_stack'.format(self.name)])
            dataset_blend_train.to_csv(OUTPUT_PATH+'{}_all_fold.csv'.format(self.name),index=False)
            dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=['{}_stack'.format(self.name)])
            dataset_blend_test.to_csv(OUTPUT_PATH+'{}_test.csv'.format(self.name),index=False)

        if self.kind == 'st':
            #Stacking(cross-validation)後に全データで学習
            clf = self.build_model()
            clf.fit(X, y)
            y_submission = clf.predict_proba(test)#[:,1]#multi-class => 消す #コード変更
            y_submission = pd.DataFrame(y_submission,columns=['{}_pred'.format(self.name)])
            y_submission.to_csv(OUTPUT_PATH+'{}_TestInAllTrainingData.csv'.format(self.name),index=False)

        return



    def load_data(self):
        '''
        flistにシリアライゼーションを渡すことでより効率的に
        data構造をここで考慮
        '''
        flist_len = len(self.flist['train'])
        X_train = pd.DataFrame()
        test = pd.DataFrame()
        for i in xrange(flist_len):
            X_train = pd.concat([X_train,pd.read_csv(PATH+self.flist['train'][i])],axis=1)
            test = pd.concat([test,pd.read_csv(PATH+self.flist['test'][i])],axis=1)

        y_train = X_train['target']
        del X_train['target']
        del test['t_id']
        print X_train.shape
        #print test.columns
        assert( (False in X_train.columns == test.columns) == False)
        return X_train, y_train, test 
        

######### Classifier Wrapper Class #########

class KerasClassifier(BaseEstimator, ClassifierMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):
        ###
        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1, 
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    show_accuracy=False,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False
        ###
        
        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=2, init='uniform'))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy',class_mode='binary')

            super(KerasModelV1, self).__init__(model,**params)
    
    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0., validation_data=None, shuffle=True, show_accuracy=True, class_weight=None, sample_weight=None, normalize=True, categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            show_accuracy=False, class_weight=None, sample_weight=None, normalize=True, categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.show_accuracy = show_accuracy
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, validation_data=None):
        X = X.values#Need for Keras
        y = y.values#Need for Keras
        if validation_data != None:
            self.validation_data = validation_data
            if self.normalize:
                self.validation_data[0] = (validation_data[0] - np.mean(validation_data[0],axis=0))/np.std(validation_data[0],axis=0)
            if self.categorize_y:
                self.validation_data[1] = np_utils.to_categorical(validation_data[1])

        if self.normalize:
            X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        if self.categorize_y:
            y = np_utils.to_categorical(y)
            
        #set initial weights
        self.nn.set_weights(self.init_weight)
        print X.shape
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, validation_data=self.validation_data, shuffle=self.shuffle, show_accuracy=self.show_accuracy, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict_proba(self, X, batch_size=128, verbose=1):
        X = X.values#Need for Keras
        if self.normalize:
            X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        
        if classification_type == 'binary':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)[:,1]#multi-class => 消す #コード変更
        elif classification_type == 'multi-class':
            return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)


class XGBClassifier(BaseEstimator, ClassifierMixin):
    """
    (Example)
    from base import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9, learning_rate=0.01,max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=0.8)
    a.fit(X_train, y_train, eval_metric='logloss',eval_set=[(X_train, y_train),(X_test, y_test)])
    
    """
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic",
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective

        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight

        self.base_score = base_score
        self.seed = seed
        self.missing = missing if missing is not None else np.nan
        self._Booster = None
        self.clf = xgb.XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                 n_estimators=n_estimators, silent=self.silent,
                 objective=self.objective,
                 nthread=self.nthread, gamma=self.gamma, min_child_weight=self.min_child_weight,
                 max_delta_step=self.max_delta_step, subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bylevel,
                 reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                 base_score=self.base_score, seed=self.seed, missing=self.missing)

    def fit(self, X, y=[], sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        return self.clf.fit(X,y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,early_stopping_rounds=early_stopping_rounds, verbose=verbose)

    def predict_proba(self, X, output_margin=False, ntree_limit=0):
        if classification_type == 'binary':
            return self.clf.predict_proba(X, output_margin=output_margin, ntree_limit=ntree_limit)[:,1]#multi-class => 消す #コード変更
        elif classification_type == 'multi-class':
            return self.clf.predict_proba(X, output_margin=output_margin, ntree_limit=ntree_limit)




class VWClassifier(BaseEstimator, ClassifierMixin):
    """
    PARAMS = {
    'trainCommand' : ("vw --loss_function logistic --l2 0.001 --learning_rate 0.015 --link=logistic --passes 20 --decay_learning_rate 0.97 --power_t 0 -d {}train_vw.data --cache_file vw.cache -f {}vw.model -b 20".format(OUTPUT_PATH, OUTPUT_PATH)).split(' '), \

    'predictCommand': ("vw -t -d {}test_vw.data -i {}vw.model -p {}vw.predict".format(OUTPUT_PATH,OUTPUT_PATH,OUTPUT_PATH)).split(' ')
    
    }
    """
    
    
    def __init__(self, trainCommand="", predictCommand="", train_vw_data="train_vw.data"):
        self.trainCommand = trainCommand
        self.predictCommand = predictCommand
        self.environmentDict = dict(os.environ, LD_LIBRARY_PATH='/usr/local/lib') 
        self.train_vw_data = train_vw_data

    def genTrainInstances(self, aRow):  
        #index = str(aRow['index'])
        #urlid = str(aRow['urlid'])
        y_row = str(int(float(aRow['target']))  )
        #rowtag = userid
        #rowText = (y_row + " 1.0  " + index)
        rowText = y_row
        col_names = aRow.index
        for i in col_names:
            if i in ['index','target']:
                continue
            rowText += " |{} {}:".format(i,i) + str(aRow[i])
        return  rowText

    def genTestInstances(self, aRow):  
        y_row = str(1)
        #index = str(aRow['index'])
        #urlid = str(aRow['urlid'])
        #rowtag = userid
        #rowText = (y_row + " 1.0  " + index)
        rowText = y_row
        col_names = aRow.index
        for i in col_names:
            if i in ['index','target']:
                continue
            rowText += " |{} {}:".format(i,i) + str(aRow[i])
        return  rowText

    def fit(self, X, y=[]):
        #delete vw.cache 
        subprocess.call(['rm','-f','{}vw.cache'.format(PATH)])
        #global df_train, trainCommand, environmentDict
        y = y.apply(lambda x: -1 if x < 1 else 1)
        X = pd.concat([X, y], axis=1)
        X = X.reset_index()
        #if os.path.isfile(OUTPUT_PATH+self.train_vw_data) == False:
        print "Generating VW Training Instances: ", asctime()
        X['TrainInstances'] = X.apply(self.genTrainInstances, axis=1)
        print "Finished Generating Train Instances: ", asctime()

        #if os.path.isfile(OUTPUT_PATH+self.train_vw_data) == False:
        print "Writing Train Instances To File: ", asctime()
        trainInstances = list(X['TrainInstances'].values)
        f = open(OUTPUT_PATH+'train_vw.data','w')
        f.writelines(["%s\n" % row  for row in trainInstances])
        f.close()
        print "Finished Writing Train Instances: ", asctime()
        #else:
        #    print 'already generated {}'.format(train_vw_data)
        subprocess.call(self.trainCommand)
        #subprocess.call(self.trainCommand, env=self.environmentDict)
        print "Finished Training: ", asctime()      
        return

    def readPredictFile(self):
        parseStr = lambda x: float(x) if '.' in x else int(x)
        y_pred = []
        with open(OUTPUT_PATH+'vw.predict', 'rb') as csvfile:
            predictions = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in predictions:
                pred = parseStr(row[0])
                y_pred.append(pred)
        return np.asarray(y_pred)  


    def predict_model(self,test):
        #global environmentDict, predictCommand, df_test
        test = test.reset_index()
        print "Building Test Instances: ", asctime()
        test['TestInstances'] = test.apply(self.genTestInstances, axis=1)
        print "Finished Generating Test Instances: ", asctime()

        print "Writing Test Instances: ", asctime()
        testInstances = list(test['TestInstances'].values)
        f = open(OUTPUT_PATH+'test_vw.data','w')
        f.writelines(["%s\n" % row  for row in testInstances])
        f.close()
        print "Finished Writing Test Instances: ", asctime()

        subprocess.call(self.predictCommand)
        #subprocess.call(self.predictCommand, env=self.environmentDict)

        #df_test['y_pred'] = readPredictFile()
        return self.readPredictFile()

    def predict_proba(self, X):
        if classification_type == 'binary':
            return self.predict_model(X)
        elif classification_type == 'multi-class':
            return self.predict_model(X) #Check!


######### Regressor Wrapper Class #########

class KerasRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import KerasClassifier
    class KerasModelV1(KerasClassifier):
        ###
        #Parameters for lerning
        #    batch_size=128,
        #    nb_epoch=100,
        #    verbose=1, 
        #    callbacks=[],
        #    validation_split=0.,
        #    validation_data=None,
        #    shuffle=True,
        #    show_accuracy=False,
        #    class_weight=None,
        #    sample_weight=None,
        #    normalize=True,
        #    categorize_y=False
        ###
        
        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=1, init='uniform'))
            model.add(Activation('linear')
            #################### CAUSION ####################
            # Change the output of last layer to 1          #
            # Change the loss to mse or mae                 #
            # Using mse loss results in faster convergence  #
            #################################################
            model.compile(optimizer='rmsprop', loss='mean_absolute_error')#'mean_squared_error'

            super(KerasModelV1, self).__init__(model,**params)
    
    KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0., validation_data=None, shuffle=True, show_accuracy=True, class_weight=None, sample_weight=None, normalize=True, categorize_y=True)
    KerasModelV1.fit(X_train, y_train,validation_data=[X_test,y_test])
    KerasModelV1.predict_proba(X_test)[:,1]
    """

    def __init__(self,nn,batch_size=128, nb_epoch=100, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            show_accuracy=False, class_weight=None, sample_weight=None, normalize=True, categorize_y=False):
        self.nn = nn
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.verbose = verbose
        self.callbacks = callbacks
        self.validation_split = validation_split
        self.validation_data = validation_data
        self.shuffle = shuffle
        self.show_accuracy = show_accuracy
        self.class_weight = class_weight
        self.sample_weight = sample_weight
        self.normalize = normalize
        self.categorize_y = categorize_y
        #set initial weights
        self.init_weight = self.nn.get_weights()

    def fit(self, X, y, validation_data=None):
        X = X.values#Need for Keras
        y = y.values#Need for Keras
        if validation_data != None:
            self.validation_data = validation_data
            if self.normalize:
                self.validation_data[0] = (validation_data[0] - np.mean(validation_data[0],axis=0))/np.std(validation_data[0],axis=0)
            #if self.categorize_y:
            #    self.validation_data[1] = np_utils.to_categorical(validation_data[1])

        if self.normalize:
            X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        #if self.categorize_y:
        #    y = np_utils.to_categorical(y)
            
        #set initial weights
        self.nn.set_weights(self.init_weight)
        print X.shape
        return self.nn.fit(X, y, batch_size=self.batch_size, nb_epoch=self.nb_epoch, verbose=self.verbose, callbacks=self.callbacks, validation_split=self.validation_split, validation_data=self.validation_data, shuffle=self.shuffle, show_accuracy=self.show_accuracy, class_weight=self.class_weight, sample_weight=self.sample_weight)

    def predict(self, X, batch_size=128, verbose=1):
        X = X.values#Need for Keras
        if self.normalize:
            X = (X - np.mean(X,axis=0))/np.std(X,axis=0)
        
        return self.nn.predict_proba(X, batch_size=batch_size, verbose=verbose)
    

class XGBRegressor(BaseEstimator, RegressorMixin):
    """
    (Example)
    from base import XGBClassifier
    class XGBModelV1(XGBClassifier):
        def __init__(self,**params):
            super(XGBModelV1, self).__init__(**params)

    a = XGBModelV1(colsample_bytree=0.9, learning_rate=0.01,max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='reg:linear', seed=0,silent=True, subsample=0.8)
    a.fit(X_train, y_train, eval_metric='logloss',eval_set=[(X_train, y_train),(X_test, y_test)])
    
    """
    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective='reg:linear',
                 nthread=-1, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective

        self.nthread = nthread
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight

        self.base_score = base_score
        self.seed = seed
        self.missing = missing if missing is not None else np.nan
        self._Booster = None
        self.clf = xgb.XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                 n_estimators=n_estimators, silent=self.silent,
                 objective=self.objective,
                 nthread=self.nthread, gamma=self.gamma, min_child_weight=self.min_child_weight,
                 max_delta_step=self.max_delta_step, subsample=self.subsample, colsample_bytree=self.colsample_bytree, colsample_bylevel=self.colsample_bylevel,
                 reg_alpha=self.reg_alpha, reg_lambda=self.reg_lambda, scale_pos_weight=self.scale_pos_weight,
                 base_score=self.base_score, seed=self.seed, missing=self.missing)

    def fit(self, X, y=[], sample_weight=None, eval_set=None, eval_metric=None,
            early_stopping_rounds=None, verbose=True):
        return self.clf.fit(X,y, sample_weight=sample_weight, eval_set=eval_set, eval_metric=eval_metric,early_stopping_rounds=early_stopping_rounds, verbose=verbose)

    def predict(self, X, output_margin=False, ntree_limit=0):
            return self.clf.predict(X, output_margin=output_margin, ntree_limit=ntree_limit)


class VWRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

