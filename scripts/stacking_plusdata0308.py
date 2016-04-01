#   load_dataでdataをcross-validationかvalidation_flagを用いるか判別し出力
#   main関数内
#      Stage 1
#         load_data
#         categorical変数の変形
#         feature_engineering
#         stage 1 の予測結果保存
#         stage 2 のためのデータセット保存
#
#      Stage 2
#         stage1で作成したload_data  
#         stage 2 の予測結果保存
#         stage 3 のためのデータセット保存  #
#
#      ...
#
#   他のデータセットで用いるにはload_dataと予測結果の保存方法変更せよ！！
#
#   予測値を元のデータにつけるversion
#
#


from __future__ import division
import numpy as np
import pandas as pd
import os

os.chdir('/Users/IkkiTanaka/numerai2')

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR
#from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import log_loss as AUC

from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier as RF  
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures

from sknn.platform import cpu64, threading
from sknn.mlp import Classifier, Layer

from base import KerasClassifier

def load_data(train_file, test_file, validation=False):
    print "Loading data"
    if validation == True:
        train = pd.read_csv( train_file )
        #   Create validation data
        iv = train.validation == 1  #indices of validation

        test = train[iv].copy()
        train = train[~iv].copy()
    else:
        train = pd.read_csv( train_file )
        test = pd.read_csv( test_file )     
    return train, test

def categorical_to_dummies(train, test, column):
    print "Converting categorical column into dummy columns "
    assert( set( train[column].unique()) == set( test[column].unique()))
    train_dummies = pd.get_dummies( train[column] )
    train_num = pd.concat(( train.drop( column, axis = 1 ), train_dummies ), axis = 1 )

    test_dummies = pd.get_dummies( test[column] )
    test_num = pd.concat(( test.drop( column, axis = 1 ), test_dummies ), axis = 1 ) 
    return train_num, test_num

def load_data(train_file, test_file, validation=False):
    print "Loading data"
    if validation == True:
        train = pd.read_csv( train_file )
        #   Create validation data
        iv = train.validation == 1  #indices of validation

        test = train[iv].copy()
        train = train[~iv].copy()
    else:
        train = pd.read_csv( train_file )
        test = pd.read_csv( test_file )     
    return train, test

def categorical_to_dummies(train, test, column):
    print "Converting categorical column into dummy columns "
    assert( set( train[column].unique()) == set( test[column].unique()))
    train_dummies = pd.get_dummies( train[column] )
    train_num = pd.concat(( train.drop( column, axis = 1 ), train_dummies ), axis = 1 )

    test_dummies = pd.get_dummies( test[column] )
    test_num = pd.concat(( test.drop( column, axis = 1 ), test_dummies ), axis = 1 ) 
    return train_num, test_num

def run(X=None, y=None, X_submission=None, y_submission_val=None,pred_type='prediction',train_file='',test_file='',output_file='',stacked_level='stage1',creating_next_data=False,clfs=[],n_folds=5):#or validation
    #train_file = 'numerai_datasets/numerai_training_data.csv'
    #test_file = 'numerai_datasets/numerai_tournament_data.csv'
    #output_file = 'prediction/predictions_lr.csv'
    #x_trainはcolumns用
    global x_train, test_num, auc_stage

    ################## Stacking ###############
    #この段階でpredictionならX, y, X_submission
    #validationならX, y, X_submission, y_submission_valがわかっていれば良い

    np.random.seed(0) # seed to shuffle the train set

    n_folds = n_folds
    verbose = True
    shuffle = True 
    
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
        #validation_flag = validation_flag[idx]


    skf = list(StratifiedKFold(y, n_folds))

    print "Creating train and test sets for blending."
    #print "\nLevel 0"

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))
    
    stacked_data_columns = x_train.columns.tolist()
    for j, clf in enumerate(clfs):
        print j, clf
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        acc = []
        auc = []
        for i, (train, test) in enumerate(skf):# # of n_fold
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            if str(clf).split("(")[0] in ['XGBClassifier']:
                #evallist  = [(X_test,'eval'), (X_train,'train')]
                clf.fit(X_train, y_train, eval_metric='logloss',eval_set=[(X_train, y_train),(X_test, y_test)])
            elif str(clf).split("(")[0] in ['Classifier']:
                #evallist  = [(X_test,'eval'), (X_train,'train')]
                mu = X_train.mean(0)
                stddev =  X_train.std(0)
                X_train = (X_train-mu) / stddev
                X_test = (X_test-mu) / stddev

                clf.fit(X_train, y_train)

            elif 'Keras' in str(clf).split("(")[0]:
                #evallist  = [(X_test,'eval'), (X_train,'train')]

                clf.fit(X_train, y_train,validation_data=[X_test,y_test])


            else:
                clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:,1]
            dataset_blend_train[test, j] = y_submission
            acc.append(accuracy( y_test, y_submission.round() ))
            auc.append(AUC( y_test, y_submission ))             
            #if using the mean of the prediction of each n_fold
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
        #if using the prediction of all train data
        #clf.fit(X, y)
        #dataset_blend_test[:,j] = clf.predict_proba(X_submission)[:,1] 
        print "clf: {}\n".format(clf)
        print 
        print "logloss: {:.4}+{:.2}, accuracy: {:.4}+{:.2} \n".format( np.mean(auc),np.std(auc), np.mean(acc), np.std(acc) )
        auc_stage[stacked_level].append(np.mean(auc))
        print
        if pred_type == 'prediction':
            print "saving individual model"
            indi_filename = output_file + '_{}{}_{}.csv'.format(str(clf).split("(")[0], j+1,stacked_level )
            test_num['probability'] = dataset_blend_test[:,j]
            test_num.to_csv( indi_filename, columns = ( 't_id', 'probability' ), index = None ) 
            stacked_data_columns.append('{}{}_{}.csv'.format(str(clf).split("(")[0], j+1,stacked_level))
    
    #
    auc_stage[stacked_level] = np.mean(auc_stage[stacked_level])

    #元のデータをpredictionにつける
    dataset_blend_train = np.concatenate((X,dataset_blend_train),axis=1)
    dataset_blend_test = np.concatenate((X_submission,dataset_blend_test),axis=1)

    # saving the stacked data for next stack level
    if pred_type == 'prediction' and creating_next_data == True:
        next_dataset_blend_train = pd.DataFrame(dataset_blend_train,columns=stacked_data_columns)
        next_dataset_blend_train['target'] = y
        #next_dataset_blend_train['validation'] = validation_flag
        next_dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=stacked_data_columns)
        next_dataset_blend_test = pd.concat([test_num['t_id'],next_dataset_blend_test],axis=1)

        next_dataset_blend_train.to_csv('stacked_datasets/stacking_train_{}.csv'.format(stacked_level),index=None)
        next_dataset_blend_test.to_csv('stacked_datasets/stacking_test_{}.csv'.format(stacked_level),index=None)

    #print "\nLevel 1"
    print "Blending."
    clf = LR()
    clf.fit(dataset_blend_train, y)
    y_submission = clf.predict_proba(dataset_blend_test)[:,1]

    #print "Linear stretch of predictions to [0,1]"
    #y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())
    if pred_type != 'prediction':
        print "final logloss in validation set"
        print AUC( y_submission_val, y_submission )
    else:
        print "saving..."

        test_num['probability'] = y_submission
        output_file = output_file + '_' + stacked_level + '.csv'
        test_num.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )



if __name__ == '__main__':
    auc_stage = {'stage1':[],'stage2':[]}
    print "Stage 0"
    print "Data Preparation for Stage 1"
    train_file = 'numerai_datasets/numerai_training_data.csv'   #  train data
    test_file = 'numerai_datasets/numerai_tournament_data.csv'  #  test data
    output_file = 'prediction/stacking_0115'            #  prediction data
    stacked_file = 'stacked_datasets/stacking_test0115_train_stage1.csv'     # stacked data
    
    pred_type = 'prediction' # Change to 'validation' !!

    # Load data
    ################ Prepare the data###################
    if pred_type == 'prediction':
        train, test = load_data(train_file, test_file, validation=False)

        # no need for validation flag
        #validation_flag = train.validation.values
        #train.drop( 'validation', axis = 1 , inplace = True )

        # Convert categorical column into dummy columns
        #train_num, test_num = categorical_to_dummies(train, test, column='c1')
        train_num = train.copy()
        test_num = test.copy()
        #
        y_train = train_num.target.values

        x_train = train_num.drop( 'target', axis = 1 )
        x_test = test_num.drop( 't_id', axis = 1 )   
        
        pipeline = Pipeline(steps=[
        ('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), 
        ('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))])
        #print pipeline
        #x_train = pipeline.fit_transform( x_train )
        #x_test = pipeline.transform( x_test )  

        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)

        X, y, X_submission = x_train.values, y_train, x_test.values
        assert(X.shape[1] == X_submission.shape[1])
        y_submission_val = None
    else:
        train, test = load_data(train_file, test_file, validation=True)

        #   drop validation column
        train.drop('validation', axis=1, inplace=True)
        test.drop('validation', axis=1, inplace=True) 
        assert( set( train.c1.unique()) == set( test.c1.unique())) 

        #   Convert categorical column into dummy columns
        #train, test = categorical_to_dummies(train, test, column='c1')
        train = train.copy()
        test = test.copy()

        #
        y_train = train.target.values
        y_test = test.target.values

        x_train = train.drop( 'target', axis = 1 )
        x_test = test.drop( 'target', axis = 1 )

        X, y, X_submission, y_submission_val = x_train.values, y_train, x_test.values, y_test
        assert(X.shape[1] == X_submission.shape[1])

    ################## Feature Engineering ###############

    # Feature Engineering
    print "Feature Engineering"
    #from sklearn.pipeline import Pipeline
    #from sklearn.preprocessing import Normalizer, PolynomialFeatures
    #pipeline = MinMaxScaler()#(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #pipeline = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
    #pipeline = Pipeline(steps=[
#('poly', PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)), 
#('scaler', MinMaxScaler(copy=True, feature_range=(0, 1)))])
    #print pipeline
    #X = pipeline.fit_transform( X )
    #X_submission = pipeline.transform( X_submission )       

    print "################## Stacking for Stage 1 ####################"
    print "\nStage 1 ensenble"

    
    clfs = [
            RandomForestClassifier(n_estimators=300, n_jobs=-1, criterion='entropy'),
            
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1, criterion='entropy'),
            xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.01,
            max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=0.8),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.05,
            max_depth=4, min_child_weight=1,n_estimators=150, nthread=-1, objective='binary:logistic', seed=71,silent=True, subsample=0.8),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.1,
            max_depth=5, min_child_weight=1,n_estimators=200, nthread=-1, objective='binary:logistic', seed=401,silent=True, subsample=0.65),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.01,
            max_depth=5, min_child_weight=1,n_estimators=300, nthread=-1, objective='binary:logistic', seed=40,silent=True, subsample=0.9),
            xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.02,
            max_depth=4, min_child_weight=1,n_estimators=300, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=0.75),
            LR(),
            xgb.XGBClassifier(colsample_bytree=1., learning_rate=0.01,
            max_depth=6, min_child_weight=5,n_estimators=200, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=1.,gamma=3),
            xgb.XGBClassifier(colsample_bytree=1., learning_rate=0.01,
            max_depth=5, min_child_weight=50,n_estimators=200, nthread=-1, objective='binary:logistic', seed=71,silent=True, subsample=1.,gamma=3),
            LR(max_iter=300,n_jobs=-1,penalty='l1'),
            LR(max_iter=300,solver='sag',n_jobs=-1),
            ]
    
    class KerasModelV1(KerasClassifier):
        """
        Parameters for lerning
            batch_size=128,
            nb_epoch=100,
            verbose=1, 
            callbacks=[],
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            show_accuracy=False,
            class_weight=None,
            sample_weight=None,
            normalize=True,
            categorize_y=False
        
        """
        def __init__(self,**params):
            model = Sequential()
            model.add(Dense(input_dim=X.shape[1], output_dim=100, init='uniform', activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(input_dim=50,output_dim=2, init='uniform'))
            model.add(Activation('softmax'))
            model.compile(optimizer='adam', loss='binary_crossentropy',class_mode='binary')

            super(KerasModelV1, self).__init__(model,**params)
    

    clfs = [#xgb.XGBClassifier(colsample_bytree=1., learning_rate=0.01,
            #max_depth=3, min_child_weight=5,n_estimators=50, nthread=-1, objective='binary:logistic', seed=71,silent=True, subsample=1.,gamma=3),
            KerasModelV1(batch_size=8, nb_epoch=10, verbose=1, callbacks=[], validation_split=0., validation_data=None, shuffle=True, show_accuracy=True, class_weight=None, sample_weight=None, normalize=True, categorize_y=True)
            ]

    run(X=X, y=y, X_submission=X_submission, y_submission_val=y_submission_val,pred_type=pred_type,train_file=train_file,test_file=test_file,output_file=output_file,stacked_level='stage1',creating_next_data=True,clfs=clfs, n_folds=10)

    print "################## Done stacking for Stage 1 ####################"
    print 
    print "Data Preparation for Stage 2"
    train_file = 'stacked_datasets/stacking_train_stage1.csv'   #  train data
    test_file = 'stacked_datasets/stacking_test_stage1.csv'  #  test data
    output_file = 'prediction/stacking_0115'            #  prediction data
    
    pred_type = 'prediction' # Change to 'validation' !!



    print "################## Stacking for Stage 2 ####################"
    print "\nStage 2 ensenble"

    clfs = [xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.01,
            max_depth=4, min_child_weight=1,n_estimators=150, nthread=-1, objective='binary:logistic', seed=0,silent=True, subsample=0.9),
            LR(),
            ]  

    run(X=X, y=y, X_submission=X_submission, y_submission_val=y_submission_val,pred_type=pred_type,train_file=train_file,test_file=test_file,output_file=output_file,stacked_level='stage2',creating_next_data=True,clfs=clfs) 

    print auc_stage


    print "################## Linear ####################"
    final_prediction = pd.read_csv('stacked_datasets/stacking_test_stage2.csv')
    final_prediction['probability'] = final_prediction.iloc[:,1:].mean(1)
    final_prediction['probability'] += 0.20
    final_prediction.to_csv( 'prediction/final_linear_stage2_0126_ver2.csv', columns = ( 't_id', 'probability' ), index = None ) #=>0.5509


    final_prediction = final_prediction[final_prediction.columns.drop([u'c1_1', u'c1_10', u'c1_11', u'c1_12', u'c1_13', u'c1_14', u'c1_15', u'c1_16', u'c1_17', u'c1_18', u'c1_19', u'c1_20', u'c1_21', u'c1_22', u'c1_23', u'c1_24', u'c1_3', u'c1_4', u'c1_5', u'c1_6', u'c1_7', u'c1_8', u'c1_9'])]
    final_prediction['probability'] = final_prediction.iloc[:,1:].mean(1)
    final_prediction.to_csv( 'prediction/final_linear_stage2_0126_ver2.csv', columns = ( 't_id', 'probability' ), index = None ) #=>0.5509

    final_prediction = pd.read_csv('stacked_datasets/stacking_test_stage2.csv')
    final_prediction['probability'] = final_prediction.iloc[:,1:].mean(1)
    a = final_prediction.corr()['probability'].copy()
    a.sort()
    new_col = a.index[-40:]
    final_prediction['probability'] = final_prediction[new_col].mean(1)
    final_prediction.to_csv( 'prediction/final_linear_stage2_0126_ver3.csv', columns = ( 't_id', 'probability' ), index = None ) #=>0.5439

