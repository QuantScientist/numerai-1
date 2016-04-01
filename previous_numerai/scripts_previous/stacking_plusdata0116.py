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
import random
random.seed(71) 

from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier as RF  
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline

import xgboost as xgb

from sknn.platform import cpu64, threading
from sknn.mlp import Classifier, Layer

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

def save_prediction(y_submission=[],id_column=[],path=''):
    data = pd.DataFrame(id_column)
    y_submission = pd.DataFrame(y_submission)
    data = pd.concat([data, y_submission],axis=1)
    data.columns = ['t_id','probability']
    data.to_csv(path, index = None)
    return 0

def run(X=None, y=None, X_submission=None, y_submission_val=None,pred_type='prediction',train_file='',test_file='',output_file='',stacked_level='stage1',creating_next_data=False,clfs=[]):#or validation
    #train_file = 'numerai_datasets/numerai_training_data.csv'
    #test_file = 'numerai_datasets/numerai_tournament_data.csv'
    #output_file = 'prediction/predictions_lr.csv'
    #x_trainはcolumns用
    global x_train, test_num, validation_flag, auc_stage

    ################## Stacking ###############
    #この段階でpredictionならX, y, X_submission
    #validationならX, y, X_submission, y_submission_valがわかっていれば良い

    np.random.seed(0) # seed to shuffle the train set

    n_folds = 3
    verbose = True
    shuffle = True 
    
    if shuffle:
        idx = np.random.permutation(y.size)
        X = X[idx]
        y = y[idx]
        validation_flag = validation_flag[idx]


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

        if str(clf).split("(")[0] in ['Classifier']: #weightのwarm_start防止
            clf_nn = clf   #clf_nnに保存



        for i, (train, test) in enumerate(skf):# # of n_fold
            print "Fold", i
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            if str(clf).split("(")[0] in ['XGBClassifier']:
                clf.fit(X_train, y_train, eval_metric='auc')
            elif str(clf).split("(")[0] in ['Classifier']:
                clf = clf_nn    #clfを初期化
                clf.fit(X_train, y_train)
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
        print "AUC: {:.4}+{:.2}, accuracy: {:.4}+{:.2} \n".format( np.mean(auc),np.std(auc), np.mean(acc), np.std(acc) )
        auc_stage[stacked_level].append(np.mean(auc))
        print
        if pred_type == 'prediction':
            print "saving individual model"
            indi_filename = output_file + '_{}{}_{}.csv'.format(str(clf).split("(")[0], j+1,stacked_level )
            save_prediction(y_submission=dataset_blend_test[:,j],id_column=id_column_list,path=indi_filename)
            #test_num['probability'] = dataset_blend_test[:,j]
            #test_num.to_csv( indi_filename, columns = ( 't_id', 'probability' ), index = None ) 
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
        next_dataset_blend_train['validation'] = validation_flag
        next_dataset_blend_test = pd.DataFrame(dataset_blend_test,columns=stacked_data_columns)
        next_dataset_blend_test = pd.concat([pd.DataFrame(id_column_list,columns=['t_id']),next_dataset_blend_test],axis=1)      
        #next_dataset_blend_test = pd.concat([test_num['t_id'],next_dataset_blend_test],axis=1)
        
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
        print "final AUC in validation set"
        print AUC( y_submission_val, y_submission )
    else:
        return y_submission


if __name__ == '__main__':
    auc_stage = {'stage1':[],'stage2':[]}
    print "Stage 0"
    print "Data Preparation for Stage 1"
    train_file = 'numerai_datasets/numerai_training_data.csv'   #  train data
    test_file = 'numerai_datasets/numerai_tournament_data.csv'  #  test data
    output_file = 'prediction/stacking_0116'            #  prediction data
    stacked_file = 'stacked_datasets/stacking_test0116_train_stage1.csv'     # stacked data
    
    pred_type = 'prediction' # Change to 'validation' !!

    # Load data
    ################ Prepare the data###################
    if pred_type == 'prediction':
        train, test = load_data(train_file, test_file, validation=False)

        # no need for validation flag
        validation_flag = train.validation.values
        train.drop( 'validation', axis = 1 , inplace = True )

        #tfidf of categorical feature
        from sklearn.preprocessing import LabelEncoder

        transformer = LabelEncoder()
        train['c1_label'] = transformer.fit_transform(train['c1'])      
        test['c1_label'] = transformer.transform(test['c1'])

        transformer = TfidfTransformer()
        tfidf_tr = transformer.fit_transform(train['c1_label'].values)

        del train['c1_label'], test['c1_label'] 
        train = pd.concat([train,pd.DataFrame(tfidf_tr.toarray().T, columns=map(lambda x:x+'_tfidf',['c1']))], axis=1)

        train_tfidf = pd.DataFrame(train.c1_tfidf.value_counts(),columns=['c1_tfidf'])
        test_tfidf = test.reset_index().merge(train[['c1','c1_tfidf']].drop_duplicates('c1'),on='c1',how='left').sort('index').drop('index',axis=1)['c1_tfidf'].values
        test = pd.concat([test,pd.DataFrame(test_tfidf, columns=['c1_tfidf'])], axis=1) 

        # Convert categorical column into dummy columns
        train_num, test_num = categorical_to_dummies(train, test, column='c1')

        #
        y_train = train_num.target.values
        y_test = None

        x_train = train_num.drop( 'target', axis = 1 )
        x_test = test_num.drop( 't_id', axis = 1 )   

    else:
        train, test = load_data(train_file, test_file, validation=True)

        #   drop validation column
        train.drop('validation', axis=1, inplace=True)
        test.drop('validation', axis=1, inplace=True) 
        assert( set( train.c1.unique()) == set( test.c1.unique())) 

        #   Convert categorical column into dummy columns
        train_num, test_num = categorical_to_dummies(train, test, column='c1')
        
        #
        y_train = train_num.target.values
        y_test = test_num.target.values

        x_train = train_num.drop( 'target', axis = 1 )
        x_test = test_num.drop( 'target', axis = 1 )

    numerical_col = [u'f1', u'f2', u'f3', u'f4', u'f5', u'f6', u'f7', u'f8', u'f9', u'f10', u'f11', u'f12', u'f13', u'f14']

    # sum of numerical features
    x_train['numerical_sum'] = x_train[numerical_col].sum(1)
    x_test['numerical_sum'] = x_test[numerical_col].sum(1)

    #tfidf of numerical feature
    transformer = TfidfTransformer()
    tfidf_tr = transformer.fit_transform(x_train[numerical_col])                         
    tfidf_te = transformer.transform(x_test[numerical_col])                         
    x_train = pd.concat([x_train,pd.DataFrame(tfidf_tr.toarray(), columns=[map(lambda x:x+'_tfidf',numerical_col)])], axis=1)
    x_test = pd.concat([x_test,pd.DataFrame(tfidf_te.toarray(), columns=[map(lambda x:x+'_tfidf',numerical_col)])], axis=1)

    # normalization
    pipeline = MinMaxScaler()#(X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    #pipeline = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
    print pipeline
    x_train[numerical_col] = pipeline.fit_transform( x_train[numerical_col] )
    x_test[numerical_col] = pipeline.transform( x_test[numerical_col] ) 
    
    x_train[['numerical_sum']] = pipeline.fit_transform( x_train['numerical_sum'] )
    x_test[['numerical_sum']] = pipeline.transform( x_test['numerical_sum'] )
    '''
    # 各numerical featuresの count を追加
    for num_col in numerical_col:
        a = x_train[num_col].value_counts()
        a = pd.DataFrame(a, columns=[num_col+'_count'])
        a[num_col] = a.index
        a.index = range(0,len(a))
        x_train = x_train.reset_index().merge(a, on=num_col, how='left').sort('index',axis=0).drop('index',axis=1)
        x_test = x_test.reset_index().merge(a, on=num_col, how='left').sort('index',axis=0).drop('index',axis=1)
    '''
    assert(~(False in x_train.columns == x_test.columns))
    X, y, X_submission, y_submission_val = x_train.values, y_train, x_test.values, y_test
    assert(X.shape[1] == X_submission.shape[1])

    id_column_list = test_num['t_id'].values
    ################## Feature Engineering ###############
    print "Feature Engineering"   

      

    print "################## Stacking for Stage 1 ####################"
    print "\nStage 1 ensenble"

    '''
    clfs = [RandomForestClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=600, n_jobs=-1, criterion='entropy'),
            ExtraTreesClassifier(n_estimators=500, n_jobs=-1, criterion='gini'),
            ExtraTreesClassifier(n_estimators=650, n_jobs=-1, criterion='entropy'),
            xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.01,
            max_depth=7, min_child_weight=1,n_estimators=800, nthread=-1, objective='binary:logistic', seed=0,silent=False, subsample=0.8),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.05,
            max_depth=8, min_child_weight=1,n_estimators=500, nthread=-1, objective='binary:logistic', seed=71,silent=False, subsample=0.8),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.1,
            max_depth=8, min_child_weight=1,n_estimators=350, nthread=-1, objective='binary:logistic', seed=401,silent=False, subsample=0.65),
            xgb.XGBClassifier(colsample_bytree=1, learning_rate=0.01,
            max_depth=6, min_child_weight=1,n_estimators=500, nthread=-1, objective='binary:logistic', seed=40,silent=False, subsample=0.9),
            xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.02,
            max_depth=7, min_child_weight=1,n_estimators=700, nthread=-1, objective='binary:logistic', seed=0,silent=False, subsample=0.75),
            xgb.XGBClassifier(colsample_bytree=0.7, learning_rate=0.01,
            max_depth=6, min_child_weight=1,n_estimators=650, nthread=-1, objective='binary:logistic', seed=0,silent=False, subsample=0.80),
            LR(),
            Classifier(layers=[
                                Layer("Rectifier", units=5, dropout=0.5),
                                Layer("Softmax")],
            learning_rate=0.01,n_iter=10,learning_rule='adam',verbose=True),
            Classifier(layers=[
                                Layer("Rectifier", units=10, dropout=0.3),
                                Layer("Softmax")],
            learning_rate=0.01,n_iter=10,learning_rule='adam',verbose=True),
            Classifier(layers=[
                                Layer("Rectifier", units=20, dropout=0.2),
                                Layer("Softmax")],
            learning_rate=0.01,n_iter=10,learning_rule='adam',verbose=True),
            ]
    
    '''
    clfs = [Classifier(layers=[
        Layer("Rectifier", units=300, dropout=0.0),
        Layer("Softmax")],
        learning_rate=0.1,n_iter=50,learning_rule='sgd',verbose=True)
            ]                                         
    '''
    clfs = [RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='gini'),
            RandomForestClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
            ]
    '''
    y_submission = run(X=X, y=y, X_submission=X_submission, y_submission_val=y_submission_val,pred_type=pred_type,train_file=train_file,test_file=test_file,output_file=output_file,stacked_level='stage1',creating_next_data=True,clfs=clfs)
    #run(pred_type='validation')
    print "saving..."
    output_file = output_file + '_stage1' + '.csv' 
    save_prediction(y_submission=y_submission,id_column=id_column_list,path=output_file)
    
    print "################## Done stacking for Stage 1 ####################"
    print 
    print "Data Preparation for Stage 2"
    train_file = 'stacked_datasets/stacking_train_stage1.csv'   #  train data
    test_file = 'stacked_datasets/stacking_test_stage1.csv'  #  test data
    output_file = 'prediction/stacking_0115'            #  prediction data
    
    pred_type = 'prediction' # Change to 'validation' !!

    # Load data
    ################ Prepare the data###################
    if pred_type == 'prediction':
        train, test = load_data(train_file, test_file, validation=False)

        # no need for validation flag
        validation_flag = train.validation.values
        train.drop( 'validation', axis = 1 , inplace = True )

        #
        y_train = train.target.values

        x_train = train.drop( 'target', axis = 1 )
        x_test = test.drop( 't_id', axis = 1 )   

        X, y, X_submission = x_train.values, y_train, x_test.values
        assert(X.shape[1] == X_submission.shape[1])
        y_submission_val = None
    else:
        train, test = load_data(train_file, test_file, validation=True)

        #   drop validation column
        train.drop('validation', axis=1, inplace=True)
        test.drop('validation', axis=1, inplace=True) 
        assert( set( train.c1.unique()) == set( test.c1.unique())) 

        #
        y_train = train.target.values
        y_test = test.target.values

        x_train = train.drop( 'target', axis = 1 )
        x_test = test.drop( 'target', axis = 1 )

        X, y, X_submission, y_submission_val = x_train.values, y_train, x_test.values, y_test
        assert(X.shape[1] == X_submission.shape[1])

    print "################## Stacking for Stage 2 ####################"
    print "\nStage 2 ensenble"

    clfs = [xgb.XGBClassifier(colsample_bytree=0.9, learning_rate=0.01,
            max_depth=6, min_child_weight=1,n_estimators=350, nthread=-1, objective='binary:logistic', seed=0,silent=False, subsample=0.9),
            LR(),
            ]  

    y_submission = run(X=X, y=y, X_submission=X_submission, y_submission_val=y_submission_val,pred_type=pred_type,train_file=train_file,test_file=test_file,output_file=output_file,stacked_level='stage2',creating_next_data=True,clfs=clfs) 
    print "saving..."
    output_file = output_file + '_stage2' + '.csv' 
    save_prediction(y_submission=y_submission,id_column=id_column_list,path=output_file)
    
    print auc_stage


    print "################## Linear ####################"
    final_prediction = pd.read_csv('stacked_datasets/stacking_test_stage2.csv')
    #final_prediction['probability'] = final_prediction.iloc[:,-len(clfs):].mean(1)
    final_prediction['probability'] = final_prediction.iloc[:,1:].mean(1)
    final_prediction.to_csv( 'prediction/final_linear_stage2_0117_v2.csv', columns = ( 't_id', 'probability' ), index = None )


     #final_prediction['probability'] = final_prediction[[u'f1', u'f2', u'f3', u'f4', u'f5', u'f6', u'f7', u'f8', u'f9', u'f10', u'f11', u'f12', u'f13', u'f14', u'c1_tfidf', u'c1_1', u'c1_10', u'c1_11', u'c1_12', u'c1_13', u'c1_14', u'c1_15', u'c1_16', u'c1_17', u'c1_18', u'c1_19', u'c1_20', u'c1_21', u'c1_22', u'c1_23', u'c1_24', u'c1_3', u'c1_4', u'c1_5', u'c1_6', u'c1_7', u'c1_8', u'c1_9', u'numerical_sum', u'f1_tfidf', u'f2_tfidf', u'f3_tfidf', u'f4_tfidf', u'f5_tfidf', u'f6_tfidf', u'f7_tfidf', u'f8_tfidf', u'f9_tfidf', u'f10_tfidf', u'f11_tfidf', u'f12_tfidf', u'f13_tfidf', u'f14_tfidf']].mean(1)


