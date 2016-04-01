#   script in 0115
#   XGBoost example code 
#

#   Import
import numpy as np
import pandas as pd
import xgboost as xgb
import os, sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score as accuracy
from sklearn.ensemble import RandomForestClassifier as RF

#
def train_and_evaluate( y_train, x_train, y_val, x_val, clf ):
    if clf == 'LR':
        lr = LR()
        lr.fit( x_train, y_train )

        p = lr.predict_proba( x_val )
        p_bin = lr.predict( x_val )

    elif clf == 'RF':
        n_trees = 200
        rf = RF( n_estimators = n_trees, verbose = True, n_jobs=4 )
        rf.fit( x_train, y_train )
        
        p = rf.predict_proba( x_val )
        p_bin = rf.predict( x_val )   

    acc = accuracy( y_val, p_bin )
    auc = AUC( y_val, p[:,1] )
    
    return ( auc, acc )

def transform_train_and_evaluate( transformer, clf='LR' ):

    global x_train, x_val, y_train
    
    x_train_new = transformer.fit_transform( x_train )
    x_val_new = transformer.transform( x_val )
    
    return train_and_evaluate( y_train, x_train_new, y_val, x_val_new, clf )


#   Set numerai folder PATH
os.chdir('/Users/IkkiTanaka/numerai/')
print 'present working directory: '
!pwd

#   Load data
data = pd.read_csv('numerai_datasets/numerai_training_data.csv')
test = pd.read_csv('numerai_datasets/numerai_tournament_data.csv')

#   Create validation data
iv = data.validation == 1  #indices of validation

val = data[iv].copy()
train = data[~iv].copy()

#   drop validation column
train.drop('validation', axis=1, inplace=True)
val.drop('validation', axis=1, inplace=True)

#   Check if the same values contains in the train and val sets
#   i.e. Check if the both data has the same number of the columns
assert( set( train.c1.unique()) == set( val.c1.unique())) 


#   Convert categorical column into dummy columns
train_dummies = pd.get_dummies(train.c1,prefix=True) 
train = pd.concat((train.drop('c1',axis=1), train_dummies), axis=1)

val_dummies = pd.get_dummies(val.c1,prefix=True) 
val = pd.concat((val.drop('c1',axis=1), val_dummies), axis=1)

#

y_train = train.target.values
y_val = val.target.values

x_train = train.drop( 'target', axis = 1 )
x_val = val.drop( 'target', axis = 1 )


#   train, predict, evaluate

auc, acc = train_and_evaluate( y_train, x_train, y_val, x_val, clf='RF'  )

print "No transformation"
print "AUC: {:.2%}, accuracy: {:.2%} \n".format( auc, acc )


# try different transformations for X

transformers = [ MaxAbsScaler(), MinMaxScaler(), RobustScaler(), StandardScaler(),  
    Normalizer( norm = 'l1' ), Normalizer( norm = 'l2' ), Normalizer( norm = 'max' ),
    PolynomialFeatures() ]

# Pipeline: Polynomial -> MinMaxScaler 
poly_scaled = Pipeline([ ( 'poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])
transformers.append( poly_scaled )

for transformer in transformers:

    print transformer
    auc, acc = transform_train_and_evaluate( transformer, clf='RF' )
    print "AUC: {:.2%}, accuracy: {:.2%} \n".format( auc, acc )




####################### prediction ##############################
train_file = 'numerai_datasets/numerai_training_data.csv'
test_file = 'numerai_datasets/numerai_tournament_data.csv'
output_file = 'numerai_datasets/predictions_lr.csv'

#

train = pd.read_csv( train_file )
test = pd.read_csv( test_file )

# no need for validation flag
train.drop( 'validation', axis = 1 , inplace = True )

# encode the categorical variable as one-hot, drop the original column afterwards
# but first let's make sure the values are the same in train and test

assert( set( train.c1.unique()) == set( test.c1.unique()))

train_dummies = pd.get_dummies( train.c1 )
train_num = pd.concat(( train.drop( 'c1', axis = 1 ), train_dummies ), axis = 1 )

test_dummies = pd.get_dummies( test.c1 )
test_num = pd.concat(( test.drop( 'c1', axis = 1 ), test_dummies ), axis = 1 )

#

y_train = train_num.target.values

x_train = train_num.drop( 'target', axis = 1 )
x_test = test_num.drop( 't_id', axis = 1 )

print "transforming..."

pipeline = MinMaxScaler()
#pipeline = Pipeline([ ('poly', PolynomialFeatures()), ( 'scaler', MinMaxScaler()) ])

x_train_new = pipeline.fit_transform( x_train )
x_test_new = pipeline.transform( x_test )

print "training..."

lr = LR()
lr.fit( x_train_new, y_train )

print "predicting..."

p = lr.predict_proba( x_test_new )

print "saving..."

test_num['probability'] = p[:,1]
test_num.to_csv( output_file, columns = ( 't_id', 'probability' ), index = None )

