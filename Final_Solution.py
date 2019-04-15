# This is the test branch code
# Import important libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import lightgbm as lgb

# Set seed for consistent results
np.random.seed(27)
seed_array = np.random.randint(5000, size=10)

# for train set
men_train = pd.read_csv('data/mens_train_file.csv')
women_train = pd.read_csv('data/womens_train_file.csv')
train = pd.concat([men_train, women_train], axis = 0)
id_train = train['id'].values
gender_train = train['gender'].values
train.drop(['id', 'train'], axis = 1, inplace = True)
temp1 = pd.DataFrame()
temp1['id'] = id_train
temp1['gender'] = gender_train
temp1 = pd.DataFrame(temp1.apply(lambda row: '_'.join(map(str, row)), axis=1))
temp1.columns = ['unique_id']
id_unique_train = temp1['unique_id'].values

# for test set
men_test = pd.read_csv('data/mens_test_file.csv')
women_test = pd.read_csv('data/womens_test_file.csv')
test = pd.concat([men_test, women_test], axis = 0)
id_test = test['id'].values
gender_test = test['gender'].values
test.drop(['id', 'train', 'outcome'], axis = 1, inplace = True)
temp2 = pd.DataFrame()
temp2['id'] = id_test
temp2['gender'] = gender_test
temp2 = pd.DataFrame(temp2.apply(lambda row: '_'.join(map(str, row)), axis=1))
temp2.columns = ['unique_id']
id_unique_test = temp2['unique_id'].values

# Label encoding the categorical variables
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['outcome'].values))
train['outcome'] = lbl.transform(list(train['outcome'].values))
            
for f in train.columns:
        if train[f].dtype=='object':
            print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values) + list(test[f].values))
            train[f] = lbl.transform(list(train[f].values))
            test[f] = lbl.transform(list(test[f].values))
            
# Target Variable
y = train['outcome'].values
train.drop('outcome', axis = 1, inplace = True)    

# 10 fold stratified cross validation
K = 10
test['UE'] = np.zeros(test.shape[0])
test['FE'] = np.zeros(test.shape[0])
test['W'] = np.zeros(test.shape[0])

j = 1
# Train lightgbm model for 10 different seeds, with 10 fold stratified cross validation
for i in seed_array:
    print(j)
    kf = StratifiedKFold(n_splits = K, random_state = i, shuffle=False)    
    for train_index, test_index in kf.split(train, y):
        # specify your configurations as a dict
        params = {'task': 'train',
                  'boosting_type': 'gbdt',
                  'objective': 'multiclass',
                  'num_class':3,
                  'metric': 'multi_logloss',
                  'learning_rate': 0.01,
                  'num_leaves': 31,
                  'feature_fraction': 0.7,
                  'bagging_fraction': 0.7,
                  'bagging_freq': 1,
                  'max_bin': 255,
                  'seed': i, 
                  'silence': True
                 }
        train_X, valid_X = train.iloc[train_index,:], train.iloc[test_index,:]
        train_y, valid_y = y[train_index], y[test_index]
        train_id, valid_id = id_unique_train[train_index], id_unique_train[test_index]
        
        lgb_train = lgb.Dataset(train_X, train_y)        
        lgb_cv = lgb.cv(params, lgb_train, num_boost_round=3000, nfold=10, shuffle=True, stratified=True, early_stopping_rounds=50)
        nround = lgb_cv['multi_logloss-mean'].index(np.min(lgb_cv['multi_logloss-mean']))
        
        model = lgb.train(params, lgb_train, num_boost_round=nround)
        
        temp2 = model.predict(test)
        test['UE'] += temp2[:,1]
        test['FE'] += temp2[:,0]
        test['W'] += temp2[:,2]  
    j = j + 1
    
test['UE'] /= (K * len(seed_array))
test['FE'] /= (K * len(seed_array))
test['W'] /= (K * len(seed_array))

# Make a submission
sub = pd.DataFrame()
sub['id'] = id_test
sub['gender'] = gender_test
sub = pd.DataFrame(sub.apply(lambda row: '_'.join(map(str, row)), axis=1))
sub= sub.rename(index=str, columns={0: "submission_id"})
sub['UE'] = test['UE'].values
sub['FE'] = test['FE'].values
sub['W'] = test['W'].values

sample = pd.read_csv('data/AUS_SubmissionFormat.csv')
sample.drop(['UE','FE','W'], axis = 1, inplace = True)
sample = sample.merge(sub, on = 'submission_id', how = 'left')
sample.to_csv('submissions/lgb_ensemble_10.csv', index=False)
