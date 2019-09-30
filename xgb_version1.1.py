#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
import time
import datetime
import warnings
import gc
import os
import pickle
import multiprocessing
import itertools
import random
from tqdm import tqdm
from scipy.stats import ks_2samp
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
#import cufflinks as cf
#from IPython.display import display,HTML
#from plotly.offline import init_notebook_mode
#cf.go_offline()
#%reload_ext autoreload
#%autoreload 2
#%matplotlib inline
#cf.set_config_file(theme='ggplot',sharing='public',offline=True)
#init_notebook_mode(connected=False)  
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# np.set_printoptions(suppress=True)
# pd.set_option('precision', 5)
# pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法


# In[2]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[3]:


def load_data(file):
    if (file.split('.'))[-1] == 'csv':
        return reduce_mem_usage(pd.read_csv(file))
    elif (file.split('.'))[-1] == 'pkl':
        return pd.read_pickle(file)
    else:
        raise IOError("Error: unknown file type: "+file)


# In[4]:


def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# In[5]:


def relax_data(df_train, df_test, col):
    cv1 = pd.DataFrame(df_train[col].value_counts().reset_index().rename({col:'train'},axis=1))
    cv2 = pd.DataFrame(df_test[col].value_counts().reset_index().rename({col:'test'},axis=1))
    cv3 = pd.merge(cv1,cv2,on='index',how='outer')
    factor = len(df_test)/len(df_train)
    cv3['train'].fillna(0,inplace=True)
    cv3['test'].fillna(0,inplace=True)
    cv3['remove'] = False
    cv3['remove'] = cv3['remove'] | (cv3['train'] < len(df_train)/10000)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] < cv3['test']/3)
    cv3['remove'] = cv3['remove'] | (factor*cv3['train'] > 3*cv3['test'])
    cv3['new'] = cv3.apply(lambda x: x['index'] if x['remove']==False else 0,axis=1)
    cv3['new'],_ = cv3['new'].factorize(sort=True)
    cv3.set_index('index',inplace=True)
    cc = cv3['new'].to_dict()
    df_train[col] = df_train[col].map(cc)
    df_test[col] = df_test[col].map(cc)
    return df_train, df_test


# In[6]:


def aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False):
    for main_col in main_cols:
        for cate_col in cate_cols:
            new_col_names = [cate_col+'_'+main_col+'_'+agg_type for agg_type in agg_types]
            temp_group = pd.concat([train_df[[cate_col, main_col]], test_df[[cate_col, main_col]]]).groupby([cate_col])
            temp_df = temp_group[main_col].agg(dict(zip(new_col_names, agg_types)))
            
            for new_col_name in new_col_names:
                train_df[new_col_name] = train_df[cate_col].map(temp_df[new_col_name])
                test_df[new_col_name] = test_df[cate_col].map(temp_df[new_col_name])
            
            
            if with_diff:
                temp_col_names = []
                nec_aggs = ['median', 'mean']
                
                for agg_type in nec_aggs:
                    if agg_type not in agg_types:
                        temp_col_name = cate_col+'_'+main_col+'_'+agg_type
                        temp_df = temp_group[main_col].agg({temp_col_name : agg_type})
                        
                        for df in [train_df, test_df]:
                            df[temp_col_name] = df[cate_col].map(temp_df[temp_col_name])
                    
                        temp_col_names.append(temp_col_name)
                
                for df in [train_df, test_df]:
                    df[cate_col+'_'+main_col+'_diff_median'] = df[main_col] - df[cate_col+'_'+main_col+'_median']
                    df[cate_col+'_'+main_col+'_diff_mean'] = df[main_col] - df[cate_col+'_'+main_col+'_mean']
                    
                for temp_col_name in temp_col_names:
                    for df in [train_df, test_df]:
                        del df[temp_col_name]

            
            if with_norm:
                temp_col_names = []
                nec_aggs = ['min', 'max', 'mean', 'std']
                
                for agg_type in nec_aggs:
                    if agg_type not in agg_types:
                        temp_col_name = cate_col+'_'+main_col+'_'+agg_type
                        temp_df = temp_group[main_col].agg({temp_col_name : agg_type})
                    
                        for df in [train_df, test_df]:
                            df[temp_col_name] = df[cate_col].map(temp_df[temp_col_name])
                    
                        temp_col_names.append(temp_col_name)
                    
                for df in [train_df, test_df]:
                    df[cate_col+'_'+main_col+'_norm'] = (df[main_col] - df[cate_col+'_'+main_col+'_min'])/(df[cate_col+'_'+main_col+'_max'] - df[cate_col+'_'+main_col+'_min'])
                    df[cate_col+'_'+main_col+'_zscore'] = (df[main_col] - df[cate_col+'_'+main_col+'_mean'])/df[cate_col+'_'+main_col+'_std']
                
                for temp_col_name in temp_col_names:
                    for df in [train_df, test_df]:
                        del df[temp_col_name]
            
    return train_df, test_df


# In[7]:


def values_normalization(dt_df, periods, columns):
    for period in periods:
        for col in columns:
            new_col = period +'_'+ col
            dt_df[col] = dt_df[col].astype(float)  

            temp_min = dt_df.groupby([period])[col].agg(['min']).reset_index()
            temp_min.index = temp_min[period].values
            temp_min = temp_min['min'].to_dict()

            temp_max = dt_df.groupby([period])[col].agg(['max']).reset_index()
            temp_max.index = temp_max[period].values
            temp_max = temp_max['max'].to_dict()

            dt_df['temp_min'] = dt_df[period].map(temp_min)
            dt_df['temp_max'] = dt_df[period].map(temp_max)

            dt_df[new_col+'_norm'] = (dt_df[col]-dt_df['temp_min'])/(dt_df['temp_max']-dt_df['temp_min'])

            del dt_df['temp_min'],dt_df['temp_max']
    return dt_df


# In[8]:


def tfidf_encoding(train_df, test_df, target_col, refer_col):
    
    df = pd.concat([train_df, test_df],axis=0)
    lists = df.groupby(refer_col)[target_col].apply(lambda s:' '.join(s.to_list()))
    
    vectorizer = CountVectorizer()
    transformer = TfidfTransformer()
    X = transformer.fit_transform(vectorizer.fit_transform(lists))
    tfidf_df = pd.DataFrame(X.toarray(), index = lists.index, columns = vectorizer.get_feature_names())
    
    train_df[target_col+'_tfidf'] = train_df[[refer_col,target_col]].apply(lambda s: tfidf_df[s[target_col]][s[refer_col]],axis=1)
    test_df[target_col+'_tfidf'] = test_df[[refer_col,target_col]].apply(lambda s: tfidf_df[s[target_col]][s[refer_col]],axis=1)
    
    return train_df, test_df


# In[9]:


def make_predictions_gkf(train_df, test_df, feature_cols, target, param, NFOLDS=2):
    gkf = GroupKFold(n_splits=NFOLDS)
    split_groups = train_df['DT_M']
    
    test_pred_prob = np.zeros(test_num)
    oof_pred_prob = np.zeros(train_num)
    
    train_values = train_df[feature_cols]
    test_values = test_df[feature_cols]
    labels = train_df['isFraud']
    split_groups = train_df['DT_M']
    
    for i, (train_idx, valid_idx) in enumerate (gkf.split(train_values, labels, groups = split_groups)):
        print(i,'fold...')
        start_time = time.time()
    
        train_x, train_y = train_values.iloc[train_idx], labels[train_idx]
        valid_x, valid_y = train_values.iloc[valid_idx], labels[valid_idx]
    
        # Construct the dataset
        train_data = lgb.Dataset(train_x, label=train_y, free_raw_data = True)
        valid_data = lgb.Dataset(valid_x, label=valid_y, reference = train_data, free_raw_data = True)
    
        # Training
        bst = lgb.train(param, train_data, valid_sets=[train_data, valid_data],verbose_eval=200)
        
        # Prediction
        valid_pred_prob = bst.predict(valid_x, num_iteration=bst.best_iteration)
        oof_pred_prob[valid_idx] = valid_pred_prob
        print('val logloss: ', log_loss(valid_y, valid_pred_prob))
        print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
        
        test_pred_prob += bst.predict(test_values, num_iteration=bst.best_iteration)/gkf.n_splits
         
        print('runtime: {}\n'.format(time.time() - start_time))
    
        # Plotting
        lgb.plot_importance(bst,max_num_features=30)
    
    print('oof logloss: ', log_loss(labels, oof_pred_prob))
    print('oof auc: ', roc_auc_score(labels, oof_pred_prob))
    
    test_df['isFraud'] = test_pred_prob
    return test_df[['TransactionID','isFraud']]


# In[10]:


def make_predictions_kf(train_df, test_df, feature_cols,cate_cols, labels, param, NFOLDS=2):
    kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    split_groups = train_df['DT_M']
    
    test_pred_prob = np.zeros(test_num)
    oof_pred_prob = np.zeros(train_num)
    
    train_values = train_df[feature_cols]
    test_values = test_df[feature_cols]
    
    feature_imp = pd.DataFrame()
    feature_imp['feature'] = train_values.columns
    
    if LOCAL_TEST:
        test_labels = test_df[TARGET]
    
    
    for i, (train_idx, valid_idx) in enumerate (kf.split(train_values, split_groups)):
        print(i,'fold...')
        start_time = time.time()
    
        train_x, train_y = train_values.iloc[train_idx], labels[train_idx]
        valid_x, valid_y = train_values.iloc[valid_idx], labels[valid_idx]
        
        if LOCAL_TEST:
            # Construct the dataset
            train_data = xgb.DMatrix(train_x, train_y)
            valid_data = xgb.DMatrix(valid_x, valid_y)
            test_data = xgb.DMatrix(test_values, test_labels)
            
            watchlist = [(train_data, 'train'), (valid_data, 'valid_data'), (test_data, 'local_test_data')]
            
            # Training
            bst = xgb.train(param, train_data, num_boost_round=20000, 
                            early_stopping_rounds=100, evals=watchlist, verbose_eval=200)
            
            # Prediction
            valid_pred_prob = bst.predict(xgb.DMatrix(valid_x), ntree_limit=bst.best_ntree_limit)
            oof_pred_prob[valid_idx] =  valid_pred_prob
            # print('val logloss: ', log_loss(valid_y, valid_pred_prob))
            print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
            
            cur_test_pred_prob = bst.predict(xgb.DMatrix(test_values), ntree_limit=bst.best_ntree_limit)
            # print('val logloss: ', log_loss(test_labels, cur_test_pred_prob))
            print('current test auc: ', roc_auc_score(test_labels, cur_test_pred_prob))
            
            test_pred_prob += cur_test_pred_prob/kf.n_splits
            
            feature_imp['fold_{}'.format(i)] = bst.get_fscore()
            
        else:   
            # Construct the dataset
            train_data = xgb.DMatrix(train_x, train_y)
            valid_data = xgb.DMatrix(valid_x, valid_y)
            
            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
            
            # Training
            bst = lgb.train(param, train_data, num_boost_round=4000, 
                            early_stopping_rounds=100, evals=watchlist, verbose_eval=200)
        
            # Prediction
            valid_pred_prob = bst.predict(xgb.DMatrix(valid_x), ntree_limit=bst.best_ntree_limit)
            oof_pred_prob[valid_idx] =  valid_pred_prob
            # print('val logloss: ', log_loss(valid_y, valid_pred_prob))
            print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
            
            test_pred_prob += bst.predict(xgb.DMatrix(test_values), ntree_limit=bst.best_ntree_limit)/kf.n_splits
         
        print('runtime: {}\n'.format(time.time() - start_time))
    
        # Plotting
        # lgb.plot_importance(bst,max_num_features=30)
    
    # print('oof logloss: ', log_loss(labels, oof_pred_prob))
    print('oof auc: ', roc_auc_score(labels, oof_pred_prob))
    
    if LOCAL_TEST:
        print('test auc: ', roc_auc_score(test_labels, test_pred_prob))
        
        feature_imp['average'] = feature_imp[['fold_{}'.format(fold) for fold in range(kf.n_splits)]].mean(axis=1)
        feature_imp.to_csv('feature_importances.csv')

        plt.figure(figsize=(16, 16))
        sns.barplot(data=feature_imp.sort_values(by='average', ascending=False).head(50), x='average', y='feature');
        plt.title('50 TOP feature importance over {} folds average'.format(kf.n_splits));
        
        test_df['pred_isFraud'] = test_pred_prob
        return test_df[['TransactionID','pred_isFraud']]
    
    else:
        test_df['isFraud'] = test_pred_prob
        return test_df[['TransactionID','isFraud']]


# # Global variables

# In[11]:


SEED = 42
seed_everything(SEED)
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
KS_TEST = False
LOCAL_TEST = False
NEG_SAMPLE = False


# # Load data

# In[12]:


if LOCAL_TEST:
    files=['data/train_transaction.pkl',
           'data/train_identity.pkl']
    
    with multiprocessing.Pool() as pool:
        print("Loading...")
        try:
            train_df, train_id = pool.map(load_data, files)
        except IOError as er:
            print (er)
        else:
            print ("Loading done")
        
        train_df['DT_M'] = train_df['TransactionDT'].apply(lambda x: (START_DATE + datetime.timedelta(seconds = x)))
        train_df['DT_M'] = (train_df['DT_M'].dt.year-2017)*12 + train_df['DT_M'].dt.month 
        test_df = train_df[train_df['DT_M']==train_df['DT_M'].max()].reset_index(drop=True)
        train_df = train_df[train_df['DT_M']<(train_df['DT_M'].max())].reset_index(drop=True)
        
        test_id  = train_id[train_id['TransactionID'].isin(test_df['TransactionID'])].reset_index(drop=True)
        train_id = train_id[train_id['TransactionID'].isin(train_df['TransactionID'])].reset_index(drop=True)
        
        del train_df['DT_M'], test_df['DT_M']
    
    
else:    
    files=['data/train_transaction.pkl',
           'data/test_transaction.pkl',
           'data/train_identity.pkl',
           'data/test_identity.pkl']

    with multiprocessing.Pool() as pool:
        print("Loading...")
        try:
            train_df, test_df, train_id, test_id = pool.map(load_data, files)
        except IOError as er:
            print (er)
        else:
            print ("Loading done")


# In[13]:


train_df = train_df[train_df['TransactionAmt']<5000].reset_index(drop=True)


# In[14]:


train_base_cols = list(train_df)
test_base_cols = list(test_df)

labels = train_df[TARGET]

train_num = train_df.shape[0]
test_num = test_df.shape[0]

train_df['TransactionAmt'] = train_df['TransactionAmt'].astype(float)
test_df['TransactionAmt'] = test_df['TransactionAmt'].astype(float)


# In[15]:


gc.collect()


# In[16]:


print('Shape control: ', train_df.shape, test_df.shape, train_id.shape, test_id.shape)


# In[17]:


train_df.head()


# # TransactionDT transformation 

# In[18]:


train_df['DT'] = train_df['TransactionDT'].apply(lambda s:(START_DATE + datetime.timedelta(seconds = s)))
test_df['DT'] = test_df['TransactionDT'].apply(lambda s:(START_DATE + datetime.timedelta(seconds = s)))

for df in [train_df, test_df]:
    # total count of time periods
    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear
    df['DT_H'] = df['DT_D']*24 + df['DT'].dt.hour
    # datetime
    df['hour'] = df['DT'].dt.hour
    df['dayofweek'] = df['DT'].dt.dayofweek
    df['day'] = df['DT'].dt.day
    df['month'] = df['DT'].dt.month


# # Possible solo features

# In[19]:


################### Number of nan
card_cols = ['card1', 'card2', 'card3', 'card4', 'card5','card6']
M_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']
C_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
D_cols = ['D{}'.format(i) for i in range(1,16)]
V1_11_cols = ['V{}'.format(i) for i in range(1,12) ]
V12_34_cols = ['V{}'.format(i) for i in range(12,35)]
V35_52_cols = ['V{}'.format(i) for i in range(35,53)]
V53_74_cols = ['V{}'.format(i) for i in range(53,75)]
V75_94_cols = ['V{}'.format(i) for i in range(75,95)]
V95_137_cols = ['V{}'.format(i) for i in range(95,138)]
V138_166_cols = ['V{}'.format(i) for i in range(138,167)]
V167_216_cols = ['V{}'.format(i) for i in range(167,217)]
V217_278_cols = ['V{}'.format(i) for i in range(217,279)]
V279_321_cols = ['V{}'.format(i) for i in range(279,322)]
V322_339_cols = ['V{}'.format(i) for i in range(322,340)]

for df in [train_df, test_df]:
    df['nulls'] = train_df.isnull().sum(axis=1)
    df['card_na'] = df[card_cols].isna().sum(axis=1).astype(np.int8)
    df['M_na'] = df[M_cols].isna().sum(axis=1).astype(np.int8)
    df['C_na'] = df[C_cols].isna().sum(axis=1).astype(np.int8)
    df['D_na'] = df[D_cols].isna().sum(axis=1).astype(np.int8)
    df['V1-11_na'] = df[V1_11_cols].isna().sum(axis=1)
    df['V12-34_na'] = df[V12_34_cols].isna().sum(axis=1)
    df['V35-52_na'] = df[V35_52_cols].isna().sum(axis=1)
    df['V53-74_na'] = df[V53_74_cols].isna().sum(axis=1)
    df['V75-94_na'] = df[V75_94_cols].isna().sum(axis=1)
    df['V95-137_na'] = df[V95_137_cols].isna().sum(axis=1)
    df['V138-166_na'] = df[V138_166_cols].isna().sum(axis=1)
    df['V167-216_na'] = df[V167_216_cols].isna().sum(axis=1)
    df['V217-278_na'] = df[V217_278_cols].isna().sum(axis=1)
    df['V279_321_na'] = df[V279_321_cols].isna().sum(axis=1)
    df['V322_339_na'] = df[V322_339_cols].isna().sum(axis=1)


# In[20]:


################## Check if the TransactionAmt is common or not
train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)


# In[21]:


################## Decimal part of TransactionAmt
for df in [train_df, test_df]:
    # df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
    # df['TransactionAmt_decimal_length'] = df['TransactionAmt'].astype(str).str.split('.', expand=True)[1].str.len()
    df['TransactionAmt_has_decimal'] = np.where((df['TransactionAmt'] - df['TransactionAmt'].astype(int))==0,0,1)


# In[22]:


################# log1p transformation of 'TransactionAmt'
train_df['TransactionAmt_log1p'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt_log1p'] = np.log1p(test_df['TransactionAmt']) 


# In[23]:


################### whether is December
for df in [train_df, test_df]:
    df['isDec']=np.where(df['month']==12,1,0)


# # Target mean

# In[24]:


#################### ProductCD and M4 Target mean
for col in ['ProductCD','M4','addr1']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)


# #  Combination labels of categorical features

# In[25]:


################# Create time labels
for df in [train_df, test_df]:
    df['D1-DT'] = df['D1'] - df['DT_D']
    df['D15-DT'] = df['D15'] - df['DT_D']


# In[26]:


################# Reset values for "noise" in 'card1'(真的需要？如果为了特定用户身份，就不需要将单条的用户聚类)
"""
i_cols = ['card1']

for col in i_cols: 
    valid_card = pd.concat([train_df[[col]], test_df[[col]]])
    valid_card = valid_card[col].value_counts()
    valid_card = valid_card[valid_card>2]
    valid_card = list(valid_card.index)

    train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
    test_df[col]  = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)

    train_df[col] = np.where(train_df[col].isin(valid_card), train_df[col], np.nan)
    test_df[col]  = np.where(test_df[col].isin(valid_card), test_df[col], np.nan)
"""


# In[27]:


######################### Create possible uid (to determine a specific acount or user)
for df in [train_df, test_df]:
    # card1,2
    df['uid1'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)
    # card1,2,3,5
    df['uid2'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)
    # card1,2,3,5 + addr1,2
    df['uid3'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
    # card1,2,3,5 + addr1,2 + email
    df['uid4'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)+'_'+    df['P_emaildomain'].astype(str)
    # card1,2,3,5 + addr1,2 + D1-DT
    df['uid5'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)+'_'+    df['D1-DT'].astype(str)
    


# # Feautures on TransactionAmt

# In[28]:


################### Statistical features of TransactionAmt by cardx and uidx
main_cols = ['TransactionAmt',]
cate_cols = ['card1', 'card2', 'card3', 'card5', 'uid1', 'uid2', 'uid3', 'uid4','uid5']
agg_types = ['mean', 'std', 'median']

train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False)


# In[29]:


################## Statistical features of TransactionAmt by time periods 
main_cols = ['TransactionAmt']
cate_cols = ['DT_M', 'DT_W', 'DT_D']
agg_types = ['mean', 'median', 'std']


train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=True)


# In[30]:


# Sliding windows: 前3,7,14,30天里的所有交易额的均值，中位数，方差（线下略微提升，线上略微降低, 是否保留暂定）
"""
for df in [train_df, test_df]:
    for offset in ['3d','7d','14d','30d']:
        rolling_obj = df[['DT','TransactionAmt']].set_index('DT').rolling(offset, min_periods = 1, closed = 'right')
        df[offset+'_TransAmt_mean'] = rolling_obj.mean().reset_index(drop=True)
        df[offset+'_TransAmt_median'] = rolling_obj.median().reset_index(drop=True)
        df[offset+'_TransAmt_std'] = rolling_obj.std().reset_index(drop=True)
        # df[offset+'_TransAmt_cnt'] = rolling_obj.count().reset_index(drop=True)
        # df[offset+'_TransAmt_diff_median'] = df['TransactionAmt'] - df[offset+'_TransAmt_median']
        # df[offset+'_TransAmt_zscore'] = (df['TransactionAmt'] - df[offset+'_TransAmt_mean']) / df[offset+'_TransAmt_std']
"""


# In[31]:


gc.collect()


# # Features on addr

# In[32]:


target_col = 'addr1'
refer_col = 'DT_D'

train_df[target_col] = train_df[target_col].fillna(999).astype(int).astype(str)
test_df[target_col] = test_df[target_col].fillna(999).astype(int).astype(str)
train_df, test_df = tfidf_encoding(train_df, test_df, target_col, refer_col)


# # Features on D cols

# In[33]:


D_cols = ['D'+str(i) for i in range(1,16)]

# (略有下降,配合D3的统计量有所上升)
for df in [train_df, test_df]:
    # D9: hour in a day
    df['D8_D9_decimal_dist'] = df['D8'].fillna(0)-df['D8'].fillna(0).astype(int)
    df['D8_D9_decimal_dist'] = ((df['D8_D9_decimal_dist']-df['D9'])**2)**0.5
    df['D8'] = df['D8'].fillna(-1).astype(int)
    
    df['D8>D1'] = np.where(df['D8']>df['D1'],1,0)
    
    df['D1_is_0'] = np.where(df['D1']==0,1,0)
    df['D3_is_0'] = np.where(df['D3']==0,1,0)

##################### Normalization on cumulative attributes
# （略有上升）
periods = ['DT_D']
D_cols.remove('D9')
D_cols.remove('D3')

for df in [train_df, test_df]:
    df = values_normalization(df, periods, D_cols)

##################### Statistical features on D3（略有上升）
main_cols = ['D3']
cate_cols = ['uid1', 'uid2', 'uid3', 'uid4','uid5']
agg_types = ['mean', 'median', 'std']

train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False)


# # Features on M cols

# In[34]:


#################### sum of T/F, sum of T, sum of F, and sum of nan
for df in [train_df, test_df]:
    df['M_sum_T'] = (df[M_cols]=='T').sum(axis=1).astype(np.int8)
    df['M_sum_F'] = (df[M_cols]=='F').sum(axis=1).astype(np.int8)
    df['M_sum'] = df['M_sum_F'] + df['M_sum_T']
    


# # Statistical features on C cols

# In[35]:


##################### Relaxation


# In[36]:


main_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14']
cate_cols = ['uid1', 'uid2', 'uid3', 'uid4','uid5',]
agg_types = ['mean', 'std']

train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=True)


# main_cols = ['C13', 'C14', 'C1','C2']
# cate_cols = ['DT_M', 'DT_W', 'DT_D']
# agg_types = ['mean', 'median', 'std']
# 
# train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False)

# # Statistical features on TransactionPerDay

# In[37]:


# （略有下降）
for df in [train_df, test_df]:
    df['TransactionAmtPerDay'] = df['TransactionAmt']/(df['D3']+1)


# In[38]:


# （有所上升）
main_cols = ['TransactionAmtPerDay']
cate_cols = ['uid1','uid2','uid3','uid4','uid5']
agg_types = ['mean', 'median', 'std']

train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False)


# In[39]:


del train_df['TransactionAmtPerDay'], test_df['TransactionAmtPerDay']


# #  ProductType

# In[40]:


# create bins
train_df['TransactionAmtBin'] = (train_df['TransactionAmt']/10).astype(int)
test_df['TransactionAmtBin'] = (test_df['TransactionAmt']/10).astype(int)
# create product_type on ProductCDxTransactionAmt
train_df['product_type'] = train_df['ProductCD'].astype(str)+'_'+train_df['TransactionAmt'].astype(str)
test_df['product_type'] = test_df['ProductCD'].astype(str)+'_'+test_df['TransactionAmt'].astype(str)


# # Email features

# In[41]:


email_dict = {
 'aim': "aol",
 'anonymous': "anon",
 'aol': "aol",
 'att': "att",
 'bellsouth': "other",
 'cableone': "other",
 'centurylink': "centurylink",
 'cfl': "other",
 'charter': "spectrum",
 'comcast': "comcast",
 'cox': "other",
 'earthlink': "other",
 'email_not_provided': 'email_not_provided',
 'embarqmail': "centurylink",
 'frontier': "yahoo",
 'frontiernet': "yahoo",
 'gmail': "google",
 'gmx': "other",
 'hotmail': "msft",
 'icloud': "apple",
 'juno': "other",
 'live': "msft",
 'mac': "apple",
 'mail': "other",
 'me': "apple",
 'msn': "msft",
 'netzero': "other",
 'optonline': "other",
 'outlook': "msft",
 'prodigy': "att",
 'protonmail': "proton",
 'ptd': "other",
 'q': "centurylink",
 'roadrunner': "other",
 'rocketmail': "yahoo",
 'sbcglobal': "att",
 'sc': "other",
 'scranton': "other",
 'servicios-ta': "other",
 'suddenlink': "other",
 'twc': "spectrum",
 'verizon': "other",
 'web': "other",
 'windstream': "other",
 'yahoo': "yahoo",
 'ymail': "yahoo"
}


# In[42]:


for df in [train_df, test_df]:
    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')
    
    df['email_check'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&(df['P_emaildomain']!='email_not_provided'),1,0)
    
    df['P_emaildomain_prefix'] = df['P_emaildomain'].apply(lambda s: s.split('.')[0])
    df['R_emaildomain_prefix'] = df['R_emaildomain'].apply(lambda s: s.split('.')[0])
    # 线下测试下降，但我感觉为了提高稳定性有必要，待定
    df['P_emaildomain_bin'] = df['P_emaildomain_prefix'].map(email_dict)
    df['R_emaildomain_bin'] = df['R_emaildomain_prefix'].map(email_dict)


# # Device

# In[43]:


train_id.head()


# In[44]:


for df in [train_id, test_id]:
    # df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    # df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    # df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric() or i=='.']))
    
    # id_30: OS
    df['id_30'] = df['id_30'].apply(lambda s : s.replace('_','.') if type(s)=='string' else s).fillna('unknown_os').str.lower()
    temp_OS_list = df['id_30'].apply(lambda s:s.replace('mac os x','mac').split())
    df['id_30_OS'] = temp_OS_list.apply(lambda s:s[0])
    df['id_30_version'] = temp_OS_list.apply(lambda s : '-1' if len(s)==1 else s[1])
    # df['id_30_version'] = temp_OS_list.apply(lambda s : ['-1'] if len(s)==1 else s[1].split('.'))
    # df['id_30_version1'] = df['id_30_version'].apply(lambda s : s[0])
    # df['id_30_version2'] = df['id_30_version'].apply(lambda s : s[1] if len(s)>1 else '-1')
    # df['id_30_version3'] = df['id_30_version'].apply(lambda s : s[2] if len(s)>2 else '-1')
    del temp_OS_list
    
    # id_31: browser
    df['id_31'] = df['id_31'].fillna('unknown_browser').str.lower()
    temp_browser_list = df['id_31'].replace('mobile safari','mobile_safari').replace('firefox mobile','firefox_mobile').                                replace(['samsung/sm-g531h','samsung/sm-g532m'],'samsung_old').map(lambda s: s.split())
    df['id_31_browser'] = df['id_31'].apply(lambda s: ''.join([i for i in s if i.isalpha()]))
    df['id_31_version'] = df['id_31'].apply(lambda s: ''.join([i for i in s if i.isnumeric() or i=='.'])).apply(lambda s : s)
    #df['id_31_version'] = df['id_31'].apply(lambda s: ''.join([i for i in s if i.isnumeric() or i=='.'])).apply(lambda s : s.split('.'))
    #df['id_31_version1'] = df['id_31_version'].apply(lambda s : s[0] if s[0] else '-1')
    #df['id_31_version2'] = df['id_31_version'].apply(lambda s : s[1] if len(s)>1 else '-1')
    del temp_browser_list
    
    # id_33:resolution
    df['id_33'].fillna('0', inplace =True)
    temp_res = df['id_33'].apply(lambda s : s.split('x'))
    df['ResVer'] = temp_res.apply(lambda s : 0 if s[0]=='0' else int(s[0])).astype('int16') # Vertical
    df['ResHor'] = temp_res.apply(lambda s : 0 if s[0]=='0' else int(s[1])).astype('int16') # Horizontal
    
    # DeviceInfo
    df['DeviceInfo_isna'] = np.where(df['DeviceInfo'].isna(), 1, 0)


# In[45]:


gc.collect()


# # Concatenate df and id

# In[46]:


temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_id, on='TransactionID', how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_id, on='TransactionID', how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)


# # Frequency features

# 'D1','D2','D3','D4','D5','D6','D7','D8',

# In[47]:


i_cols = ['card1','card2','card3','card4','card5','card6',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D3',
          'addr1','addr2',
          'dist1', 'dist2',
          'P_emaildomain', 'R_emaildomain',
          'DeviceInfo',
          'id_30','id_30_OS', 'id_30_version',
          'id_31', 'id_31_browser', 'id_31_version',
          'id_33',
          'uid1', 'uid2', 'uid3', 'uid4','uid5',
          'product_type',
         ]


# In[48]:


for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


# In[49]:


########################## The ratio of counts in time perios(TODO: 增加类别)
for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total']  = test_df[col].map(fq_encode)


i_cols = ['uid3','uid4','uid5','product_type','addr1',]
periods = ['DT_M','DT_W','DT_D']

for period in periods:
    for col in i_cols:
        new_column = col + '_rt_in_' + period
            
        temp_df = pd.concat([train_df[[col,period]], test_df[[col,period]]])
        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
        fq_encode = temp_df[new_column].value_counts().to_dict()
            
        train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
        test_df[new_column]  = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
        
        train_df[new_column] = train_df[new_column]/train_df[period+'_total']
        test_df[new_column]  = test_df[new_column]/test_df[period+'_total']


# In[50]:


gc.collect()


# In[51]:


train_df.shape, test_df.shape


# # PCA for V

# In[52]:


rm_V_cols = []


# In[53]:


# 11 parts
parts = []
parts.append(['V{}'.format(i) for i in range(1,12) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(12,35) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(35,53) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(53,75) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(75,95) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(95,138) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(138,167) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(167,217) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(217,279) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(279,322) if 'V{}'.format(i) not in rm_V_cols])
parts.append(['V{}'.format(i) for i in range(322,340) if 'V{}'.format(i) not in rm_V_cols])


# In[54]:


for i, part in enumerate(parts):
    temp_df = pd.concat([train_df[part], test_df[part]]).fillna(-1)
    pca = PCA(n_components = 3)
    pca.fit(temp_df)
    v_pca = pca.transform(temp_df)
    train_df = pd.concat([train_df, pd.DataFrame(v_pca[:train_num], columns=['V_part{}'.format(i)+'_0', 'V_part{}'.format(i)+'_1', 'V_part{}'.format(i)+'_2'])],axis=1)
    test_df = pd.concat([test_df, pd.DataFrame(v_pca[train_num:],columns=['V_part{}'.format(i)+'_0', 'V_part{}'.format(i)+'_1', 'V_part{}'.format(i)+'_2'])],axis=1)


# In[55]:


test_df.shape


# ## Removing list

# In[56]:


rm_cols = [
    'TransactionID', 'TransactionAmt', 'TransactionDT','TransactionAmt_decimal',
    'card1', 'card2',
    'uid1', 'uid2', 'uid3', 'uid4', 'uid5',
    'P_emaildomain', 'R_emaildomain', 'R_emaildomain_prefix', 'P_emaildomain_prefix',
    'day', 'month', 'hour', 'dayofweek',
    'DT', 'DT_M', 'DT_W', 'DT_D', 'DT_H',
    'DT_D_total', 'DT_W_total', 'DT_M_total',
    'id_30','id_31','id_33',
    'id_30_version', 'id_31_version',
    'DeviceInfo',
    'isFraud',
    'product_type',
    'TransactionAmtBin',
    'D15-DT', 'D1-DT',
    'addr1'
]


# In[57]:


################### Remove original V columns
rm_cols += ['V{}'.format(i) for i in range(0,340) if 'V{}'.format(i) not in rm_cols]


# ################### Remove columns with too many null values
# a = train_df.loc[:, train_df.isnull().sum(axis=0)/train_num > 0.95].keys()
# b = test_df.loc[:, test_df.isnull().sum(axis=0)/test_num > 0.95].keys()
# 
# rm_cols = rm_cols + list(a) + list(b)

# In[58]:


feature_cols = [col for col in train_df.keys() if col not in rm_cols]
print(feature_cols)


# # All the category features

# In[59]:


# M1-M9
tran_cols = []
for i in range(1,10):
    if 'M{}'.format(i) in feature_cols:
        tran_cols.append('M{}'.format(i))
        
# id_12-id_38
id_cols = []
for i in range(12,39):
    if 'id_{}'.format(i) in feature_cols:
        id_cols.append('id_{}'.format(i))
        
#for col in id_cols:
#    if train_df[col].nunique() > 300:
#        print(col+": ",train_df[col].nunique())
#        id_cols.remove(col)

cate_cols = ['card2', 'card3', 'card5', 'card4', 'card6', 'addr2', 'ProductCD', 'P_emaildomain_bin', 
             'P_emaildomain_bin', *tran_cols, *id_cols, 'id_30_OS', 'id_31_browser', 'DeviceType']

print('Categprical columns: ', cate_cols )


# In[60]:


# Fillna and label encoding
for col in cate_cols:
    #print(col)
    train_df[col].fillna('unknown', inplace=True)
    test_df[col].fillna('unknown', inplace=True)
    
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
    le = LabelEncoder()
    le.fit(list(train_df[col])+list(test_df[col]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


# In[61]:


# XGB can hardly deal with the high cardinality categorical features, so remove them
if train_df.shape[1] != test_df.shape[1]:
    test_df['isFraud'] = 0
    
train_num = train_df.shape[0]

df = pd.concat([train_df, test_df], axis = 0)


# In[62]:


new_cate_cols = []
for col in cate_cols:
    if df[col].nunique()>100:
        rm_cols.append(col)
    else:
        new_cate_cols.append(col)


# In[63]:


df = pd.get_dummies(df, columns=new_cate_cols)
train_df = df[:train_num]
test_df = df[train_num:]

del df


# In[ ]:


gc.collect()


# In[ ]:


feature_cols = [col for col in train_df.keys() if col not in rm_cols]
print(feature_cols)


# In[ ]:


len(feature_cols)


# In[ ]:


## ks test
if KS_TEST:
    feature_cols = set(feature_cols).difference(train_base_cols+rm_cols)
    list_p_value =[]

    for i in tqdm(feature_cols):
        list_p_value.append(ks_2samp(test_df[i] , train_df[i])[1])

    Se = pd.Series(list_p_value, index = feature_cols).sort_values() 
    list_discarded = list(Se[Se==0].index)

    print(list_discarded)

    feature_cols = [col for col in train_df.keys() if col not in rm_cols + list_discarded]
    cate_cols = [col for col in cate_cols if col not in rm_cols + list_discarded]


# # Training

# In[ ]:


if NEG_SAMPLE:
    # Negative downsampling
    train_pos = train_df[train_df['isFraud']==1]
    train_neg = train_df[train_df['isFraud']==0]

    train_neg = train_neg.sample(int(train_df.shape[0] * 0.2), random_state=SEED)
    train_df = pd.concat([train_pos,train_neg]).sort_index().reset_index(drop=True)

    labels = train_df[TARGET]
    
    train_num = train_df.shape[0]


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


param = {
    'booster': 'gbtree',
    'learning_rate': 0.01,
    'max_depth': 6,
    'subsample': 0.6,
    'colsample_bytree': 0.6,
    'objective': 'binary:logistic',
#     'n_estimators': 10000,
    'min_child_weight': 3,
    'gamma': 0,
#     'silent': True,
    'nthread': 10,
    'random_state': 4590,
    'reg_alpha': 0,
    'reg_lambda': 5,
#     'verbose': 200,
    'eval_metric': 'auc',
    'early_stopping_rounds': 100
}


# In[ ]:


if LOCAL_TEST:
    param['learning_rate'] = 0.01
    #param['n_estimators'] = 20000
    #param['early_stopping_rounds'] = 100
    test_predictions = make_predictions_kf(train_df, test_df, feature_cols,cate_cols, labels, param)
else:
    param['learning_rate'] = 0.005
    #param['n_estimators'] = 1800
    #param['early_stopping_rounds'] = 100
    # test_predictions = make_predictions_gkf(train_df, test_df, feature_cols, labels, lgb_param, NFOLDS=6)
    test_predictions = make_predictions_kf(train_df, test_df, feature_cols, cate_cols, labels, param, NFOLDS=10)
    test_predictions.to_csv(f"./sub/sub_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_.csv", index=False)


# In[ ]:




