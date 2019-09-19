#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import lightgbm as lgb
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
from scipy import sparse
from scipy.stats import kurtosis
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
        oof_pred_prob[valid_idx] =  valid_pred_prob
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


# In[9]:


def make_predictions_kf(train_df, test_df, feature_cols, target, param, NFOLDS=2):
    kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED)
    split_groups = train_df['DT_M']
    
    test_pred_prob = np.zeros(test_num)
    oof_pred_prob = np.zeros(train_num)
    
    train_values = train_df[feature_cols]
    test_values = test_df[feature_cols]
    labels = train_df['isFraud']
    
    if LOCAL_TEST:
        test_labels = test_df['isFraud']
    
    
    for i, (train_idx, valid_idx) in enumerate (kf.split(train_values, labels)):
        print(i,'fold...')
        start_time = time.time()
    
        train_x, train_y = train_values.iloc[train_idx], labels[train_idx]
        valid_x, valid_y = train_values.iloc[valid_idx], labels[valid_idx]
        
        if LOCAL_TEST:
            # Construct the dataset
            train_data = lgb.Dataset(train_x, label=train_y)
            valid_data = lgb.Dataset(valid_x, label=valid_y, reference = train_data)
            test_data = lgb.Dataset(test_values, label=test_labels, reference = train_data)
            
            # Training
            bst = lgb.train(param, train_data, valid_sets=[train_data, valid_data, test_data],verbose_eval=200)
            
            # Prediction
            valid_pred_prob = bst.predict(valid_x, num_iteration=bst.best_iteration)
            oof_pred_prob[valid_idx] =  valid_pred_prob
            # print('val logloss: ', log_loss(valid_y, valid_pred_prob))
            print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
            
            cur_test_pred_prob = bst.predict(test_values, num_iteration=bst.best_iteration)
            # print('val logloss: ', log_loss(test_labels, cur_test_pred_prob))
            print('current test auc: ', roc_auc_score(test_labels, cur_test_pred_prob))
            
            test_pred_prob += cur_test_pred_prob/kf.n_splits
            
            feature_imp = pd.DataFrame(sorted(zip(bst.feature_importance(),train_x.columns)), columns=['Value','Feature'])
            print(feature_imp)
            
        else:   
            # Construct the dataset
            train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols, free_raw_data = True)
            valid_data = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols, reference = train_data, free_raw_data = True)
    
            # Training
            bst = lgb.train(param, train_data, valid_sets=[train_data, valid_data],verbose_eval=200)
        
            # Prediction
            valid_pred_prob = bst.predict(valid_x, num_iteration=bst.best_iteration)
            oof_pred_prob[valid_idx] =  valid_pred_prob
            # print('val logloss: ', log_loss(valid_y, valid_pred_prob))
            print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
            
            test_pred_prob += bst.predict(test_values, num_iteration=bst.best_iteration)/kf.n_splits
         
        print('runtime: {}\n'.format(time.time() - start_time))
    
        # Plotting
        # lgb.plot_importance(bst,max_num_features=30)
    
    # print('oof logloss: ', log_loss(labels, oof_pred_prob))
    print('oof auc: ', roc_auc_score(labels, oof_pred_prob))
    
    if LOCAL_TEST:
        print('test auc: ', roc_auc_score(test_labels, test_pred_prob))
        test_df['pred_isFraud'] = test_pred_prob
        return test_df[['TransactionID','pred_isFraud']]
    
    else:
        test_df['isFraud'] = test_pred_prob
        return test_df[['TransactionID','isFraud']]


# # Global variables

# In[10]:


SEED = 42
seed_everything(SEED)
TARGET = 'isFraud'
START_DATE = datetime.datetime.strptime('2017-11-30', '%Y-%m-%d')
KS_TEST = False
LOCAL_TEST = False


# # Load data

# In[11]:


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
        # 只选择1,2,3月，因为12月太特殊，删除试试
        train_df = train_df[(train_df['DT_M'].min()<train_df['DT_M']) & (train_df['DT_M']<(train_df['DT_M'].max()))].reset_index(drop=True)
        
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


# In[ ]:


gc.collect()


# In[ ]:


print('Shape control: ', train_df.shape, test_df.shape, train_id.shape, test_id.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_base_cols = list(train_df)
test_base_cols = list(test_df)

labels = train_df[TARGET]

train_num = train_df.shape[0]
test_num = test_df.shape[0]

train_df['TransactionAmt'] = train_df['TransactionAmt'].astype(float)
test_df['TransactionAmt'] = test_df['TransactionAmt'].astype(float)


# ## . Check if the TransactionAmt is common or not

# In[ ]:


train_df['TransactionAmt_check'] = np.where(train_df['TransactionAmt'].isin(test_df['TransactionAmt']), 1, 0)
test_df['TransactionAmt_check']  = np.where(test_df['TransactionAmt'].isin(train_df['TransactionAmt']), 1, 0)


# ## . log1p transformation of 'TransactionAmt'

# In[ ]:


train_df['TransactionAmt_log1p'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt_log1p'] = np.log1p(test_df['TransactionAmt']) 


# # TransactionDT

# In[ ]:


train_df['DT'] = train_df['TransactionDT'].apply(lambda s:(START_DATE + datetime.timedelta(seconds = s)))
test_df['DT'] = test_df['TransactionDT'].apply(lambda s:(START_DATE + datetime.timedelta(seconds = s)))


# In[ ]:


for df in [train_df, test_df]:
    # total count of time periods
    df['DT_M'] = (df['DT'].dt.year-2017)*12 + df['DT'].dt.month
    df['DT_W'] = (df['DT'].dt.year-2017)*52 + df['DT'].dt.weekofyear
    df['DT_D'] = (df['DT'].dt.year-2017)*365 + df['DT'].dt.dayofyear
    df['DT_H'] = df['DT_D']*24 + df['DT'].dt.hour
    # datetime
    df['date'] = df['DT'].dt.date
    df['hour'] = df['DT'].dt.hour
    df['dayofweek'] = df['DT'].dt.dayofweek
    df['day'] = df['DT'].dt.day
    df['month'] = df['DT'].dt.month
    # D9 column
    df['D9'] = np.where(df['D9'].isna(),0,1)


# In[ ]:


train_df['month'].unique()


# In[ ]:


test_df['month'].unique()


# In[ ]:


train_df.head()


# ## . count of M1-M9

# In[ ]:


i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

# sum of T/F, sum of T, sum of F, and sum of nan
for df in [train_df, test_df]:
    df['M_sum_T'] = (df[i_cols]=='T').sum(axis=1).astype(np.int8)
    df['M_sum_F'] = (df[i_cols]=='F').sum(axis=1).astype(np.int8)
    df['M_sum'] = df['M_sum_F'] + df['M_sum_T']
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)


# # uid by card1-card6 & addr1, addr2 & P_emaildomain

# In[ ]:


########################### Reset values for "noise" in 'card1'
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


# In[ ]:


cols = ['card1','card2','card3','card4','card5','card6','addr1','addr2','P_emaildomain']

# fill nan in cols
for col in cols:
    train_df[col].fillna(0, inplace=True)
# In[ ]:


# Create possible uid by these features
for df in [train_df, test_df]:
    # card1,2
    df['uid1'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)
    # card1,2,3,5
    df['uid2'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)
    # card1,2,3,5 + addr1,2
    df['uid3'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)
    # card1,2,3,5 + addr1,2 + email
    df['uid4'] = df['card1'].astype(str)+'_'+df['card2'].astype(str)+'_'+df['card3'].astype(str)    +'_'+df['card5'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['addr2'].astype(str)+'_'+    df['P_emaildomain'].astype(str)


# In[ ]:


train_df[['uid1','uid2','uid3','uid4']].head()


# ## . Statistical features on TransactionAmt grouped by the attributes related with user
# 对与uid有关的各个类别做TransactionAmt的统计特征，包括均值、中位数、标准差、数量、差值特征、标准化特征

# In[ ]:


# Statistical features by categories
cate_cols = ['card1', 'card2', 'card3', 'card5', 'uid1', 'uid2', 'uid3', 'uid4']
agg_types = ['mean', 'std']
main_cols = ['TransactionAmt']

train_df, test_df = aggregation(train_df, test_df, main_cols, cate_cols, agg_types, with_diff=False, with_norm=False)


# In[ ]:


gc.collect()


# ## . ProductCD and M4 Target mean

# In[ ]:


for col in ['ProductCD','M4']:
    temp_dict = train_df.groupby([col])[TARGET].agg(['mean']).reset_index().rename(
                                                        columns={'mean': col+'_target_mean'})
    temp_dict.index = temp_dict[col].values
    temp_dict = temp_dict[col+'_target_mean'].to_dict()

    train_df[col+'_target_mean'] = train_df[col].map(temp_dict)
    test_df[col+'_target_mean']  = test_df[col].map(temp_dict)


# ## . Email features

# In[ ]:


for df in [train_df, test_df]:
    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')
    
    df['email_check'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&(df['P_emaildomain']!='email_not_provided'),1,0)
    
    df['P_emaildomain_prefix'] = df['R_emaildomain'].apply(lambda s: s.split('.')[0])
    df['R_emaildomain_prefix'] = df['R_emaildomain'].apply(lambda s: s.split('.')[0])


# In[ ]:


train_df['P_emaildomain_prefix'].unique()


# # Device

# In[ ]:


train_id.head()


# In[ ]:


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


# In[ ]:


gc.collect()


# # Concatenate df and id

# In[ ]:


temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_id, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_id, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)


# In[ ]:


train_df.tail(100)


# # Frequency features

# In[ ]:


i_cols = ['card1','card2','card3','card4','card5','card6',
          'C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8',
          'addr1','addr2',
          'dist1', 'dist2',
          'P_emaildomain', 'R_emaildomain',
          'DeviceInfo',
          'id_30','id_30_OS', 'id_30_version',
          'id_31', 'id_31_browser',
          'id_33',
          'uid1', 'uid2', 'uid3'
         ]


# In[ ]:


for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


# In[ ]:


for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total']  = test_df[col].map(fq_encode)


# In[ ]:


periods = ['DT_M','DT_W','DT_D']
i_cols = ['uid1','uid2','uid3','uid4']
for period in periods:
    for col in i_cols:
        new_column = col + '_' + period
            
        temp_df = pd.concat([train_df[[col,period]], test_df[[col,period]]])
        temp_df[new_column] = temp_df[col].astype(str) + '_' + (temp_df[period]).astype(str)
        fq_encode = temp_df[new_column].value_counts().to_dict()
            
        train_df[new_column] = (train_df[col].astype(str) + '_' + train_df[period].astype(str)).map(fq_encode)
        test_df[new_column]  = (test_df[col].astype(str) + '_' + test_df[period].astype(str)).map(fq_encode)
        
        train_df[new_column] /= train_df[period+'_total']
        test_df[new_column]  /= test_df[period+'_total']


# In[ ]:


gc.collect()


# In[ ]:


train_df.shape, test_df.shape


# # Delete useless columns

# In[ ]:


rm_cols = [
    'TransactionID', 'TransactionAmt', 'TransactionDT',
    'uid1', 'uid2', 'uid3', 'uid4',
    'date', 
    'day', 'month', 'hour', 'dayofweek',
    'DT', 'DT_M', 'DT_W', 'DT_D', 'DT_H',
    'DT_D_total', 'DT_W_total', 'DT_M_total',
    'id_30','id_31','id_33',
    'DeviceInfo',
    'isFraud',
]


# In[ ]:


feature_cols = [col for col in train_df.keys() if col not in rm_cols]


# In[ ]:


print(feature_cols)


# # All the category features

# In[ ]:


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

cate_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','P_emaildomain_prefix', 'R_emaildomain_prefix', *tran_cols, *id_cols, 'id_30_OS',             'id_31_browser', 'id_30_version', 'id_31_version', 'DeviceType']


# In[ ]:


# Fillna and label encoding
for col in cate_cols:
    train_df[col].fillna('unknown', inplace=True)
    test_df[col].fillna('unknown', inplace=True)
    
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
    le = LabelEncoder()
    le.fit(list(train_df[col])+list(test_df[col]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])
    
    train_df[col] = train_df[col].astype('category')
    test_df[col] = test_df[col].astype('category')


# In[ ]:


gc.collect()


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


# In[ ]:


print(train_df.shape, test_df.shape)


# In[ ]:


lgb_param = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'num_iterations': 10000,
    'learning_rate': 0.01,
    'num_leaves': 2**8,
    'num_threads': -1,
    'seed': SEED,
    'max_depth': -1,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'feature_fraction': 0.7,
    'early_stopping_round': 100,
    'metric': 'auc'
}
'''
    'min_data_in_leaf': 5,
    'min_sum_hessian_in_leaf': 4,
    'lambda_l1': 3,
    'lambda_l2': 5,
'''


# In[ ]:


if LOCAL_TEST:
    lgb_param['learning_rate'] = 0.01
    lgb_param['n_estimators'] = 20000
    lgb_param['early_stopping_rounds'] = 100
    test_predictions = make_predictions_kf(train_df, test_df, feature_cols, labels, lgb_param)
else:
    lgb_param['learning_rate'] = 0.005
    lgb_param['n_estimators'] = 1800
    lgb_param['early_stopping_rounds'] = 100
    # test_predictions = make_predictions_gkf(train_df, test_df, feature_cols, labels, lgb_param, NFOLDS=6)
    test_predictions = make_predictions_kf(train_df, test_df, feature_cols, labels, lgb_param, NFOLDS=10)
    test_predictions.to_csv(f"./sub/sub_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_.csv", index=False)


# In[ ]:




