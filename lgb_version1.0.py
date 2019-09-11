#!/usr/bin/env python
# coding: utf-8

# In[3]:


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
from scipy import sparse
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from scipy import sparse
from scipy.stats import kurtosis
# import cufflinks as cf
# from IPython.display import display,HTML
# from plotly.offline import init_notebook_mode
# cf.go_offline()
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline
# cf.set_config_file(theme='ggplot',sharing='public',offline=True)
# init_notebook_mode(connected=False)  
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

np.set_printoptions(suppress=True)
pd.set_option('precision', 5)
pd.set_option('display.float_format', lambda x: '%.5f' % x) #为了直观的显示数字，不采用科学计数法


# # Load data

# In[4]:


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


# In[5]:


def load_data(file):
    if (file.split('.'))[-1] == 'csv':
        return reduce_mem_usage(pd.read_csv(file))
    elif (file.split('.'))[-1] == 'pkl':
        return pd.read_pickle(file)
    else:
        raise IOError("Error: unknown file type: "+file)


# In[6]:


files=['data/train_transaction.pkl',
       'data/test_transaction.pkl',
       'data/train_identity.pkl',
       'data/test_identity.pkl']


# In[7]:


with multiprocessing.Pool() as pool:
    print("Loading...")
    try:
        train_df, test_df, train_id, test_id = pool.map(load_data, files)
    except IOError as er:
        print (er)
    else:
        print ("Loading done")


# In[8]:


gc.collect()


# In[ ]:


# with open('./data/train_transaction.pkl', 'wb') as f:
#     pickle.dump(train_df, f)
# with open('./data/test_transaction.pkl', 'wb') as f:
#     pickle.dump(test_df, f)
# with open('./data/train_identity.pkl', 'wb') as f:
#     pickle.dump(train_id, f)
# with open('./data/test_identity.pkl', 'wb') as f:
#     pickle.dump(test_id, f)


# In[ ]:


# with open('./data/train.pkl', 'rb') as f:
#     train = pickle.load(f)
# with open('./data/test.pkl', 'rb') as f:
#     test = pickle.load(f)


# In[9]:


print('Shape control: ', train_df.shape, test_df.shape, train_id.shape, test_id.shape)


# In[10]:


train_df.head()


# In[11]:


labels = train_df['isFraud']
sub = test_df[['TransactionID']]


# In[12]:


train_num = train_df.shape[0]
test_num = test_df.shape[0]


# In[13]:


train_df['TransactionAmt'] = train_df['TransactionAmt'].astype(float)
test_df['TransactionAmt'] = test_df['TransactionAmt'].astype(float)


# # TransactionDT

# In[14]:


train_df['DT'] = train_df['TransactionDT'].apply(lambda s:(datetime.datetime.strptime('2017-11-30', '%Y-%m-%d') + datetime.timedelta(seconds = s)))
test_df['DT'] = test_df['TransactionDT'].apply(lambda s:(datetime.datetime.strptime('2017-11-30', '%Y-%m-%d') + datetime.timedelta(seconds = s)))


# In[15]:


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
    # D9
    df['D9'] = np.where(df['D9'].isna(),0,1)


# # Time series features

# In[19]:


# Statistical features by time periods
cols = ['DT_M', 'DT_W', 'DT_D']
agg_types = ['mean', 'std', 'median', 'max', 'min']

for col in cols:
    new_col_names = [col + '_TransactionAmt_' + agg_type for agg_type in agg_types]
    temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
    temp_df = temp_df.groupby([col])['TransactionAmt'].agg(dict(zip(new_col_names, agg_types)))
    
    for new_col_name in new_col_names:
        train_df[new_col_name] = train_df[col].map(temp_df[new_col_name])
        test_df[new_col_name] = test_df[col].map(temp_df[new_col_name])
    
    for df in [train_df, test_df]:
        df[col + '_TransactionAmt_diff_median'] = df['TransactionAmt'] - df[col + '_TransactionAmt_median']
        df[col + '_TransactionAmt_diff_mean'] = df['TransactionAmt'] - df[col + '_TransactionAmt_mean']
        df[col + '_TransactionAmt_zscore'] = df[col + '_TransactionAmt_diff_mean'] / df[col + '_TransactionAmt_std']


# In[20]:


# Sliding windows on days
temp_df = pd.concat([train_df[['DT_D', 'TransactionAmt']], test_df[['DT_D', 'TransactionAmt']]])

def get_sliding_mean_by_days(df, day, delta_days):
    start_day = day - delta_days
    end_day = day
    return df[np.array(df['DT_D']>=start_day) & np.array(df['DT_D']<end_day)]['TransactionAmt'].mean()

def get_sliding_cnt_by_days(df, day, delta_days):
    start_day = day - delta_days
    end_day = day
    return df[np.array(df['DT_D']>=start_day) & np.array(df['DT_D']<end_day)]['TransactionAmt'].count()

for df in [train_df, test_df]:
    DT_D_temp = df['DT_D'].unique()
    for delta in [1,3,7,14,30]:
        DT_D_mean_temp = [get_sliding_mean_by_days(temp_df, day, delta) for day in DT_D_temp]
        df[str(delta)+'_day_TransAmt_mean'] = df['DT_D'].map(dict(zip(DT_D_temp, DT_D_mean_temp)))
        DT_D_cnt_temp = [get_sliding_cnt_by_days(temp_df, day, delta) for day in DT_D_temp]
        df[str(delta)+'_day_TransAmt_cnt'] = df['DT_D'].map(dict(zip(DT_D_temp, DT_D_cnt_temp)))
        df[str(delta)+'_day_TransAmt_diff'] = df['TransactionAmt'] - df[str(delta)+'_day_TransAmt_mean']

del temp_df


# In[21]:


train_df[['DT_D', '1_day_TransAmt_mean', '1_day_TransAmt_cnt', '1_day_TransAmt_diff']].tail()


# In[22]:


gc.collect()


# In[23]:


# Sliding windows on hours
temp_df = pd.concat([train_df[['DT_H', 'TransactionAmt']], test_df[['DT_H', 'TransactionAmt']]])

def get_sliding_mean_by_hours(df, hour, delta_hours):
    start_hour = hour - delta_hours
    end_hour = hour
    return df[np.array(df['DT_H']>=start_hour) & np.array(df['DT_H']<end_hour)]['TransactionAmt'].mean()

def get_sliding_cnt_by_hours(df, hour, delta_hours):
    start_hour = hour - delta_hours
    end_hour = hour
    return df[np.array(df['DT_H']>=start_hour) & np.array(df['DT_H']<end_hour)]['TransactionAmt'].count()

for df in [train_df, test_df]:
    DT_H_temp = df['DT_H'].unique()
    for delta in [3,6,12,24]:
        DT_H_mean_temp = [get_sliding_mean_by_hours(temp_df, hour, delta) for hour in DT_H_temp]
        df[str(delta)+'_hour_TransAmt_mean'] = df['DT_H'].map(dict(zip(DT_H_temp, DT_H_mean_temp)))
        DT_H_cnt_temp = [get_sliding_cnt_by_hours(temp_df, hour, delta) for hour in DT_H_temp]
        df[str(delta)+'_hour_TransAmt_cnt'] = df['DT_H'].map(dict(zip(DT_D_temp, DT_D_cnt_temp)))
        df[str(delta)+'_hour_TransAmt_diff'] = df['TransactionAmt'] - df[str(delta)+'_hour_TransAmt_mean']

del temp_df


# In[24]:


gc.collect()


# # M1-M9

# In[25]:


i_cols = ['M1','M2','M3','M5','M6','M7','M8','M9']

# sum of T/F, sum of T, sum of F, and sum of nan
for df in [train_df, test_df]:
    df['M_sum_T'] = (df[i_cols]=='T').sum(axis=1).astype(np.int8)
    df['M_sum_F'] = (df[i_cols]=='F').sum(axis=1).astype(np.int8)
    df['M_sum'] = df['M_sum_F'] + df['M_sum_T']
    df['M_na'] = df[i_cols].isna().sum(axis=1).astype(np.int8)


# In[26]:


train_df.head()


# # Features with card1-card6 & addr1, addr2

# In[27]:


cols = ['card1','card2','card3','card4','card5','card6','addr1','addr2']


# In[28]:


for col in cols:
    # Delete the attributes only appearing in either training set or test set
    train_df[col] = np.where(train_df[col].isin(test_df[col]), train_df[col], np.nan)
    test_df[col] = np.where(test_df[col].isin(train_df[col]), test_df[col], np.nan)
    
    # Count the attributes
    cnt = pd.concat([train_df[col],test_df[col]]).value_counts(dropna=False).to_dict()
    train_df[col + '_cnt'] = train_df[col].map(cnt)
    test_df[col + '_cnt'] = test_df[col].map(cnt)
    
    # Replace the cases appearing less than 3 times with '0'
    train_df[train_df[col + '_cnt']<3][col] = 0
    test_df[test_df[col + '_cnt']<3][col] = 0


# In[29]:


# 2-level and 3-level feature crosses
uids = []
i = 0

# 2-level
for comb in list(itertools.combinations([['card1','card2','card3'],['card4'],['card5'],['card6'],['addr1','addr2']], 2)):
    for pair in list(itertools.product(*comb)):
        for df in [train_df, test_df]:
            df['uid'+str(i)] = df[pair[0]].astype(str) + '_' + df[pair[1]].astype(str)
        uids.append('uid'+str(i))
        i += 1
        
# 3-level
for comb in list(itertools.combinations([['card1','card2','card3'],['card4'],['card5'],['card6'],['addr1','addr2']], 3)):
    for pair in list(itertools.product(*comb)):
        for df in [train_df, test_df]:
            df['uid'+str(i)] = df[pair[0]].astype(str) + '_' + df[pair[1]].astype(str) + '_' + df[pair[2]].astype(str)
        uids.append('uid'+str(i))
        i += 1

print('uid0 - uid{} are created'.format(i-1))
del i


# In[30]:


gc.collect()


# In[31]:


# Statistical features by categories
i_cols = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', *uids]
agg_types = ['mean', 'std', 'median', 'max', 'min']

for col in i_cols:
    new_col_names = [col + '_TransactionAmt_' + agg_type for agg_type in agg_types]
    temp_df = pd.concat([train_df[[col, 'TransactionAmt']], test_df[[col, 'TransactionAmt']]])
    temp_df = temp_df.groupby([col])['TransactionAmt'].agg(dict(zip(new_col_names, agg_types)))
    
    for new_col_name in new_col_names:
        train_df[new_col_name] = train_df[col].map(temp_df[new_col_name])
        test_df[new_col_name] = test_df[col].map(temp_df[new_col_name])
        
    train_df[col + '_TransactionAmt_diff'] = train_df['TransactionAmt'] - train_df[col + '_TransactionAmt_mean']
    test_df[col + '_TransactionAmt_diff'] = test_df['TransactionAmt'] - test_df[col + '_TransactionAmt_mean']


# In[32]:


gc.collect()


# # log1p transformation of 'TransactionAmt'

# In[33]:


train_df['TransactionAmt_log1p'] = np.log1p(train_df['TransactionAmt'])
test_df['TransactionAmt_log1p'] = np.log1p(test_df['TransactionAmt']) 


# In[34]:


train_df.head()


# # Email features

# In[35]:


for df in [train_df, test_df]:
    df['P_emaildomain'] = df['P_emaildomain'].fillna('email_not_provided')
    df['R_emaildomain'] = df['R_emaildomain'].fillna('email_not_provided')
    
    df['email_check'] = np.where((df['P_emaildomain']==df['R_emaildomain'])&(df['P_emaildomain']!='email_not_provided'),1,0)
    
    df['P_emaildomain_prefix'] = df['R_emaildomain'].apply(lambda s: s.split('.')[0])
    df['R_emaildomain_prefix'] = df['R_emaildomain'].apply(lambda s: s.split('.')[0])


# # Device

# In[36]:


train_id.head()


# In[37]:


for df in [train_id, test_id]:
    # df['DeviceInfo'] = df['DeviceInfo'].fillna('unknown_device').str.lower()
    # df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isalpha()]))
    # df['DeviceInfo_device'] = df['DeviceInfo'].apply(lambda x: ''.join([i for i in x if i.isnumeric() or i=='.']))
    
    # id_30: OS
    df['id_30'] = df['id_30'].replace('_','.').fillna('unknown_os').str.lower()
    temp_OS_list = df['id_30'].apply(lambda s:s.replace('mac os x','mac').split())
    df['id_30_OS'] = temp_OS_list.apply(lambda s:s[0])
    df['id_30_version'] = temp_OS_list.apply(lambda s : 'unknown_version' if len(s)==1 else s[1])
    del temp_OS_list
    
    # id_31: browser
    df['id_31'] = df['id_31'].fillna('unknown_browser').str.lower()
    temp_browser_list = df['id_31'].replace('mobile safari','mobile_safari').replace('firefox mobile','firefox_mobile').                                replace(['samsung/sm-g531h','samsung/sm-g532m'],'samsung_old').map(lambda s: s.split())
    df['id_31_browser'] = df['id_31'].apply(lambda s: ''.join([i for i in s if i.isalpha()]))
    df['id_31_version'] = df['id_31'].apply(lambda s: ''.join([i for i in s if i.isnumeric() or i=='.']))
    
    # id_33:resolution
    df['id_33'].fillna('0', inplace =True)
    temp_res = df['id_33'].apply(lambda s : s.split('x'))
    df['ResVer'] = temp_res.apply(lambda s : 0 if s[0]=='0' else int(s[0])).astype('int16') # Vertical
    df['ResHor'] = temp_res.apply(lambda s : 0 if s[0]=='0' else int(s[1])).astype('int16') # Horizontal


# In[38]:


gc.collect()


# # Concatenate df and id

# In[39]:


temp_df = train_df[['TransactionID']]
temp_df = temp_df.merge(train_id, on=['TransactionID'], how='left')
del temp_df['TransactionID']
train_df = pd.concat([train_df,temp_df], axis=1)
    
temp_df = test_df[['TransactionID']]
temp_df = temp_df.merge(test_id, on=['TransactionID'], how='left')
del temp_df['TransactionID']
test_df = pd.concat([test_df,temp_df], axis=1)


# # Frequency features

# In[40]:


i_cols = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14',
          'D1','D2','D3','D4','D5','D6','D7','D8',
          'dist1', 'dist2',
          'P_emaildomain', 'R_emaildomain',
          'DeviceInfo',
          'id_30','id_30_OS','id_30_version',
          'id_31', 'id_31_browser','id_31_version',
          'id_33'
         ]


# In[41]:


for col in i_cols:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts(dropna=False).to_dict()   
    train_df[col+'_fq_enc'] = train_df[col].map(fq_encode)
    test_df[col+'_fq_enc']  = test_df[col].map(fq_encode)


# In[42]:


for col in ['DT_M','DT_W','DT_D']:
    temp_df = pd.concat([train_df[[col]], test_df[[col]]])
    fq_encode = temp_df[col].value_counts().to_dict()
            
    train_df[col+'_total'] = train_df[col].map(fq_encode)
    test_df[col+'_total']  = test_df[col].map(fq_encode)


# In[43]:


periods = ['DT_M','DT_W','DT_D']
i_cols = ['card1','card2','card3','addr2']
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


# In[47]:


gc.collect()


# In[45]:


train_df.shape, test_df.shape


# # Dump the train_df and test_df

# In[46]:


with open('./data/train.pkl', 'wb') as f:
    pickle.dump(train_df, f)
with open('./data/test.pkl', 'wb') as f:
    pickle.dump(test_df, f)


# # Delete useless columns

# In[48]:


rm_cols = [
    'TransactionID', 'TransactionDT',
    'addr1', 'addr2',
    'card1', 'card2', 'card3', 'card5',
    *uids,
    'date',
    'day', 'hour', 'month',
    'DT', 'DT_M', 'DT_W', 'DT_D', 'DT_H',
    'DT_D_total', 'DT_W_total', 'DT_M_total',
    'DeviceInfo',
    'isFraud'
]


# In[49]:


# id_12-id_38
id_cols = []
for i in range(12,39):
    if 'id_{}'.format(i) in train_df.keys():
        id_cols.append('id_{}'.format(i))
        
# Delete catagorical features of high cardinality 
for col in id_cols:
    if train_df[col].nunique() > 54:
        print(col+": ",train_df[col].nunique())
        rm_cols.append(col)


# In[50]:


print(rm_cols)


# In[51]:


feature_cols = [col for col in train_df.keys() if col not in rm_cols]


# # All the category features

# In[52]:


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

cate_cols = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','P_emaildomain_prefix', 'R_emaildomain_prefix', *tran_cols, *id_cols, 'id_30_OS',              'id_30_version', 'id_31_browser', 'id_31_version', 'DeviceType', 'dayofweek']


# In[53]:


# Fillna and label encoding
for col in cate_cols:
    train_df[col].fillna('unkown', inplace=True)
    test_df[col].fillna('unkown', inplace=True)
    
    train_df[col] = train_df[col].astype(str)
    test_df[col] = test_df[col].astype(str)
    
    le = LabelEncoder()
    le.fit(list(train_df[col])+list(test_df[col]))
    train_df[col] = le.transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


# In[54]:


gc.collect()


# In[55]:


train_df.shape, test_df.shape


# In[56]:


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
test_pred_prob = np.zeros(test_num)
oof_pred_prob = np.zeros(train_num)


# In[57]:


param = {
    'boosting': 'gbdt',
    'objective': 'binary',
    'num_iterations': 400,
    'learning_rate': 0.1,
    'num_leaves': 2**8,
    'num_threads': 4,
    'seed': 2019,
    'max_depth': -1,
    'min_data_in_leaf': 5,
    'min_sum_hessain_in_leaf': 4,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'feature_fraction': 0.5,
    'lambda_l1': 3,
    'lambda_l2': 5,
    'min_gain_to_split': 0,
    'early_stopping_round': 50,
    'metric': 'auc'
}


# In[ ]:


train_values = train_df[feature_cols]
test_values = test_df[feature_cols]
labels = train_df['isFraud']

for i, (train_idx, valid_idx) in enumerate (skf.split(train_values, labels)):
    print(i,'fold...')
    start_time = time.time()
    
    train_x, train_y = train_values.iloc[train_idx], labels[train_idx]
    valid_x, valid_y = train_values.iloc[valid_idx], labels[valid_idx]
    
    # Construct the dataset
    train_data = lgb.Dataset(train_x, label=train_y, categorical_feature=cate_cols, free_raw_data = True)
    valid_data = lgb.Dataset(valid_x, label=valid_y, categorical_feature=cate_cols, reference = train_data, free_raw_data = True)
    
    # Training
    bst = lgb.train(param, train_data, valid_sets=[train_data, valid_data],verbose_eval=20)
    
    # Prediction
    valid_pred_prob = bst.predict(valid_x, num_iteration=bst.best_iteration)
    oof_pred_prob[valid_idx] =  valid_pred_prob
    print('val logloss: ', log_loss(valid_y, valid_pred_prob))
    print('val auc: ', roc_auc_score(valid_y, valid_pred_prob))
    
    test_pred_prob += bst.predict(test_values, num_iteration=bst.best_iteration)/skf.n_splits
     
    print('runtime: {}\n'.format(time.time() - start_time))
    
    # Plotting
    lgb.plot_importance(bst,max_num_features=20)


# In[46]:


print('oof logloss: ', log_loss(labels, oof_pred_prob))
print('oof auc: ', roc_auc_score(labels, oof_pred_prob))


# In[47]:


test_pred_prob.size


# In[58]:


sub = pd.read_csv('./sub/sub_2019-09-10 01-57-18.csv')


# In[59]:


sub['isFraud'] = 1-sub['isFraud']
sub.to_csv(f"./sub/sub_{datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv", index=False)


# In[ ]:




