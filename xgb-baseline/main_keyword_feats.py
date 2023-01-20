#%%

# -*- coding: utf-8 -*-


import warnings

warnings.simplefilter('ignore')

import os
import gc
import re
import glob

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
from tqdm.auto import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, auc, f1_score

from urllib.parse import quote, unquote, urlparse
from xgboost import XGBClassifier

#%%


#%%

import random


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化


set_seed(2022)

#%%

# train

train_files = glob.glob('../data/train/*.csv')

df_train = pd.DataFrame()

for filepath in tqdm(train_files):
    df = pd.read_csv(filepath)
    df_train = pd.concat([df_train, df]).reset_index(drop=True)

df_train.fillna('__NaN__', inplace=True)

# 强迫症发作..
df_train = df_train.rename(columns={'lable': 'label'})
df_train
print(len(df_train))

#%%

# label
# 0. 白
# 1. SQL 注入
# 2. 目录历遍
# 3. 远程代码执行
# 4. 命令执行
# 5. XSS 跨站脚本

# %%

df_test = pd.read_csv('../data/test/test.csv')
df_test.fillna('__NaN__', inplace=True)
df_test

# %%

df = pd.concat([df_train, df_test]).reset_index(drop=True)
df.shape

#%%
print('-------------------提取body, url的len特征 start...------------------------')
#df['body_len'] = df['body'].apply(lambda x: len(list(x)))
#df['url_len'] = df['url'].apply(lambda x: len(list(x)))
print('-------------------提取body. url的len特征 end...------------------------')
#%%
#%%
#%%
print('-------------------提取攻击关键词特征 start...------------------------')
XSS_key_words_high = ['<scrip', '</script', '<iframe', '</iframe', 'response',
                 'write(','eval(','prompt(','alert(','javascript;','document','cookie']
XSS_key_words_middle = ['onclick=','onerror=','<!--','-->','<base','</base>>','location','hash','window','name','<form','</form']
XSS_key_words_low = ['echo','print','href=','sleep']
XSS_key_words = XSS_key_words_high + XSS_key_words_middle + XSS_key_words_low

SQL_key_words_high = ['and','or','xp_','substr','utl_','benchmark','shutdown','@@version','mformation_schema','hex(']
SQL_key_words_middle = ['select','if(','union','group','by','--','count(','/**/','char(','drop','delete','concat','orderby',
                        'case when','assic(','exec(','length']
SQL_key_words_low = ['and','or','like','from','insert', 'update','create','else', 'exist','table' ,'database','where','sleep','mid',
                    'updatexml(','null','sqlmap','md5(','floorm','rand','cast','dual','fetch','print','declare','cursor',
                    'extractvalue(','upperjoin','exec','innier','convert','distinct']
SQL_key_words = SQL_key_words_high+SQL_key_words_middle+SQL_key_words_low
#%%

#%%
def Have_XSS_attack_feat(x):
    for kw in XSS_key_words:
        if kw in x:
            return 1
    return 0

def Nums_XSS_attack_feat(x):
    count = 0
    for kw in XSS_key_words:
        if kw in x:
            count += 1
    return count


#df['Have_XSS_attack_feat_body'] = df['body'].apply(lambda x: Have_XSS_attack_feat(x))
df['Nums_XSS_attack_feat_body'] = df['body'].apply(lambda x: Nums_XSS_attack_feat(x))

#%%
def Have_SQL_attack_feat(x):
    for kw in SQL_key_words:
        if kw in x:
            return 1
    return 0

def Nums_SQL_attack_feat(x):
    count = 0
    for kw in SQL_key_words:
        if kw in x:
            count += 1
    return count


#df['Have_SQL_attack_feat_body'] = df['body'].apply(lambda x: Have_SQL_attack_feat(x))
df['Nums_SQL_attack_feat_body'] = df['body'].apply(lambda x: Nums_SQL_attack_feat(x))

print('-------------------提取攻击关键词特征 end...------------------------')
#%%
print('-------------------提取特殊字符特征 start...------------------------')
special_chars = ['\\', '/', '!', '%', '#', '&', '?', '+', '-', '*', '=', '{', '}', '(', ')',
                '[', ']', '@', '^', '$', '<', '>', ':', ';', '~']

def Nums_special_char_feat(x):
    count = 0
    for kw in special_chars:
        if kw in x:
            count += 1
    return count
#%%
df['Nums_special_char_feat_body'] = df['body'].apply(lambda x: Nums_special_char_feat(x))
df['Nums_special_char_feat_url'] = df['url'].apply(lambda x: Nums_special_char_feat(x))
#%%
print('-------------------提取特殊字符特征 end...------------------------')
#%%
#%%
#%%
#%%
#%%

#%%

def get_url_query(s):
    li = re.split('[=&]', urlparse(s)[4])
    return [li[i] for i in range(len(li)) if i % 2 == 1]


def find_max_str_length(x):
    max_ = 0
    li = [len(i) for i in x]
    return max(li) if len(li) > 0 else 0


def find_str_length_std(x):
    max_ = 0
    li = [len(i) for i in x]
    return np.std(li) if len(li) > 0 else -1


df['url_unquote'] = df['url'].apply(unquote)
df['url_query'] = df['url_unquote'].apply(lambda x: get_url_query(x))
df['url_query_num'] = df['url_query'].apply(len)
df['url_query_max_len'] = df['url_query'].apply(find_max_str_length)
df['url_query_len_std'] = df['url_query'].apply(find_str_length_std)


# %%

def find_url_filetype(x):
    try:
        return re.search(r'\.[a-z]+', x).group()
    except:
        return '__NaN__'


df['url_path'] = df['url_unquote'].apply(lambda x: urlparse(x)[2])
df['url_filetype'] = df['url_path'].apply(lambda x: find_url_filetype(x))

df['url_path_len'] = df['url_path'].apply(len)
df['url_path_num'] = df['url_path'].apply(lambda x: len(re.findall('/', x)))

# %%

df['ua_short'] = df['user_agent'].apply(lambda x: x.split('/')[0])
df['ua_first'] = df['user_agent'].apply(lambda x: x.split(' ')[0])

# %%

# % % time


def add_tfidf_feats(df, col, n_components=16):
    text = list(df[col].values)
    tf = TfidfVectorizer(min_df=1,
                         analyzer='char_wb',
                         ngram_range=(1, 3),
                         stop_words='english')
    tf.fit(text)
    X = tf.transform(text)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    X_svd = svd.transform(X)
    for i in range(n_components):
        df[f'{col}_tfidf_{i}'] = X_svd[:, i]
    return df

def add_tfidf_feats_word(df, col, n_components=16):
    text = list(df[col].values)
    tf = TfidfVectorizer(min_df=1,
                         analyzer='word',
                         ngram_range=(1, 3),
                         stop_words='english')
    tf.fit(text)
    X = tf.transform(text)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    X_svd = svd.transform(X)
    for i in range(n_components):
        df[f'{col}_tfidf_word_{i}'] = X_svd[:, i]
    return df


def add_tfidf_feats_char(df, col, n_components=16):
    text = list(df[col].values)
    tf = TfidfVectorizer(min_df=1,
                         analyzer='char',
                         ngram_range=(1, 3),
                         stop_words='english')
    tf.fit(text)
    X = tf.transform(text)
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(X)
    X_svd = svd.transform(X)
    for i in range(n_components):
        df[f'{col}_tfidf_char_{i}'] = X_svd[:, i]
    return df

df = add_tfidf_feats(df, 'url_unquote', n_components=256)
df = add_tfidf_feats(df, 'user_agent', n_components=256)
df = add_tfidf_feats(df, 'body', n_components=256)


df = add_tfidf_feats_word(df, 'url_unquote', n_components=256)
df = add_tfidf_feats_word(df, 'user_agent', n_components=256)
df = add_tfidf_feats_word(df, 'body', n_components=256)

df = add_tfidf_feats_char(df, 'url_unquote', n_components=256)
df = add_tfidf_feats_char(df, 'user_agent', n_components=256)
df = add_tfidf_feats_char(df, 'body', n_components=256)



# %%

for col in tqdm(['method', 'refer', 'url_filetype', 'ua_short', 'ua_first']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# %%

not_use_feats = ['id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path', 'label']
use_features = [col for col in df.columns if col not in not_use_feats]

# %%

train = df[df['label'].notna()]
test = df[df['label'].isna()]

train.shape, test.shape

# %%

NUM_CLASSES = 6
FOLDS = 5
TARGET = 'label'

from sklearn.preprocessing import label_binarize


def run_lgb(df_train, df_test, use_features):
    target = TARGET
    oof_pred = np.zeros((len(df_train), NUM_CLASSES))
    y_pred = np.zeros((len(df_test), NUM_CLASSES))

    folds = StratifiedKFold(n_splits=FOLDS)
    for fold, (tr_ind, val_ind) in enumerate(folds.split(train, train[TARGET])):
        print(f'Fold {fold + 1}')
        x_train, x_val = df_train[use_features].iloc[tr_ind], df_train[use_features].iloc[val_ind]
        y_train, y_val = df_train[target].iloc[tr_ind], df_train[target].iloc[val_ind]


        params = {'random_seed': 2022}
        model = XGBClassifier(**params)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)],
                  verbose=10, eval_metric='auc')



        oof_pred[val_ind] = model.predict_proba(x_val)
        y_pred += model.predict_proba(df_test[use_features]) / folds.n_splits

        #print("Features importance...")
        #gain = model.feature_importance('gain')
        #feat_imp = pd.DataFrame({'feature': model.feature_name(),
                                 #'split': model.feature_importance('split'),
                                 #'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
        #print('Top 50 features:\n', feat_imp.head(50))

        del x_train, x_val, y_train, y_val
        gc.collect()

    return y_pred, oof_pred


y_pred, oof_pred = run_lgb(train, test, use_features)

# %%

print('acc: ', accuracy_score(np.argmax(oof_pred, axis=1), df_train['label']))

# %%

sub = pd.read_csv('../data/submit_example.csv')
sub['predict'] = np.argmax(y_pred, axis=1)
sub

# %%

sub['predict'].value_counts()

# %%

sub.to_csv('main_keyword_feats.csv', index=False)
print('f1: ', f1_score(np.argmax(oof_pred, axis=1), df_train['label'], average='macro'))

