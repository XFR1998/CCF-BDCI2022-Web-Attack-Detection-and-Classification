# %%
# -*- coding: utf-8 -*-

# %%
import warnings

warnings.simplefilter('ignore')

import os
import gc
import re
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup, logging
# from torch.utils.data import TensorDataset,SequentialSampler,RandomSampler,DataLoader
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from urllib.parse import quote, unquote, urlparse
# %%

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.manual_seed(seed)


setup_seed(2022)
# %%

# train

train_files = glob.glob('../data/train/*.csv')

df_train = pd.DataFrame()

for filepath in tqdm(train_files):
    df = pd.read_csv(filepath)
    df_train = pd.concat([df_train, df]).reset_index(drop=True)

df_train.fillna('__NaN__', inplace=True)

# 强迫症发作..
df_train = df_train.rename(columns={'lable': 'label'})
df_train.info()
print(len(df_train))
# %%
type(df_train['label'][0])
# %%

df_train.head(5)

# %%
df_train['label'].value_counts()

# %%




# %%

df_test = pd.read_csv('../data/test/test.csv')
df_test.fillna('__NaN__', inplace=True)
print(len(df_test))

# %%
df_test.head(5)


# %%

df = pd.concat([df_train, df_test]).reset_index(drop=True)
df.shape


# %%

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


df = add_tfidf_feats(df, 'url_unquote', n_components=256)
df = add_tfidf_feats(df, 'user_agent', n_components=256)
df = add_tfidf_feats(df, 'body', n_components=256)
df = add_tfidf_feats(df, 'url', n_components=256)


for col in tqdm(['url_filetype', 'ua_short', 'ua_first']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


not_use_feats = ['method', 'refer','id', 'user_agent', 'url', 'body', 'url_unquote', 'url_query', 'url_path', 'label']
use_features = [col for col in df.columns if col not in not_use_feats]

df_train = df[df['label'].notna()]
df_train.reset_index(drop=True)







# print(df_train[use_features].values.shape)
# assert 1==2









# %%
class My_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len, test_mode, manual_feats):
        self.method = df['method'].values
        self.user_agent = df['user_agent'].values
        self.url = df['url'].values
        self.refer = df['refer'].values
        self.body = df['body'].values
        self.manual_feats = df[manual_feats].values.astype(float)


        if not test_mode:
            self.label = df['label'].values

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.test_mode = test_mode

    def __len__(self):
        return len(self.method)

    def tokenize_text(self, text: str, max_len=512) -> tuple:

        encoded_inputs = self.tokenizer(text, max_length=max_len, padding='max_length', truncation=True)
        input_ids = torch.LongTensor(encoded_inputs['input_ids'])
        mask = torch.LongTensor(encoded_inputs['attention_mask'])
        return input_ids, mask

    def __getitem__(self, idx):

        method_maxlen = 4
        user_agent_maxlen = 60
        url_maxlen = 128
        refer_maxlen = 60
        body_maxlen = 256

        method = self.method[idx]
        user_agent = self.user_agent[idx]
        url = self.url[idx]
        refer = self.refer[idx]
        body = self.body[idx]

        # sep: '</s>'

        if len(method) > method_maxlen:
            method = method[:method_maxlen // 2] + method[-(method_maxlen // 2):]

        if len(user_agent) > user_agent_maxlen:
            user_agent = user_agent[:user_agent_maxlen // 2] + user_agent[-(user_agent_maxlen // 2):]

        if len(url) > url_maxlen:
            url = url[:url_maxlen // 2] + url[-(url_maxlen // 2):]

        if len(refer) > refer_maxlen:
            refer = refer[:refer_maxlen // 2] + refer[-(refer_maxlen // 2):]

        if len(body) > body_maxlen:
            body = body[:body_maxlen // 2] + body[-(body_maxlen // 2):]

        cat_text = method + '</s>' + body + '</s>' + user_agent + '</s>' + url + '</s>' + refer
        cat_input, cat_mask = self.tokenize_text(cat_text, max_len=self.max_len)

        sample = dict(
            input_ids=cat_input,
            attention_mask=cat_mask,
            manual_feats = torch.tensor(self.manual_feats[idx])
        )

        if not self.test_mode:
            sample['label'] = torch.LongTensor([self.label[idx]])

        return sample


# %%
def create_data_loader(df, tokenizer, max_len, batch_size, test_mode=False, manual_feats=None):
    ds = My_Dataset(
        df=df,
        tokenizer=tokenizer,
        max_len=max_len,
        test_mode=test_mode,
        manual_feats=manual_feats
    )
    if test_mode:
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=True)


# %%


# %%


# %%


# %%
df_train, df_val = train_test_split(df_train, test_size=0.1, random_state=2022)
df_train.index = range(len(df_train))
df_val.index = range(len(df_val))

print('train_data_len: ', len(df_train))
print('val_data_len: ', len(df_val))
# %%


PRE_TRAINED_MODEL_PATH = '../hfl/roberta-large/'
print('use pretrain model: ', PRE_TRAINED_MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH, lowercase=True, use_fast=True)
train_data_loader = create_data_loader(df=df_train,
                                       tokenizer=tokenizer,
                                       max_len=300,
                                       batch_size=16,
                                       test_mode=False,
                                       manual_feats=use_features)

val_data_loader = create_data_loader(df=df_val,
                                     tokenizer=tokenizer,
                                     max_len=300,
                                     batch_size=16,
                                     test_mode=False,
                                     manual_feats=use_features)
# %%


# %%
len(train_data_loader)

# %%

data = next(iter(train_data_loader))
data.keys()
print(data['input_ids'].shape)
print(data['attention_mask'].shape)

# %%
# %%
# %%
# for inputs in train_data_loader:
#     print()
# %%
# inputs
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%
# %%

# %%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use device: ', device)


# %%

class WebAttack_Classfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_cnofig = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_PATH + './config.json')
        # self.bert_cnofig.update({'output_hidden_states': True})
        self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_PATH, config=self.bert_cnofig)
        self.fc = nn.Linear(1024*2, 6)

        self.manual_dense = nn.Linear(1032, 1024)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        mean_output = output.mean(1)



        manual_feats = inputs['manual_feats'].to(device).float()

        manual_feats = self.manual_dense(manual_feats)
        manual_feats = torch.tanh(manual_feats)

        out = torch.cat((mean_output, manual_feats), dim=-1)
        out = self.dropout(out)




        out = self.fc(out)

        return out


# %%

model = WebAttack_Classfier()
model = model.to(device)

# %%

# y = model(data)


# %%
# y.shape


# %%
EPOCHS = 6  # 训练轮数
print('EPOCH: ', EPOCHS)


def build_optimizer(model, learning_rate, num_total_steps):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    print('learning_rate: ', learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-6)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps=args.max_steps)
    print('num_training_steps: ', num_total_steps)
    print('warmup_steps: ', num_total_steps * 0.1)
    # print('num_training_steps: ', args.max_steps)
    # print('warmup_steps: ', args.warmup_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
    #                                             num_training_steps= args.max_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_total_steps * 0.1,
                                                num_training_steps=num_total_steps)
    return optimizer, scheduler


total_steps = len(train_data_loader) * EPOCHS
optimizer, scheduler = build_optimizer(model, learning_rate=2e-5, num_total_steps=total_steps)

loss_fn = nn.CrossEntropyLoss().to(device)
# %%


# %%
from sklearn.metrics import accuracy_score, auc, f1_score


# print('f1: ', f1_score(np.argmax(oof_pred, axis=1), df_train['label'], average='macro'))
def train_epoch(args, model, data_loader, loss_fn, optimizer, device, scheduler):
    model = model.train()
    losses = []

    pred_list = []
    target_list = []

    for inputs in tqdm(data_loader):
        targets = inputs["label"].to(device)
        targets = targets.squeeze(1)

        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)

        pred_list.extend(preds.cpu().numpy().tolist())
        target_list.extend(targets.cpu().numpy().tolist())

        loss = loss_fn(outputs, targets)
        losses.append(loss.item())
        loss.backward()

        # -----------------------------------对抗攻击------------------------------------------------
        if args.use_fgm:
            # 对抗训练
            fgm.attack()  # 在embedding上添加对抗扰动
            outputs = model(inputs)
            loss_adv = loss_fn(outputs, targets)
            loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
            fgm.restore()  # 恢复embedding参数

        if args.use_pgd:
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    model.zero_grad()
                else:
                    pgd.restore_grad()

                outputs = model(inputs)
                loss_adv = loss_fn(outputs, targets)
                loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度

            pgd.restore()

        # ----------------------------------------------------------------------------------------

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if args.ema != False:
            args.ema.update()

    acc = accuracy_score(y_true=target_list, y_pred=pred_list)
    f1 = f1_score(y_true=target_list, y_pred=pred_list, average='macro')
    mean_loss = np.mean(losses)

    return acc, f1, mean_loss


# %%
def eval_model(args, model, data_loader, loss_fn, device):
    model = model.eval()  # 验证预测模式
    if args.ema != False:
        args.ema.apply_shadow()

    losses = []

    pred_list = []
    target_list = []

    with torch.no_grad():
        for inputs in tqdm(data_loader):
            targets = inputs["label"].to(device)
            targets = targets.squeeze(1)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            pred_list.extend(preds.cpu().numpy().tolist())
            target_list.extend(targets.cpu().numpy().tolist())

            loss = loss_fn(outputs, targets)

            losses.append(loss.item())

    acc = accuracy_score(y_true=target_list, y_pred=pred_list)
    f1 = f1_score(y_true=target_list, y_pred=pred_list, average='macro')
    mean_loss = np.mean(losses)

    return acc, f1, mean_loss


# %%
# %%
# %%
class Args:
    def __init__(self):
        self.ema = True
        self.use_fgm = True
        self.use_pgd = False


args = Args()
# %%
# %%
# %%

if args.ema == True:
    print('-' * 10, '采用EMA机制训练', '-' * 10)
    from tricks import EMA

    args.ema = EMA(model, 0.999)
    args.ema.register()

if args.use_fgm == True:
    print('-' * 10, '采用FGM对抗训练', '-' * 10)
    from tricks import FGM

    # 初始化
    fgm = FGM(model)

if args.use_pgd == True:
    print('-' * 10, '采用PGD对抗训练', '-' * 10)
    from tricks import PGD

    # 初始化
    pgd = PGD(model=model)
    K = 3

# %%
from collections import defaultdict

history = defaultdict(list)  # 记录10轮loss和acc
best_f1 = 0

# -------------------控制早停--------------
early_stop_epochs = 2
no_improve_epochs = 0

for epoch in range(EPOCHS):

    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_acc, train_f1, train_loss = train_epoch(
        args,
        model,
        train_data_loader,
        loss_fn,
        optimizer,
        device,
        scheduler
    )

    print(f'train_loss: {train_loss} \n train_acc: {train_acc} \n train_f1: {train_f1}')

    val_acc, val_f1, val_loss = eval_model(
        args,
        model,
        val_data_loader,
        loss_fn,
        device
    )

    print(f'val_loss: {val_loss} \n val_acc: {val_acc} \n val_f1: {val_f1}')
    print()

    history['train_acc'].append(train_acc)
    history['train_f1'].append(train_f1)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_f1'].append(val_f1)
    history['val_loss'].append(val_loss)

    # torch.save(model.state_dict(), f'./save model/best_model{epoch}.bin')

    if val_f1 > best_f1:
        print('best model saved!!!!!!!!!!!!!')
        torch.save(model.state_dict(), f'./save model/best_model.bin')
        best_f1 = val_f1

        no_improve_epochs = 0

    else:
        no_improve_epochs += 1

    if no_improve_epochs == early_stop_epochs:
        print('no improve score !!! stop train !!!')
        break

    if args.ema != False:
        args.ema.restore()
