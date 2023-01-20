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


# %%

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.manual_seed(seed)


setup_seed(2022)


df_test = pd.read_csv('../data/test/test.csv')
df_test.fillna('__NaN__', inplace=True)
print(len(df_test))

# %%
df_test.head(5)


# %%
class My_Dataset(Dataset):
    def __init__(self, df, tokenizer, max_len, test_mode):
        self.method = df['method'].values
        self.user_agent = df['user_agent'].values
        self.url = df['url'].values
        self.refer = df['refer'].values
        self.body = df['body'].values
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
            attention_mask=cat_mask
        )

        if not self.test_mode:
            sample['label'] = torch.LongTensor([self.label[idx]])

        return sample


# %%
def create_data_loader(df, tokenizer, max_len, batch_size, test_mode=False):
    ds = My_Dataset(
        df=df,
        tokenizer=tokenizer,
        max_len=max_len,
        test_mode=test_mode
    )
    if test_mode:
        return DataLoader(ds, batch_size=batch_size, shuffle=False)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=True)



PRE_TRAINED_MODEL_PATH = 'E:/打工/预训练模型/hfl/roberta-base/'
tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_PATH)

test_data_loader = create_data_loader(df=df_test,
                                     tokenizer=tokenizer,
                                     max_len=512,
                                     batch_size=16,
                                     test_mode=True)



class WebAttack_Classfier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert_cnofig = RobertaConfig.from_pretrained(PRE_TRAINED_MODEL_PATH + './config.json')
        self.bert = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_PATH, config=self.bert_cnofig)
        self.fc = nn.Linear(768, 6)


    def forward(self, inputs):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        output, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # out = self.fc(pooled_output)

        mean_output = output.mean(1)
        out = self.fc(mean_output)

        return out



# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('use device: ', device)
model = WebAttack_Classfier()


path = './save model/best_model.bin'
model.load_state_dict(torch.load(path))
model = model.to(device)


model.eval()

pred_list = []
with torch.no_grad():
    for inputs in tqdm(test_data_loader):

        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)

        pred_list.extend(preds.cpu().numpy().tolist())


print('pred_list_len: ', len(pred_list))

sub = pd.read_csv('../data/submit_example.csv')
sub['predict'] = pred_list


# %%

sub['predict'].value_counts()

# %%

sub.to_csv('bert_main.csv', index=False)



# # 3. inference
# predictions = []
# with torch.no_grad():
#     for batch in tqdm(dataloader):
#         pred_label_id = model(batch, inference=True)
#         predictions.extend(pred_label_id.cpu().numpy())
#
# # 4. dump results
# with open(args.test_output_csv, 'w') as f:
#     for pred_label_id, ann in zip(predictions, dataset.anns):
#         video_id = ann['id']
#         category_id = lv2id_to_category_id(pred_label_id)
#         f.write(f'{video_id},{category_id}\n')










