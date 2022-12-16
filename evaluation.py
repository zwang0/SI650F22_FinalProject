# naive evalution using pretrain-BERT model and test dataset

import pandas as pd
import numpy as np
import itertools
import os
from pathlib import Path
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

ROOT_DIR = os.path.abspath('.')
# test_orders = pd.read_csv(os.path.join(ROOT_DIR, "test_orders.csv"))
# order_products = pd.read_csv(os.path.join(ROOT_DIR, "filtered_order_products.csv"))
# products = pd.read_csv(os.path.join(ROOT_DIR, "filtered_products.csv"))

model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def BERT_embedding(texts, model, tokenizer):
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    for text in texts:
        new_tokens = tokenizer.encode_plus(text, max_length=32, truncation=True,
                                padding='max_length', return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask
    mean_pooled = mean_pooled.detach().numpy()
    return mean_pooled


# # using user_id
# user_ids = test_orders['user_id'].unique()[:10]
# # products_pooled = BERT_embedding(products['product_name'].to_list(), model, tokenizer)

# b1_cos_sims = np.empty((0, 10))
# b2_cos_sims = np.empty((0, 10))

# for id in user_ids:
#     userid = test_orders[test_orders['user_id'] == id]
#     userid_test = userid[userid['eval_set'] == 'test']
#     userid_testy = userid[userid['eval_set'] == 'test_y']

#     test_product_id = order_products.loc[order_products['order_id'].isin(userid_test['order_id']), 'product_id']
#     test_product_names = products.loc[products['product_id'].isin(test_product_id), 'product_name'].to_list()
#     test_y_product_id = order_products.loc[order_products['order_id'].isin(userid_testy['order_id']), 'product_id']
#     test_y_product_names = products.loc[products['product_id'].isin(test_y_product_id), 'product_name'].to_list()

#     test_prod_names_pooled = BERT_embedding(test_product_names, model, tokenizer)
#     test_y_prod_names_pooled = BERT_embedding(test_y_product_names, model, tokenizer)

#     # baseline 1, randomly pick products
#     avg_shopping_len = int(np.ceil((len(test_product_id) + len(test_y_product_id)) / userid.shape[0]))
#     random_product_names = products.sample(avg_shopping_len, random_state=650)['product_name'].to_list()
#     random_product_names_pooled = BERT_embedding(random_product_names, model, tokenizer)
#     b1_cos_sim = cosine_similarity(test_prod_names_pooled, random_product_names_pooled)
#     b1_cos_sims = np.append(b1_cos_sims, [b1_cos_sim.mean()])
#     print(id)
#     print(avg_shopping_len)
#     print(b1_cos_sim.mean())

#     # baseline 2, top frequet products
#     freq_product_names = products.sort_values("product_freq", ascending=False, ignore_index=True).loc[:avg_shopping_len, "product_name"].to_list()
#     freq_product_names_pooled = BERT_embedding(freq_product_names, model, tokenizer)
#     b2_cos_sim = cosine_similarity(test_prod_names_pooled, freq_product_names_pooled)
#     b2_cos_sims = np.append(b2_cos_sims, [b2_cos_sim.mean()])
#     print(b2_cos_sim.mean())

#     # history
#     test_y_product_names_pooled = BERT_embedding(test_y_product_names, model, tokenizer)
#     hist_cos_sim = cosine_similarity(test_prod_names_pooled, test_y_product_names_pooled)
#     print(hist_cos_sim.mean())

# print(b1_cos_sims.mean())
# print(b2_cos_sims.mean())

# prediction evaluation
results_dir = os.path.join(ROOT_DIR, "train", "results")

pred_orders = pd.read_csv(os.path.join(results_dir,"prediction_orders2.csv"), sep=";")
eval_orders = pd.read_csv(os.path.join(results_dir,"test_y_df.csv"), sep=";")

eval_user_ids = pred_orders.user_id.unique()

cos_sims = pd.DataFrame({'user_id':eval_user_ids, 'cos_sim':[0]*len(eval_user_ids)})

for id in eval_user_ids:
    pred_order = pred_orders.loc[pred_orders.user_id == id, 'predict_order'].to_list()[0]
    pred_order_names = pred_order.split(',')
    eval_order = eval_orders.loc[eval_orders.user_id == id, 'order_names'].to_list()
    eval_order_names = [order.split(', ') for order in eval_order]
    eval_order_names = np.unique(list(itertools.chain.from_iterable(eval_order_names)))
    pred_order_pooled = BERT_embedding(pred_order, model, tokenizer)
    eval_order_pooled = BERT_embedding(eval_order, model, tokenizer)
    cos_sim = cosine_similarity(pred_order_pooled, eval_order_pooled)
    cos_sims.loc[cos_sims.user_id==id, 'cos_sim'] = cos_sim.mean()

cos_sims.to_csv(os.path.join(ROOT_DIR, "eval_cos_sim_scores2.csv"), index=False)