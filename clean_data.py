# clean data

import pandas as pd
import numpy as np
import os
from pathlib import Path
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

ROOT_DIR = os.path.abspath('.')
instacart_path = os.path.join(ROOT_DIR, "instacart-market-basket-analysis")
# select instacart products in selected aisle id
# 2	    specialty cheeses
# 24	fresh fruits
# 42	frozen vegan vegetarian
# 47	vitamins supplements
# 57	granola
# 83	fresh vegetables
# 84	milk
# 86	eggs
# 91	soy lactosefree
# 95	canned meat seafood
# 96	lunch meat
# 112	bread
# 117	nuts seeds dried fruit
# 120	yogurt
# 121	cereal
# 122	meat counter
# 123	packaged vegetables fruits
# 128	tortillas flat bread
selected_aisle_id = (2, 24, 42, 47, 57, 83, 84, 86, 91, 95, 96, 112, 117, 120, 121, 122, 123, 128)
products = pd.read_csv(os.path.join(instacart_path, "products.csv"))
nutrition  = pd.read_csv(os.path.join(ROOT_DIR, "nutrition.csv"))
nutrition.rename(columns={'Unnamed: 0': 'nid'}, inplace=True)
selected_products = products.loc[products['aisle_id'].isin(selected_aisle_id)].reset_index()
product_names = list(selected_products['product_name'])
nutrition_names = list(nutrition['name'])

# BERT embedding
# load pre-trained BERT model
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def BERT_embedding(texts, model, tokenizer):
    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}
    for text in texts:
        new_tokens = tokenizer.encode_plus(text, max_length=64, truncation=True,
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

prod_names_pooled = BERT_embedding(product_names, model, tokenizer)
nutri_names_pooled = BERT_embedding(nutrition_names, model, tokenizer)
# calculate cos similiarty
match_nid = np.empty(prod_names_pooled.shape[0], dtype=int)
for i, prod_name_pooled in enumerate(prod_names_pooled):
    cos_sim = cosine_similarity([prod_name_pooled], nutri_names_pooled)
    match_nid[i] = cos_sim.argmax()

selected_products['nid'] = pd.Series(match_nid)
selected_products.to_csv(os.path.join(ROOT_DIR,"selected_products.csv"), index=False)