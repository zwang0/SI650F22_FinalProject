# clean data

import pandas as pd
import numpy as np
import os
from pathlib import Path
from transformers import BertModel, BertTokenizer
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
nutrition['nid'] = range(1,np.shape(nutrition)[0])
selected_products = products.loc[products['aisle_id'].isin(selected_aisle_id)].reset_index()
product_names = selected_products['product_name']

# function match the product with nutrition id
np.unique(selected_products['product_name'])


# BERT embedding
# load pre-trained BERT model
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

