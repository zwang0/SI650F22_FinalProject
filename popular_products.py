# find popular products in shopping carts

import pandas as pd
import numpy as np
import os
from pathlib import Path
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import spacy

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
order_products = pd.read_csv(os.path.join(instacart_path, "order_products__prior.csv"))
selected_products = products.loc[products['aisle_id'].isin(selected_aisle_id)].reset_index(drop=True)
product_names = list(selected_products['product_name'])

order_prod_counter = Counter(order_products.iloc[:,1]) # count product_id appearance
popular_products = order_prod_counter.most_common(5000) # (prod_id, appearance)
popular_products_id_freq = list(zip(*popular_products))
popular_prod_id = popular_products_id_freq[0]
selected_popular_id = [id for id in list(popular_prod_id) if id in list(selected_products['product_id'])]

# find orders contains
filtered_order_products = order_products.loc[order_products['product_id'].isin(selected_popular_id)]
# filtered_order_products.to_csv(os.path.join(ROOT_DIR,"filtered_order_products.csv"), index=False)

# find order id
filtered_order_id = filtered_order_products['order_id'].unique()

# find customers in these order id
orders = pd.read_csv(os.path.join(instacart_path, "orders.csv"))
filtered_orders = orders.loc[orders['order_id'].isin(filtered_order_id)]
# filtered_orders.to_csv(os.path.join(ROOT_DIR,"filtered_orders.csv"), index=False)
filtered_user_id = filtered_orders['user_id'].unique()

# nutrition types

# Calcium
# Dietary
# Fat
# Magnesium
# Manganese
# Phosphorus
# Potassium
# Vitamin C
# Vitamin D
# Vitamin K
# Biotin #NA
# Chloride #NA
# Chromium #NA
# Copper
# Folate/Folic Acid
# Molybdenum #NA
# Niacin
# Pantothenic Acid
# Riboflavin
# Selenium
# Sodium
# Thiamin
# Total carbohydrate
# Vitamin A
# Vitamin B6
# Vitamin B12
# Vitamin E
# Zinc
# Cholesterol
# Iodine # NA
# Iron
# Protein
# Saturated fat
# Added sugars
# Choline

# calories,total_fat,saturated_fat,cholesterol,sodium,choline,folate,folic_acid,
# niacin,pantothenic_acid,riboflavin,thiamin,vitamin_a,vitamin_a_rae,carotene_alpha,
# carotene_beta,cryptoxanthin_beta,lutein_zeaxanthin,lucopene,vitamin_b12,vitamin_b6,
# vitamin_c,vitamin_d,vitamin_e,tocopherol_alpha,vitamin_k,calcium,copper,irom,
# magnesium,manganese,phosphorous,potassium,selenium,zink,protein,alanine,arginine,
# aspartic_acid,cystine,glutamic_acid,glycine,histidine,hydroxyproline,isoleucine,
# leucine,lysine,methionine,phenylalanine,proline,serine,threonine,tryptophan,tyrosine,
# valine,carbohydrate,fiber,sugars,fructose,galactose,glucose,lactose,maltose,sucrose,
# fat,saturated_fatty_acids,monounsaturated_fatty_acids,polyunsaturated_fatty_acids,
# fatty_acids_total_trans,alcohol,ash,caffeine,theobromine,water

# in common
nutri_list = ["total_fat", "fat", "saturated_fat", "cholesterol", "sodium", "choline", "folate",
"folic_acid", "niacin", "pantothenic_acid", "riboflavin", "thiamin", "vitamin_a", 
"vitamin_b12", "vitamin_b6", "vitamin_c", "vitamin_d", "vitamin_e", "vitamin_k",
"calcium", "copper", "iron", "magnesium", "manganese", "phosphorus", "potassium",
"selenium", "zinc", "protein", "carbohydrate", "sugars"]

# filtered products data frame
nutri_prod = pd.read_csv("nutri_prod.csv")
colnames_list = ['product_id', 'product_name', 'aisle_id', 'department_id', 'nid',
       'name', 'serving_size']
colnames_list.extend(nutri_list)
filtered_products = nutri_prod[colnames_list]
filtered_products = filtered_products.loc[filtered_products['product_id'].isin(selected_popular_id)]
filtered_products['product_freq'] = filtered_products['product_id'].apply(lambda x: 
    popular_products_id_freq[1][popular_products_id_freq[0].index(x)])
# filtered_products.to_csv(os.path.join(ROOT_DIR,"filtered_products.csv"), index=False)