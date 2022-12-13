# generate csv file contains
#   order id
#   order products
#   user id
#   eval_set

import pandas as pd
import os

ROOT_DIR = os.path.abspath('.')
# order_products = pd.read_csv(os.path.join(ROOT_DIR, 'filtered_order_products.csv'))
# products = pd.read_csv(os.path.join(ROOT_DIR, 'filtered_products_update1.csv'))

# def id2name(df):
#     id_list = df.product_id.to_list()
#     order_names = ', '.join([products.loc[products.product_id==id, 'product_name'].to_string(index=False) for id in id_list])
#     return order_names

# groupby_output = order_products.groupby('order_id').apply(lambda df: id2name(df))

# order_names = pd.DataFrame(groupby_output)
# order_names.reset_index(inplace=True)
# order_names.columns = ['order_id', 'order_names']
# order_names.to_csv(os.path.join(ROOT_DIR, 'order_names.csv'), index=False)


order_names = pd.read_csv(os.path.join(ROOT_DIR, 'order_names.csv'))
order_product_names = pd.read_csv(os.path.join(ROOT_DIR, 'order_product_names.csv'))

# train data
train_orders = pd.read_csv(os.path.join(ROOT_DIR, 'train_orders.csv'))
train_all = train_orders[['order_id', 'user_id', 'eval_set']]
train_all = train_all.merge(order_names, on='order_id')

# train_all = train_all.merge(order_product_names, on='order_id')
# train_all.to_csv('train/data/train_all.csv', index=False)

train_count = sum(train_all.user_id.value_counts() == 10)
train_all_1k_id = train_all.user_id.value_counts()[-train_count:].index.to_list()
train_all_1k = train_all.loc[train_all.user_id.isin(train_all_1k_id),]
train_all_1k = train_all_1k.merge(order_product_names, on='order_id')
rel_score = train_all_1k.groupby('order_id').apply(lambda df: 1/df.shape[0])
rel_score = pd.DataFrame(rel_score, columns=['rel_score']).reset_index()
train_all_1k = train_all_1k.merge(rel_score, on='order_id')
# train_all_1k['rel_score'] = train_all_1k['rel_score'].astype(str)

train_all_1k.to_csv('train/data/train_all_4k.csv', index=False, header=False, sep=';')

# validation data
valid_orders = pd.read_csv(os.path.join(ROOT_DIR, 'valid_orders.csv'))
valid_all = valid_orders[['order_id', 'user_id', 'eval_set']]
valid_all = valid_all.merge(order_names, on='order_id')

valid_count = sum(valid_all.user_id.value_counts() == 10)
valid_all_100_id = valid_all.user_id.value_counts()[-valid_count:].index.to_list()
valid_all_100 = valid_all.loc[valid_all.user_id.isin(valid_all_100_id),]
valid_all_100 = valid_all_100.merge(order_product_names, on='order_id')
rel_score = valid_all_100.groupby('order_id').apply(lambda df: 1/df.shape[0])
rel_score = pd.DataFrame(rel_score, columns=['rel_score']).reset_index()
valid_all_100 = valid_all_100.merge(rel_score, on='order_id')
valid_all_100.to_csv('train/data/valid_all_700.csv', index=False, header=False, sep=';')

# test data
test_orders = pd.read_csv(os.path.join(ROOT_DIR, 'test_orders.csv'))
test_all = test_orders[['order_id', 'user_id', 'eval_set']]
test_all = test_all.merge(order_names, on='order_id')

test_count = sum(test_all.user_id.value_counts() == 10)
test_all_100_id = test_all.user_id.value_counts()[-test_count:].index.to_list()
test_all_100 = test_all.loc[test_all.user_id.isin(test_all_100_id),]
test_all_100 = test_all_100.merge(order_product_names, on='order_id')
rel_score = test_all_100.groupby('order_id').apply(lambda df: 1/df.shape[0])
rel_score = pd.DataFrame(rel_score, columns=['rel_score']).reset_index()
test_all_100 = test_all_100.merge(rel_score, on='order_id')
test_all_100.to_csv('train/data/test_all_2k.csv', index=False, header=False, sep=';')