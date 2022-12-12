# generate csv file contains
#   order id
#   order products
#   user id
#   eval_set

import pandas as pd
import os

ROOT_DIR = os.path.abspath('.')
order_products = pd.read_csv(os.path.join(ROOT_DIR, 'filtered_order_products.csv'))
products = pd.read_csv(os.path.join(ROOT_DIR, 'filtered_products_update1.csv'))

def id2name(df):
    id_list = df.product_id.to_list()
    order_names = ', '.join([products.loc[products.product_id==id, 'product_name'].to_string(index=False) for id in id_list])
    return order_names

groupby_output = order_products.groupby('order_id').apply(lambda df: id2name(df))

order_names = pd.DataFrame(groupby_output)
order_names.reset_index(inplace=True)
order_names.columns = ['order_id', 'order_names']
order_names.to_csv(os.path.join(ROOT_DIR, 'order_names.csv'))



order_names = pd.read_csv(os.path.join(ROOT_DIR, 'order_names.csv'))
train_orders = pd.read_csv(os.path.join(ROOT_DIR, 'train_orders.csv'))