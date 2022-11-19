import pandas as pd
import numpy as np
from collections import Counter
import os

ROOT_DIR = os.path.abspath('.')

orders = pd.read_csv(os.path.join(ROOT_DIR, "filtered_orders.csv"))
order_products = pd.read_csv(os.path.join(ROOT_DIR, "filtered_order_products.csv"))
# user_ids = np.array(orders['user_id'].unique())
user_counter = Counter(orders['user_id'])
user_order_freq = list(zip(*user_counter.most_common()))
user_ids = [user_order_freq[0][index] for index, value in enumerate(user_order_freq[1]) if value >= 10]

np.random.seed(650) # set seed
np.random.shuffle(user_ids) # shuffle user ids for split
train_ids = user_ids[:int(np.ceil(len(user_ids)*0.6))]
valid_ids = user_ids[int(np.ceil(len(user_ids)*0.6)):int(np.ceil(len(user_ids)*0.7))]
test_ids = user_ids[int(np.ceil(len(user_ids)*0.7)):]

def relabel(df, ids):
    for id in ids:
        df_index = df[df['user_id'] == id].index
        y_num = int(np.floor(len(df_index)*0.3))
        shuffled_list = np.arange(len(df_index))
        np.random.shuffle(shuffled_list)
        df_y_index = df_index[shuffled_list[:y_num]]
        df.loc[df_y_index,'eval_set'] = 'y'
    return df

train_orders = orders.loc[orders['user_id'].isin(train_ids)]
train_orders = relabel(train_orders, train_ids)
train_orders.replace("prior", "train", inplace=True)
train_orders.replace("y", "train_y", inplace=True)

valid_orders = orders.loc[orders['user_id'].isin(valid_ids)]
valid_orders = relabel(valid_orders, valid_ids)
valid_orders.replace("prior", "valid", inplace=True)
valid_orders.replace("y", "valid_y", inplace=True)

test_orders = orders.loc[orders['user_id'].isin(test_ids)]
test_orders = relabel(test_orders, test_ids)
test_orders.replace("prior", "test", inplace=True)
test_orders.replace("y", "test_y", inplace=True)

train_orders.to_csv(os.path.join(ROOT_DIR, "train_orders.csv"), index=False)
valid_orders.to_csv(os.path.join(ROOT_DIR, "valid_orders.csv"), index=False)
test_orders.to_csv(os.path.join(ROOT_DIR, "test_orders.csv"), index=False)

# check kinds/number of products in each orders dataset
len(order_products.loc[order_products['order_id'].isin(list(train_orders['order_id']))]['product_id'].unique())
len(order_products.loc[order_products['order_id'].isin(list(valid_orders['order_id']))]['product_id'].unique())
len(order_products.loc[order_products['order_id'].isin(list(test_orders['order_id']))]['product_id'].unique())