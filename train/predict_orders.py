# # transpose relevance-scores file in each epoch and clean scores data
# import re
# import os
# import pandas as pd
# for epoch in range(5):
#     file_path = os.path.join('results', 'epoch'+str(epoch)+'relevance-scores.csv')
#     df = pd.read_csv(file_path)
#     df = df.transpose()
#     df.reset_index(inplace=True)
#     df.columns = ['scores']
#     score_list = list(df['scores'])
#     score_list = [re.sub('.[0-9]$', '', score) for score in score_list]
#     score_list = [re.sub('e-$', '',score) for score in score_list]
#     score_list = [float(i) for i in score_list]
#     df['scores'] = score_list
#     save_path = os.path.join('results', 'epoch' + str(epoch) + '_relevance-scores_transpose.csv')
#     df.to_csv(save_path, index=False)


# generate product prediction for users
import pandas as pd
import numpy
import os
ROOT_DIR = os.path.abspath("./train")
results_dir = os.path.join(ROOT_DIR, "results")
test_combined_df = pd.read_csv(os.path.join(results_dir, "test_combined_df.csv"), sep=";")

def gen_order(df, thres):
    product_list = df.loc[df.pred_scores > thres, 'product_names'].to_list()
    if not product_list:
        product_list = df.loc[df.pred_scores == df.pred_scores.max(), 'product_names'].to_list()
    pre_order = ", ".join(product for product in product_list)
    return pre_order

def gen_order2(df, topk):
    product_list = df.nlargest(topk, 'pred_scores')['product_names'].to_list()
    pre_order = ", ".join(product for product in product_list)
    return pre_order

def order_length(orders):
    lens = []
    for order in orders:
        if order == "":
            lens.append(0)
        else:
            lens.append(len(order.split(",")))
    return lens

# predict_df = test_combined_df.groupby('user_id').apply(lambda df: gen_order(df, 0.3))
# predict_df = pd.DataFrame(predict_df)
# predict_df.reset_index(inplace=True)
# predict_df.columns = ['user_id', 'predict_order']
# predict_df['order_length'] = order_length(predict_df.predict_order.to_list())

# predict_df.to_csv(os.path.join(results_dir, "prediction_orders.csv"), sep=";")

predict_df2 = test_combined_df.groupby('user_id').apply(lambda df: gen_order2(df, 10))
predict_df2 = pd.DataFrame(predict_df2)
predict_df2.reset_index(inplace=True)
predict_df2.columns = ['user_id', 'predict_order']
predict_df2['order_length'] = order_length(predict_df2.predict_order.to_list())

predict_df2.to_csv(os.path.join(results_dir, "prediction_orders2.csv"), sep=";", index=False)

    