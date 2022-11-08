import pandas as pd
import os
ROOT_DIR = os.path.abspath('.')
selected_products_path = os.path.join(ROOT_DIR, "selected_products.csv")
nutrition_path = os.path.join(ROOT_DIR, "nutrition.csv")
selected_products = pd.read_csv(selected_products_path)
nutrition = pd.read_csv(nutrition_path)
nutrition.rename(columns={'Unnamed: 0': 'nid'}, inplace=True)
nutri_prod = selected_products.merge(nutrition, on='nid')
nutri_prod.to_csv(os.path.join(ROOT_DIR,"nutri_prod.csv"), index=False)