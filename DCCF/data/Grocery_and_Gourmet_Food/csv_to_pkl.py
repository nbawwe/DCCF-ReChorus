import pickle
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

# f = open('./data/Grocery_and_Gourmet_Food/train.pkl', 'rb')
# data = pickle.load(f)
# print(data)

df = pd.read_csv("./train.csv", sep='\t').reset_index(drop=True).sort_values(by = ['user_id','time'])
row = np.array(df[['user_id']]).T.ravel()-1 #-1：索引从0开始
col = np.array(df[['item_id']]).T.ravel()-1
value = np.ones((1, col.size)).ravel()
data = coo_matrix((value, (row, col)))
print(data)
with open('./train.pkl', 'wb') as file:
    pickle.dump(data, file)