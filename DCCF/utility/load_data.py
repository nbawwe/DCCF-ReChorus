import pickle
import numpy as np
from time import time
from tqdm import tqdm
import scipy.sparse as sp

class Data(object):
    def __init__(self, args):

        self.path = args.data_path + args.dataset
        self.n_batch = args.n_batch     # n_batch = 40
        self.batch_size = args.batch_size   # 10240
        self.train_num = args.train_num     # 10000 模型看到训练样本的总数
        self.sample_num = args.sample_num   # 采样多少个正负样本对 40

        try:
            train_file = self.path + '/train.pkl'
            test_file = self.path + '/test.pkl'
            with open(train_file, 'rb') as f:
                train_mat = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_mat = pickle.load(f)
        except Exception as e:
            print("Try an alternative way of reading the data.")
            train_file = self.path + '/train_index.pkl'
            test_file = self.path + '/test_index.pkl'
            with open(train_file, 'rb') as f:
                train_index = pickle.load(f)
            with open(test_file, 'rb') as f:
                test_index = pickle.load(f)
            train_row, train_col = train_index[0], train_index[1]
            n_user = max(train_row) + 1
            n_item = max(train_col) + 1
            train_mat = sp.coo_matrix((np.ones(len(train_row)), (train_row, train_col)), shape=[n_user, n_item])
            test_row, test_col = test_index[0], test_index[1]
            test_mat = sp.coo_matrix((np.ones(len(test_row)), (test_row, test_col)), shape=[n_user, n_item])

        # get number of users and items
        self.n_users, self.n_items = train_mat.shape[0], train_mat.shape[1]     # 所有用户和物品数, 用户50821, 物品57440
        self.n_train, self.n_test = len(train_mat.row), len(test_mat.row)       # 返回矩阵中非零元素的行索引,长度为1172425 (Aindices的一半，因为A中重复了两次)

        self.print_statistics()

        self.R = train_mat.todok()      # 创建关系矩阵, 存储变成字典：键是 (row, col) 元组，值是非零元素
        self.train_items, self.test_set = {}, {}
        train_uid, train_iid = train_mat.row, train_mat.col
        for i in range(len(train_uid)):
            uid = train_uid[i]
            iid = train_iid[i]
            if uid not in self.train_items:
                self.train_items[uid] = [iid]
            else:
                self.train_items[uid].append(iid)
        test_uid, test_iid = test_mat.row, test_mat.col
        for i in range(len(test_uid)):
            uid = test_uid[i]
            iid = test_iid[i]
            if uid not in self.test_set:
                self.test_set[uid] = [iid]
            else:
                self.test_set[uid].append(iid)

    def get_adj_mat(self):
        adj_mat = self.create_adj_mat()
        return adj_mat

    def create_adj_mat(self):
        t1 = time()
        rows = self.R.tocoo().row
        cols = self.R.tocoo().col
        new_rows = np.concatenate([rows, cols + self.n_users], axis=0)
        new_cols = np.concatenate([cols + self.n_users, rows], axis=0)
        adj_mat = sp.coo_matrix((np.ones(len(new_rows)), (new_rows, new_cols)), shape=[self.n_users + self.n_items, self.n_users + self.n_items]).tocsr().tocoo()
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)
        return adj_mat.tocsr()

    def uniform_sample(self):
        # 生成随机用户ID，总共n_batch * batch_size = 409600个，数值在0~50820之间
        users = np.random.randint(0, self.n_users, int(self.n_batch * self.batch_size))
        train_data = []
        for i, user in tqdm(enumerate(users), desc='Sampling Data', total=len(users)):
            # 从正样本中随机选择一个物品
            pos_for_user = self.train_items[user]
            pos_index = np.random.randint(0, len(pos_for_user))
            pos_item = pos_for_user[pos_index]
            # 随机选择一个样本,直到确认是负样本
            while True:
                neg_item = np.random.randint(0, self.n_items)
                if self.R[user, neg_item] == 1:
                    continue
                else:
                    break
            # 训练数据格式：(用户，正样本，负样本)
            train_data.append([user, pos_item, neg_item])
        # 存到本地
        self.train_data = np.array(train_data)
        return len(self.train_data)

    def mini_batch(self, batch_idx):
        # batch第一个和最后一个的位置
        st = batch_idx * self.batch_size
        ed = min((batch_idx + 1) * self.batch_size, len(self.train_data))
        # 从存储的训练数据中切片出来
        batch_data = self.train_data[st: ed]
        users = batch_data[:, 0]
        pos_items = batch_data[:, 1]
        neg_items = batch_data[:, 2]
        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('n_interactions=%d' % (self.n_train + self.n_test))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items)))

    def get_statistics(self):
        sta = ""
        sta += 'n_users=%d, n_items=%d\t' % (self.n_users, self.n_items)
        sta += 'n_interactions=%d\t' % (self.n_train + self.n_test)
        sta += 'n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train)/(self.n_users * self.n_items))
        return sta
