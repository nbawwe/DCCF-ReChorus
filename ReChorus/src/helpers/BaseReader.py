# -*- coding: UTF-8 -*-

import os
import pickle
import argparse
import logging
import numpy as np
import pandas as pd

from utils import utils


class BaseReader(object):
    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--path', type=str, default='data/',
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='Grocery_and_Gourmet_Food',
                            help='Choose a dataset.')
        parser.add_argument('--sep', type=str, default='\t',
                            help='sep of csv file.')
        return parser

    def __init__(self, args):
        self.sep = args.sep
        self.prefix = args.path
        self.dataset = args.dataset
        self._read_data()
        # 存储每一个用户在trian数据集点击的物品的集合
        self.train_clicked_set = dict()  # store the clicked item set of each user in training set
        # 每个用户在 dev 和 test 数据集点击的物品集合。
        self.residual_clicked_set = dict()  # store the residual clicked item set of each user
        # 遍历每个数据集
        for key in ['train', 'dev', 'test']:
            df = self.data_df[key]
            # 遍历当前数据集中每一行的 user_id（用户 ID）和 item_id（物品 ID）
            for uid, iid in zip(df['user_id'], df['item_id']):
                if uid not in self.train_clicked_set:
                    self.train_clicked_set[uid] = set()
                    self.residual_clicked_set[uid] = set()
                if key == 'train':
                    self.train_clicked_set[uid].add(iid)
                else:
                    self.residual_clicked_set[uid].add(iid)

    def _read_data(self):
        logging.info('Reading data from \"{}\", dataset = \"{}\" '.format(self.prefix, self.dataset))
        self.data_df = dict()
        for key in ['train', 'dev', 'test']:
            # 加载各个数据集,重置索引，按照user_id和time排序
            self.data_df[key] = pd.read_csv(os.path.join(self.prefix, self.dataset, key + '.csv'),
                                            sep=self.sep).reset_index(drop=True)#.sort_values(by=['user_id', 'time'])
            self.data_df[key] = self.data_df[key].sample(frac=1, random_state=42).reset_index(drop=True)
            self.data_df[key] = utils.eval_list_columns(self.data_df[key])

        logging.info('Counting dataset statistics...')
        # 定义了一组需要用到的关键列， 如果data_df['train']有label列则（为ctr的）也加入
        key_columns = ['user_id','item_id','time']
        if 'label' in self.data_df['train'].columns: # Add label for CTR prediction
            key_columns.append('label')
        # 合并train, dev, test到一个大的df
        self.all_df = pd.concat([self.data_df[key][key_columns] for key in ['train', 'dev', 'test']])
        # 计算用户和物品的总数
        self.n_users, self.n_items = self.all_df['user_id'].max() + 1, self.all_df['item_id'].max() + 1
        # 如果dev和test包含 neg_items（负样本）列，转换为np数组
        for key in ['dev', 'test']:
            if 'neg_items' in self.data_df[key]:
                neg_items = np.array(self.data_df[key]['neg_items'].tolist())
                assert (neg_items >= self.n_items).sum() == 0  # assert negative items don't include unseen ones
        logging.info('"# user": {}, "# item": {}, "# entry": {}'.format(
            self.n_users - 1, self.n_items - 1, len(self.all_df)))
        if 'label' in key_columns:
            positive_num = (self.all_df.label==1).sum()
            logging.info('"# positive interaction": {} ({:.1f}%)'.format(
				positive_num, positive_num/self.all_df.shape[0]*100))
        
