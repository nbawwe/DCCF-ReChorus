import torch
import numpy as np
import torch_sparse
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F

class DCCF(nn.Module):
    """模型中超参数：
        self.emb_dim = args.embed_size  默认值32
        self.n_layers = args.n_layers   默认值2
        self.n_intents = args.n_intents 默认装128
        self.temp = args.temp

        self.batch_size = args.batch_size
        self.emb_reg = args.emb_reg
        self.cen_reg = args.cen_reg
        self.ssl_reg = args.ssl_reg
        """
    def __init__(self, data_config, args):
        super(DCCF, self).__init__()

        self.n_users = data_config['n_users']       # 所有用户数量50821
        self.n_items = data_config['n_items']       # 所有物品数量57440
        self.plain_adj = data_config['plain_adj']   # 创建的矩阵(108261, 108261)  108261 = 50821+57440
        self.all_h_list = data_config['all_h_list']
        # print(len(self.all_h_list))     # 2344850 稀疏矩阵非空元素的横坐标
        self.all_t_list = data_config['all_t_list']
        # print(len(self.all_t_list))     # 2344850 稀疏矩阵的纵坐标
        self.A_in_shape = self.plain_adj.tocoo().shape      # (108261, 108261)
        self.A_indices = torch.tensor([self.all_h_list, self.all_t_list], dtype=torch.long).cuda()  # 生成A的坐标对,维度(2, 2344850)
        self.D_indices = torch.tensor([list(range(self.n_users + self.n_items)), list(range(self.n_users + self.n_items))], dtype=torch.long).cuda()
        # D的坐标对 (2, 108261)
        self.all_h_list = torch.LongTensor(self.all_h_list).cuda()  # 放cuda上
        self.all_t_list = torch.LongTensor(self.all_t_list).cuda()
        # 归一化后的A，暂存在这 格式同A
        self.G_indices, self.G_values = self._cal_sparse_adj()
        self.emb_dim = args.embed_size      # emb大小32
        self.n_layers = args.n_layers       # 模型层数2
        self.n_intents = args.n_intents     # 意图数量128
        self.temp = args.temp               # 计算cl用到的温度 1

        self.batch_size = args.batch_size   # 10240
        self.emb_reg = args.emb_reg         # loss参数，下同
        self.cen_reg = args.cen_reg
        self.ssl_reg = args.ssl_reg


        """
        *********************************************************
        Create Model Parameters
        """
        # 用户与物品嵌入的定义    (10821, 32)
        self.user_embedding = nn.Embedding(self.n_users, self.emb_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.emb_dim)
        # 意图嵌入   (32, 128)
        _user_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_user_intent)
        self.user_intent = torch.nn.Parameter(_user_intent, requires_grad=True)
        # 物品上下文嵌入
        _item_intent = torch.empty(self.emb_dim, self.n_intents)
        nn.init.xavier_normal_(_item_intent)
        self.item_intent = torch.nn.Parameter(_item_intent, requires_grad=True)

        """
        *********************************************************
        Initialize Weights
        """
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def _cal_sparse_adj(self):
        """归一化函数，返回归一化矩阵\tidleA(即G)的边和值"""
        A_values = torch.ones(size=(len(self.all_h_list), 1)).view(-1).cuda()   # 创建一个长度和稀疏矩阵元素个数相同长的，为1的序列
        # 用来填充A_tensor(关系矩阵),=1表示有关系
        # 创建稀疏矩阵A_tensor
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=A_values, sparse_sizes=self.A_in_shape).cuda()
        # 归一化
        D_values = A_tensor.sum(dim=1).pow(-0.5)    # 行求和得到度向量再开根号
        # G为归一化后的结果，分开为G_indices 和 G_values存储
        G_indices, G_values = torch_sparse.spspmm(self.D_indices, D_values, self.A_indices, A_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])
        G_indices, G_values = torch_sparse.spspmm(G_indices, G_values, self.D_indices, D_values, self.A_in_shape[0], self.A_in_shape[1], self.A_in_shape[1])

        return G_indices, G_values

    def _adaptive_mask(self, head_embeddings, tail_embeddings):
        """计算解缠掩码M和应用后求得的隐性关系矩阵G"""
        # 对节点嵌入进行归一化,两个节点的内积值就可以直接反映它们之间的余弦相似性。
        # 传入(2344840, 32) 每一行的行元素归一化，即对特征归一化
        head_embeddings = torch.nn.functional.normalize(head_embeddings)
        tail_embeddings = torch.nn.functional.normalize(tail_embeddings)
        # 计算它们的内积得到s，再+1再/2得到M
        # 逐元素相乘得到(23344840, 32)再对特征维度求和得到(23344840), 然后每个元素+1再/2
        edge_alpha = (torch.sum(head_embeddings * tail_embeddings, dim=1).view(-1) + 1) / 2
        # 载入A的值(这里已经逐元素相乘了M)
        A_tensor = torch_sparse.SparseTensor(row=self.all_h_list, col=self.all_t_list, value=edge_alpha, sparse_sizes=self.A_in_shape).cuda()

        # 计算A的归一化矩阵D^-1,这样先对A作归一化，G=A*M得到的就是已归一化的结果
        # D_scores_inv 每行度之和 ** -1 的tensor列表
        D_scores_inv = A_tensor.sum(dim=1).pow(-1).nan_to_num(0, 0, 0).view(-1)

        G_indices = torch.stack([self.all_h_list, self.all_t_list], dim=0)  # 再次拼接出G的坐标(2, 23344840)
        G_values = D_scores_inv[self.all_h_list] * edge_alpha   # 与权重逐元素相乘 （23344840）

        return G_indices, G_values

    def inference(self):
        all_embeddings = [torch.concat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        gnn_embeddings = []
        int_embeddings = []
        gaa_embeddings = []
        iaa_embeddings = []

        for i in range(0, self.n_layers):

            # Graph-based Message Passing
            # z = AE，包括了人和物
            gnn_layer_embeddings = torch_sparse.spmm(self.G_indices, self.G_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])

            # Intent-aware Information Aggregation
            # 把人和物切割开处理(50821, 32)和(……， 32)
            u_embeddings, i_embeddings = torch.split(all_embeddings[i], [self.n_users, self.n_items], 0)

            # 利用其来获取意图编码
            # 矩阵乘法 (50821,32) @ (32, 128) -> （50821， 128）@ (128, 32) ->（50821， 32）
            u_int_embeddings = torch.softmax(u_embeddings @ self.user_intent, dim=1) @ self.user_intent.T
            i_int_embeddings = torch.softmax(i_embeddings @ self.item_intent, dim=1) @ self.item_intent.T
            # 再拼接
            int_layer_embeddings = torch.concat([u_int_embeddings, i_int_embeddings], dim=0)

            # Adaptive Augmentation
            # 自适应权重数据增强
            gnn_head_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_h_list)      # 依照所有非空元素的行找出emb，可重复([2344850, 32])
            # print(gnn_head_embeddings.shape)
            gnn_tail_embeddings = torch.index_select(gnn_layer_embeddings, 0, self.all_t_list)      # 依照非空元素的列找出emb
            int_head_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_h_list)      # 意图也是如此
            int_tail_embeddings = torch.index_select(int_layer_embeddings, 0, self.all_t_list)
            G_graph_indices, G_graph_values = self._adaptive_mask(gnn_head_embeddings, gnn_tail_embeddings)
            G_inten_indices, G_inten_values = self._adaptive_mask(int_head_embeddings, int_tail_embeddings)

            gaa_layer_embeddings = torch_sparse.spmm(G_graph_indices, G_graph_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])
            iaa_layer_embeddings = torch_sparse.spmm(G_inten_indices, G_inten_values, self.A_in_shape[0], self.A_in_shape[1], all_embeddings[i])

            gnn_embeddings.append(gnn_layer_embeddings)
            int_embeddings.append(int_layer_embeddings)
            gaa_embeddings.append(gaa_layer_embeddings)
            iaa_embeddings.append(iaa_layer_embeddings)

            all_embeddings.append(gnn_layer_embeddings + int_layer_embeddings + gaa_layer_embeddings + iaa_layer_embeddings + all_embeddings[i])

        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.sum(all_embeddings, dim=1, keepdim=False)

        self.ua_embedding, self.ia_embedding = torch.split(all_embeddings, [self.n_users, self.n_items], 0)

        return gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings

    def cal_ssl_loss(self, users, items, gnn_emb, int_emb, gaa_emb, iaa_emb):
        users = torch.unique(users)
        items = torch.unique(items)

        cl_loss = 0.0

        def cal_loss(emb1, emb2):
            pos_score = torch.exp(torch.sum(emb1 * emb2, dim=1) / self.temp)
            neg_score = torch.sum(torch.exp(torch.mm(emb1, emb2.T) / self.temp), axis=1)
            loss = torch.sum(-torch.log(pos_score / (neg_score + 1e-8) + 1e-8))
            loss /= pos_score.shape[0]
            return loss

        for i in range(len(gnn_emb)):
            u_gnn_embs, i_gnn_embs = torch.split(gnn_emb[i], [self.n_users, self.n_items], 0)
            u_int_embs, i_int_embs = torch.split(int_emb[i], [self.n_users, self.n_items], 0)
            u_gaa_embs, i_gaa_embs = torch.split(gaa_emb[i], [self.n_users, self.n_items], 0)
            u_iaa_embs, i_iaa_embs = torch.split(iaa_emb[i], [self.n_users, self.n_items], 0)

            u_gnn_embs = F.normalize(u_gnn_embs[users], dim=1)
            u_int_embs = F.normalize(u_int_embs[users], dim=1)
            u_gaa_embs = F.normalize(u_gaa_embs[users], dim=1)
            u_iaa_embs = F.normalize(u_iaa_embs[users], dim=1)

            i_gnn_embs = F.normalize(i_gnn_embs[items], dim=1)
            i_int_embs = F.normalize(i_int_embs[items], dim=1)
            i_gaa_embs = F.normalize(i_gaa_embs[items], dim=1)
            i_iaa_embs = F.normalize(i_iaa_embs[items], dim=1)

            cl_loss += cal_loss(u_gnn_embs, u_int_embs)
            cl_loss += cal_loss(u_gnn_embs, u_gaa_embs)
            cl_loss += cal_loss(u_gnn_embs, u_iaa_embs)

            cl_loss += cal_loss(i_gnn_embs, i_int_embs)
            cl_loss += cal_loss(i_gnn_embs, i_gaa_embs)
            cl_loss += cal_loss(i_gnn_embs, i_iaa_embs)

        return cl_loss

    def forward(self, users, pos_items, neg_items):
        users = torch.LongTensor(users).cuda()
        pos_items = torch.LongTensor(pos_items).cuda()
        neg_items = torch.LongTensor(neg_items).cuda()

        gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings = self.inference()

        # bpr
        u_embeddings = self.ua_embedding[users]
        pos_embeddings = self.ia_embedding[pos_items]
        neg_embeddings = self.ia_embedding[neg_items]
        pos_scores = torch.sum(u_embeddings * pos_embeddings, 1)
        neg_scores = torch.sum(u_embeddings * neg_embeddings, 1)
        mf_loss = torch.mean(F.softplus(neg_scores - pos_scores))

        # embeddings
        u_embeddings_pre = self.user_embedding(users)
        pos_embeddings_pre = self.item_embedding(pos_items)
        neg_embeddings_pre = self.item_embedding(neg_items)
        emb_loss = (u_embeddings_pre.norm(2).pow(2) + pos_embeddings_pre.norm(2).pow(2) + neg_embeddings_pre.norm(2).pow(2))
        emb_loss = self.emb_reg * emb_loss

        # intent prototypes
        cen_loss = (self.user_intent.norm(2).pow(2) + self.item_intent.norm(2).pow(2))
        cen_loss = self.cen_reg * cen_loss

        # self-supervise learning
        cl_loss = self.ssl_reg * self.cal_ssl_loss(users, pos_items, gnn_embeddings, int_embeddings, gaa_embeddings, iaa_embeddings)

        return mf_loss, emb_loss, cen_loss, cl_loss

    def predict(self, users):
        u_embeddings = self.ua_embedding[torch.LongTensor(users).cuda()]
        i_embeddings = self.ia_embedding
        batch_ratings = torch.matmul(u_embeddings, i_embeddings.T)
        return batch_ratings