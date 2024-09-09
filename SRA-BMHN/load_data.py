import numpy as np
import scipy
import pickle
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp


def load_data(prefix=r'D:\桌面文件\模型\阿里复现数据集\三分类'):
    features_0 = scipy.sparse.load_npz(prefix + '/product_feature.npz').toarray()  # features_0为商品特征 features_1为用户特征
    features_1 = scipy.sparse.load_npz(prefix + '/user_feature.npz').toarray()
    features_0 = torch.FloatTensor(features_0)
    features_1 = torch.FloatTensor(features_1)
    features = [features_0, features_1]

    labels = np.load(prefix + '/labels.npy')

    labels = torch.LongTensor(labels).flatten()

    train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
    train_idx = train_val_test_idx['train_idx']
    val_idx = train_val_test_idx['val_idx']
    test_idx = train_val_test_idx['test_idx']
    num_classes = 3

    # meta_path
    item_buy_user_item = scipy.sparse.load_npz(prefix + '/item_buy_user_item.npz').toarray()
    item_cart_user_item = scipy.sparse.load_npz(prefix + '/item_cart_user_item.npz').toarray()
    item_pav_user_item = scipy.sparse.load_npz(prefix + '/item_fav_user_item.npz').toarray()
    item_pv_user_item = scipy.sparse.load_npz(prefix + '/item_pv_user_item.npz').toarray()
    meta_path = [item_buy_user_item, item_cart_user_item, item_pav_user_item, item_pv_user_item]
    meta_path = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in meta_path]

    item_item_brand = sp.load_npz(prefix + "/item_item_brand.npz").toarray()
    item_item_shop = sp.load_npz(prefix + "/item_item_shop.npz").toarray()
    # item_item = item_item_brand + item_item_shop
    # item_item = dgl.DGLGraph(item_item)
    #下面是item_user 文件命名有错误
    user_item_buy = sp.load_npz(prefix + "/adj_user_item_buy.npz").toarray()
    user_item_cart = sp.load_npz(prefix + "/adj_user_item_cart.npz").toarray()
    user_item_pav = sp.load_npz(prefix + "/adj_user_item_fav.npz").toarray()
    user_item_pv = sp.load_npz(prefix + "/adj_user_item_pv.npz").toarray()


    # user_user_age = sp.load_npz(prefix + "/user_user_age.npz").toarray()
    # user_user_gender = sp.load_npz(prefix + "/user_user_gender.npz").toarray()
    # user_user_purchase_power = sp.load_npz(prefix + "/user_user_purchase_power.npz").toarray()

    #user_user = [user_user_age, user_user_gender, user_user_purchase_power]


    item_user = [user_item_buy, user_item_cart, user_item_pav, user_item_pv]
    item_item = [item_item_brand, item_item_shop]
    #user_user = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in user_user]
    item_user = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in item_user]
    item_item = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in item_item]
    #网络模式的连接，heco格式
    # item_user_buy = np.load(prefix + "/connection/nei_item_user_buy.npy", allow_pickle=True)
    # item_user_cart = np.load(prefix + "/connection/nei_item_user_cart.npy", allow_pickle=True)
    # item_user_pav = np.load(prefix + "/connection/nei_item_user_pav.npy", allow_pickle=True)
    # item_user_pv = np.load(prefix + "/connection/nei_item_user_pv.npy", allow_pickle=True)
    # item_user_buy = [th.LongTensor(i) for i in item_user_buy]
    # item_user_cart = [th.LongTensor(i) for i in item_user_cart]
    # item_user_pav = [th.LongTensor(i) for i in item_user_pav]
    # item_user_pv = [th.LongTensor(i) for i in item_user_pv]
    # item_user = [item_user_buy,item_user_cart,item_user_pav,item_user_pv]

    ADJ = [item_item,meta_path,item_user]
    return ADJ,features, labels, num_classes, train_idx, val_idx, test_idx

# def load_data_dblp(prefix=r'D:\桌面文件\dblp_3'):
#     features_0 = scipy.sparse.load_npz(prefix + '/paper_features.npz').toarray()  # features_0为商品特征 features_1为用户特征
#     features_1 = scipy.sparse.load_npz(prefix + '/author_features.npz').toarray()
#     features_0 = torch.FloatTensor(features_0)
#     features_1 = torch.FloatTensor(features_1)
#     features = [features_0,features_1]
#
#     labels = np.load(prefix + '/labels.npy')
#
#     labels = torch.LongTensor(labels).flatten()
#
#     train_val_test_idx = np.load(prefix + '/train_val_test_idx1.npz')
#     train_idx = train_val_test_idx['train_idx']
#     val_idx = train_val_test_idx['val_idx']
#     test_idx = train_val_test_idx['test_idx']
#     num_classes = 3
#
#     # meta_path
#     P_A_P_important = scipy.sparse.load_npz(prefix + '/P_A_P_important.npz').toarray()
#     P_A_P_ordinary = scipy.sparse.load_npz(prefix + '/P_A_P_ordinary.npz').toarray()
#     meta_path = [P_A_P_important,P_A_P_ordinary]
#     meta_path = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in meta_path]
#     P_P_cite = sp.load_npz(prefix + "/P_P_cite.npz").toarray()
#     P_P_coauthor = sp.load_npz(prefix + "/P_P_coauthor.npz").toarray()
#     P_P_covenue = sp.load_npz(prefix + "/P_P_covenue.npz").toarray()
#     P_A_important = sp.load_npz(prefix + "/P_A_important.npz").toarray()
#     P_A_ordinary = sp.load_npz(prefix + "/P_A_ordinary.npz").toarray()
#     P_A = [P_A_important,P_A_ordinary]
#     P_P = [P_P_cite,P_P_coauthor,P_P_covenue]
#     # user_user_age = sp.load_npz(prefix + "/user_user_age.npz").toarray()
#     # user_user_gender = sp.load_npz(prefix + "/user_user_gender.npz").toarray()
#     # user_user_purchase_power = sp.load_npz(prefix + "/user_user_purchase_power.npz").toarray()
#     # user_user = [user_user_age, user_user_gender, user_user_purchase_power]
#
#
#
#
#     # user_user = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in user_user]
#     P_A = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in P_A]
#     P_P = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in P_P]
#
#     ADJ = [P_P,meta_path,P_A]
#     return ADJ,features, labels, num_classes, train_idx, val_idx, test_idx


####以下是GCN和GAT
# def load_data_dblp(prefix=r'D:\桌面文件\dblp_3'):
#     features_0 = scipy.sparse.load_npz(prefix + '/paper_features.npz').toarray()  # features_0为商品特征 features_1为用户特征
#     features_1 = scipy.sparse.load_npz(prefix + '/author_features.npz').toarray()
#     features_0 = torch.FloatTensor(features_0)
#     features_1 = torch.FloatTensor(features_1)
#     features = [features_0]
#
#     labels = np.load(prefix + '/labels.npy')
#
#     labels = torch.LongTensor(labels).flatten()
#
#     train_val_test_idx = np.load(prefix + '/train_val_test_idx1.npz')
#     train_idx = train_val_test_idx['train_idx']
#     val_idx = train_val_test_idx['val_idx']
#     test_idx = train_val_test_idx['test_idx']
#     num_classes = 3
#
#     # meta_path
#     # P_A_P_important = scipy.sparse.load_npz(prefix + '/P_A_P_important.npz').toarray()
#     # P_A_P_ordinary = scipy.sparse.load_npz(prefix + '/P_A_P_ordinary.npz').toarray()
#     # meta_path = [P_A_P_important,P_A_P_ordinary]
#     # meta_path = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in meta_path]
#     P_P_cite = sp.load_npz(prefix + "/P_P_cite.npz").toarray()
#     P_P_coauthor = sp.load_npz(prefix + "/P_P_coauthor.npz").toarray()
#     P_P_covenue = sp.load_npz(prefix + "/P_P_covenue.npz").toarray()
#     # P_A_important = sp.load_npz(prefix + "/P_A_important.npz").toarray()
#     # P_A_ordinary = sp.load_npz(prefix + "/P_A_ordinary.npz").toarray()
#     # P_A = [P_A_important,P_A_ordinary]
#     P_P = P_P_cite + P_P_coauthor + P_P_covenue
#
#     # user_user_age = sp.load_npz(prefix + "/user_user_age.npz").toarray()
#     # user_user_gender = sp.load_npz(prefix + "/user_user_gender.npz").toarray()
#     # user_user_purchase_power = sp.load_npz(prefix + "/user_user_purchase_power.npz").toarray()
#     # user_user = [user_user_age, user_user_gender, user_user_purchase_power]
#
#     P_P_sparse = sp.csr_matrix(P_P)
#
#     # 使用 dgl.from_scipy 转换为 DGL 图对象
#     P_P = dgl.from_scipy(P_P_sparse)
#     P_P = dgl.add_self_loop(P_P)
#
#
#     # user_user = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in user_user]
#     # P_A = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in P_A]
#     # P_P = [F.normalize(torch.FloatTensor(i), dim=1, p=2) for i in P_P]
#     ADJ = P_P
#     return ADJ,features, labels, num_classes, train_idx, val_idx, test_idx
#
#
# 以下GCN、GAT
# def load_data(prefix=r'D:\桌面文件\模型\阿里复现数据集\三分类'):
#
#         # 加载商品和用户特征
#         features_0 = scipy.sparse.load_npz(prefix + '/product_feature.npz').toarray()  # features_0为商品特征 features_1为用户特征
#         features_1 = scipy.sparse.load_npz(prefix + '/user_feature.npz').toarray()
#         features_0 = torch.FloatTensor(features_0)
#         features_1 = torch.FloatTensor(features_1)
#         features = [features_0]
#
#         labels = np.load(prefix + '/labels.npy')
#
#         labels = torch.LongTensor(labels)
#
#         # 加载训练、验证、测试索引
#         train_val_test_idx = np.load(prefix + '/train_val_test_idx.npz')
#         train_idx = train_val_test_idx['train_idx']
#         val_idx = train_val_test_idx['val_idx']
#         test_idx = train_val_test_idx['test_idx']
#
#         # 加载关系矩阵，并相加为一个整体的矩阵
#         # 加载 item_item 数据并转换为稀疏矩阵
#         item_item_brand = sp.load_npz(prefix + "/item_item_brand.npz").toarray()
#         item_item_shop = sp.load_npz(prefix + "/item_item_shop.npz").toarray()
#         item_item = item_item_brand + item_item_shop
#
#         # 转换为稀疏矩阵
#         item_item_sparse = sp.csr_matrix(item_item)
#
#         # 使用 dgl.from_scipy 转换为 DGL 图对象
#         item_item = dgl.from_scipy(item_item_sparse)
#         item_item = dgl.add_self_loop(item_item)
#
#         # 将 item_item_dgl 放入 ADJ 列表中，以便在模型中使用
#         ADJ = item_item
#
#         num_classes = 3  # 设定类别数
#
#         return ADJ, features, labels, num_classes, train_idx, val_idx, test_idx


if __name__ == "__main__":
    load_data()
