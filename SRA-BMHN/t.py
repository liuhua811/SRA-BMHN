import numpy as np
import torch
import dgl
import torch.nn.functional as F
import torch as th
import scipy.sparse as sp

prefix=r'G:\OtherCode\__dataset\Double_Classes and Mul Relation\AL_CIKM2019'

user_item_buy = np.load(prefix+"/connection/nei_item_user_buy.npy",allow_pickle=True)
user_item_buy = [th.LongTensor(i) for i in user_item_buy]
print(user_item_buy)
