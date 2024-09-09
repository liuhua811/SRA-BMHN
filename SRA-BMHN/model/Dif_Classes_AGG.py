import torch
import torch.nn as nn
import torch.nn.functional as F
from .Classes_Relation_AGG import AggAttention,GCN
import numpy as np

class Dif_Classes_AGG(nn.Module):
    '''
    1.先将自身特征投影到特定空间
    2.再将特定关系的特征投影到特定空间
    3.对特定关系进行注意力聚合，得到多个特定关系的矩阵特征
    4.对不同关系矩阵特征使用注意力进行聚合
    '''
    def __init__(self,hid_dim,relation_num,drop):
        '''
        :param relation_num: 目标类型-其他类型的关系数量
        '''
        super(Dif_Classes_AGG, self).__init__()
        self.non_linear = nn.Tanh()
        self.relation_num = relation_num
        self.project_target = nn.Linear(hid_dim,hid_dim)    #先将自身类型投影到特定空间
        self.project_difRelation = nn.ModuleList([nn.Linear(hid_dim,hid_dim) for _ in range(relation_num)])     #对不同类型的边邻居节点投影到不同的空间
        self.intra_att = nn.ModuleList([GCN(hid_dim) for _ in range(relation_num)])          #将邻居节点按注意力机制聚合，获得不同关系的关系特征矩阵
        self.inter = AggAttention(hid_dim,name="关系聚合")             #将不同关系特征矩阵按注意力机制进行聚合
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,features,adj):
        '''
        :param features: 所有类型节点特征，features[0]为目标节点类型，features[1]为其他类型节点特征
        :param adj:列表内为目标类型-其他类型 不同关系的邻接矩阵
        '''
        target_feat = self.project_target(features[0])    #得到目标节点的特征映射
        relation_feat = [self.non_linear(self.project_difRelation[i](features[0])) for i in range(self.relation_num)]     #得到各个关系的特征映射
        # 将目标节点特征、特定关系特征、特定关系邻接矩阵 传入不同的注意力机制中，得到关系特征矩阵列表
        relation_matrix = [self.intra_att[i](relation_feat[i],adj[i])  for i in range(self.relation_num)]
        relation_agg = self.dropout(self.inter(torch.stack(relation_matrix,dim=1)))
        return relation_agg +target_feat





# class AttentionAggregator(nn.Module):
#     def __init__(self, in_features):
#         super(AttentionAggregator, self).__init__()
#         self.W = nn.Linear(in_features, in_features)
#         self.a = nn.Linear(2*in_features, 1)
#
#     def forward(self, H, Z, A):
#         # 计算注意力权重
#         Wh = self.W(H)
#         Wz = self.W(Z)
#         e = self.attention_score(Wh, Wz, A)
#
#         # 使用注意力权重进行聚合
#         alpha = F.softmax(e, dim=1)
#         h_agg = torch.matmul(alpha, Z)
#
#         return h_agg
#
#     def attention_score(self, Wh, Wz, A):
#         # 计算注意力得分
#         a_input = torch.cat([Wh, Wz], dim=1)
#         e = self.a(a_input).squeeze(-1)
#         e = torch.mul(e, A)  # 将注意力矩阵A作为mask
#         return e
        '''
        掩码操作
        e = e.masked_fill(A == 0, -1e9) # shape: (N, M)
        '''





'''
下面是heco对邻居节点的聚合方式
'''
# GAT============计算节点P和邻居节点A1 A2 A3 A4 的注意力
# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_dim, attn_drop):
#         super(AttentionLayer, self).__init__()
#         self.att = nn.Parameter(torch.empty(size=(1, 2*hidden_dim)), requires_grad=True)
#         nn.init.xavier_normal_(self.att.data, gain=1.414)
#         if attn_drop:
#             self.attn_drop = nn.Dropout(attn_drop)
#         else:
#             self.attn_drop = lambda x: x
#
#         self.softmax = nn.Softmax(dim=1)
#         self.leakyrelu = nn.LeakyReLU()
#
#     def forward(self, nei, h, h_refer):
#         nei_emb = F.embedding(nei, h)
#         h_refer = torch.unsqueeze(h_refer, 1)
#         h_refer = h_refer.expand_as(nei_emb)
#         all_emb = torch.cat([h_refer, nei_emb], dim=-1)
#         attn_curr = self.attn_drop(self.att)
#         att = self.leakyrelu(all_emb.matmul(attn_curr.t()))
#         att = self.softmax(att)
#         nei_emb = (att*nei_emb).sum(dim=1)
#         return nei_emb
#
# class inter_att(nn.Module):
#     def __init__(self, hidden_dim, attn_drop):
#         super(inter_att, self).__init__()
#         self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
#         nn.init.xavier_normal_(self.fc.weight, gain=1.414)
#
#         self.tanh = nn.Tanh()
#         self.att = nn.Parameter(torch.empty(size=(1, hidden_dim)), requires_grad=True)
#         nn.init.xavier_normal_(self.att.data, gain=1.414)
#
#         self.softmax = nn.Softmax(dim=0)
#         if attn_drop:
#             self.attn_drop = nn.Dropout(attn_drop)
#         else:
#             self.attn_drop = lambda x: x
#
#     def forward(self, embeds):
#         beta = []
#         attn_curr = self.attn_drop(self.att)
#         for embed in embeds:
#             sp = self.tanh(self.fc(embed)).mean(dim=0)
#             beta.append(attn_curr.matmul(sp.t()))
#         beta = torch.cat(beta, dim=-1).view(-1)
#         beta = self.softmax(beta)
#         z_mc = 0
#         for i in range(len(embeds)):
#             z_mc += embeds[i] * beta[i]
#         return z_mc
#
#
#




