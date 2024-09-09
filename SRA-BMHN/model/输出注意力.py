import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Classes_Relation_AGG import Classes_Relation_AGG
from model.Dif_Classes_AGG import Dif_Classes_AGG
from model.Weight_Schema import Weight_Schema

class AggAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(AggAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).squeeze(-1)
        beta = torch.softmax(w, dim=1)
        beta = beta.unsqueeze(-1)
        return (beta * z).sum(1), beta.squeeze(-1)


class Model(nn.Module):
    def __init__(self, input_dim, hid_dim, out_size, feat_drop, same_type_num, relation_num):
        '''
        :param same_type_num: 目标节点之间的关系数量
        :param relation_num: 目标节点与其他节点之间的关系数量
        '''
        super(Model, self).__init__()
        self.dropout = nn.Dropout(p=feat_drop)
        self.non_linear = nn.Tanh()
        self.feat_mapping = nn.ModuleList([nn.Linear(m, hid_dim, bias=True) for m in input_dim])
        self.project2 = nn.Linear(hid_dim, out_size)
        # 1.异类型节点不同关系间的聚合
        self.meta_rela_agg = Dif_Classes_AGG(hid_dim, relation_num, feat_drop)
        # 2.二部子图间聚合
        self.weight_schema_agg = Weight_Schema(hid_dim)
        # 3.同类型节点不同关系间的聚合
        self.item_classes_AGG = Classes_Relation_AGG(hid_dim, feat_drop, same_type_num)
        self.attention = AggAttention(hid_dim)

    def forward(self, features, ADJ):
        '''
        :param features: 目标节点特征，其他节点特征(商品，用户）
        :param ADJ: 是三类不同的关系列表，每个列表中存在多个矩阵，分别是：目标节点-目标节点，目标节点-其他节点，其他节点-其他节点
                    item_item, meta_path, user_item
        :return:
        '''
        h = [self.non_linear(self.feat_mapping[i](features[i])) for i in range(len(features))]  # 映射到相同特征空间
        # 第一步：元关系路径聚合
        relation_agg = self.dropout(self.meta_rela_agg(h, ADJ[1]))  # 此处带dropout比不带效果好
        # 第二步：构建二部子图，item吸收带权重的user信息
        item_classes_agg = self.weight_schema_agg(h[1], ADJ[2])  # 带权重的GCN聚合, 不带dropout效果好
        # 第三步：对吸收后的带权重的item特征，执行item间的信息聚合。
        item_item = self.dropout(self.item_classes_AGG(item_classes_agg, ADJ[0]))  # 带dropout效果好

        # 语义注意力聚合
        H = [relation_agg, item_item]
        H = torch.stack(H, dim=1)
        H = self.attention(H)  # 语义级注意力聚合
        # 相加聚合
        # H = 0.7*self.non_linear(relation_agg) + 0.3*item_item
        # H = 0.7*self.non_linear(relation_agg) + 0.3*item_item + h[0]
        return self.project2(H), H
