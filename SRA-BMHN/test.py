import argparse
import torch
from tools import evaluate_results_nc
from pytorchtools import EarlyStopping
from load_data import load_data
import numpy as np
import random
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import psutil
import os
from model.model import Model
import warnings


warnings.filterwarnings('ignore')

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args):
    #ADJ: 是三类不同的关系列表，每个列表中存在多个矩阵，分别是：目标节点-目标节点，目标节点-其他节点，其他节点-其他节点
    ADJ, features, labels, num_classes, train_idx, val_idx, test_idx = load_data()
    input_dim = [i.shape[1] for i in features]
    same_type_num = len(ADJ[0])
    relation_num = len(ADJ[1])
    model = Model(input_dim,args["hidden_units"],num_classes,args["feat_drop"],same_type_num,relation_num)

    early_stopping = EarlyStopping(patience=args['patience'], verbose=True,save_path='checkpoint/checkpointTest_{}.pt'.format(args['dataset']))  # 提早停止，设置的耐心值为5
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],weight_decay=args['weight_decay'])
    for i in range(1):
        for epoch in range(args['num_epochs']):
            model.train()
            logits, h = model(features, ADJ)
            loss = loss_fcn(logits[train_idx], labels[train_idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model.eval()
            logits, h = model(features, ADJ)
            val_loss = loss_fcn(logits[val_idx], labels[val_idx])
            test_loss = loss_fcn(logits[test_idx], labels[test_idx])
            print('Epoch{:d}| Train Loss{:.4f}| Val Loss{:.4f}| Test Loss{:.4f}'.format(epoch + 1, loss.item(),val_loss.item(),test_loss.item()))
            early_stopping(val_loss.data.item(), model)
            if early_stopping.early_stop:
                print('Early stopping!')
                break
        print('\ntesting...')
        model.load_state_dict(torch.load('checkpoint/checkpointTest_{}.pt'.format(args['dataset'])))
        model.eval()
        logits, h = model(features, ADJ)
        Y = labels[test_idx].cpu().numpy()
        ml = TSNE(n_components=2)
        node_pos = ml.fit_transform(h[test_idx].detach().cpu().numpy())
        color_idx = {}
        for i in range(len(h[test_idx].detach().cpu().numpy())):
            color_idx.setdefault(Y[i], [])
            color_idx[Y[i]].append(i)
        for c, idx in color_idx.items():  # c是类型数，idx是索引
            if str(c) == '1':
                plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#DAA520', s=15, alpha=1)
            elif str(c) == '2':
                plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#8B0000', s=15, alpha=1)
            elif str(c) == '0':
                plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#6A5ACD', s=15, alpha=1)
            elif str(c) == '3':
                plt.scatter(node_pos[idx, 0], node_pos[idx, 1], c='#006400', s=15, alpha=1)
        plt.legend()
        # plt.savefig(".\\visualization\DBLP_" + str(args['dataset']) + "分类图" + str('1') + ".png", dpi=1000,
        #             bbox_inches='tight')
        plt.show()
        evaluate_results_nc(h[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(),int(labels.max()) + 1)

    #print("MA:{}\n MI:{}\n NMI:{}\n ARI:{}".format(MA,MI,NMI,ARI))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='CIKM2019', help='数据集')
    parser.add_argument('--lr', default=0.005, help='学习率')
    parser.add_argument('--num_heads', default=[8], help='多头注意力数及网络层数')
    parser.add_argument('--hidden_units', default=64, help='隐藏层数（实际隐藏层数：隐藏层数*注意力头数）')
    parser.add_argument('--dropout', default=0.6, help='丢弃率')
    parser.add_argument('--feat_drop', default=0.6, help='丢弃率')
    parser.add_argument('--sample_rate', default=[7,1], help='采样率')
    parser.add_argument('--num_epochs', default=1000, help='最大迭代次数')
    parser.add_argument('--weight_decay', default=0.0005, help='权重衰减')
    parser.add_argument('--patience', type=int, default=5, help='耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=100, help='重复训练和测试次数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

