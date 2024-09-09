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
    # 初始化时间和内存统计变量
    total_time = 0.0
    total_memory = 0.0
    epoch_count = 0

    # 训练模型
    for epoch in range(args['num_epochs']):
        start_time = time.time()  # 开始计时
        model.train()
        logits, h_list = model(features, ADJ)
        loss = loss_fcn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        logits, h_list = model(features, ADJ)
        val_loss = loss_fcn(logits[val_idx], labels[val_idx])
        test_loss = loss_fcn(logits[test_idx], labels[test_idx])

        # 计算每轮的时间
        end_time = time.time()
        epoch_time = end_time - start_time
        total_time += epoch_time

        # 获取当前内存使用情况
        memory_used = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2  # 转换为MB
        total_memory += memory_used
        epoch_count += 1

        print(
            'Epoch {:d} | Train Loss {:.4f} | Val Loss {:.4f} | Test Loss {:.4f} | Time {:.2f}s | Memory {:.2f}MB'.format(
                epoch + 1, loss.item(), val_loss.item(), test_loss.item(), epoch_time, memory_used))

        early_stopping(val_loss.data.item(), model)
        if early_stopping.early_stop:
            print('Early stopping!')
            break

    # 输出平均时间和内存
    avg_time = total_time / epoch_count
    avg_memory = total_memory / epoch_count
    print('Average Time per Epoch: {:.2f}s'.format(avg_time))
    print('Average Memory Usage per Epoch: {:.2f}MB'.format(avg_memory))

    # 测试模型
    print('\nTesting...')
    model.load_state_dict(torch.load('checkpoint/checkpointTest_{}.pt'.format(args['dataset'])))
    model.eval()
    logits, h_list = model(features, ADJ)

    evaluate_results_nc(h_list[test_idx].detach().cpu().numpy(), labels[test_idx].cpu().numpy(), int(labels.max()) + 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='这是我们基于GAT所构建的HAN模型')
    parser.add_argument('--dataset', default='CIKM2019', help='数据集')
    parser.add_argument('--prefix', default='data/', help='数据文件前缀路径')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--num_heads', type=list, default=[8], help='多头注意力数及网络层数')
    parser.add_argument('--hidden_units', type=int, default=64, help='隐藏层数（实际隐藏层数：隐藏层数*注意力头数）')
    parser.add_argument('--dropout', type=float, default=0.6, help='丢弃率')
    parser.add_argument('--feat_drop', type=float, default=0.6, help='丢弃率')
    parser.add_argument('--sample_rate', type=list, default=[7, 1], help='采样率')
    parser.add_argument('--num_epochs', type=int, default=1000, help='最大迭代次数')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='权重衰减')
    parser.add_argument('--patience', type=int, default=10, help='耐心值')
    parser.add_argument('--device', type=str, default='cpu', help='使用cuda:0或者cpu')
    parser.add_argument('--repeat', type=int, default=100, help='重复训练和测试次数')
    parser.add_argument('--seed', type=int, default=0, help='随机种子')
    args = parser.parse_args().__dict__
    set_random_seed(args['seed'])
    print(args)
    main(args)

