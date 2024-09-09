import numpy as np
import scipy.sparse as sp
prefix=r'D:\桌面文件\交叉实验'

#生成训练集-验证集-测试集的代码
#4000 十分之一训练集  十分之一验证集  剩下测试集
# 生成包含序号的数组
data = np.arange(0, 9089)

# 将数据打乱
np.random.shuffle(data)

# 划分数据集
train_idx = data[:910]
val_idx = data[910:1820]
test_idx = data[1820:]
t = sp.coo_matrix(train_idx)

# 保存为.npz文件
np.savez(prefix+'/train_val_test_idx4.npz', train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)


