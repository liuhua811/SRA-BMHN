import numpy as np
import scipy.sparse as sp

####################################################
# This tool is to collect neighbors, and reform them
# as numpy.array style for futher usage.
####################################################

# This is for DBLP
pa = np.genfromtxt("item_user_pv.txt")
a_n = {}  # a_n是个字典{262: [0], 263: [1, 2, 5868, 5870, 5871, 5872, 5874, 5876, 5878, 5880, 7467]...}
for i in pa:
    if i[1] not in a_n:
        a_n[int(i[1])] = []
        a_n[int(i[1])].append(int(i[0]))
    else:
        a_n[int(i[1])].append(int(i[0]))

keys = sorted(a_n.keys())  # keys是a_n字典中所有的键
a_n = [a_n[i] for i in
       keys]  # 通过for循环遍历键，取出了每个键对应的值（列表格式），放到一个列表里 现在格式为：[[2364, 6457], [2365, 2366, 2389, 2431, 2461, 2463, 2475, 2482, 5424, 6244]...]
a_n = np.array([np.array(i) for i in
                a_n])  # 把里面的小列表变成np格式，再把整个大列表变成np格式 现在格式为：[[2364, 6457], [2365, 2366, 2389, 2431, 2461, 2463, 2475, 2482, 5424, 6244]...]
np.save("item_user_pv.npy", a_n)


