import heapq
import logging
import math
import warnings
from abc import abstractmethod, ABCMeta
import random
from functools import wraps

import networkx as nx
import numpy as np
import collections
import torch
from torch.utils.data import ConcatDataset
from tqdm import tqdm

try:
    import community.community_louvain
except:
    pass

def classify_label(dataset, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, datum in enumerate(dataset):
        list1[datum[1]].append(idx)
    return list1
index_func=lambda X:[xi[-1] for xi in X]

# 装饰器函数
def calculate_sample_stats(func):
    @wraps(func)
    def wrapper(self, data):
        local_datas = func(self, data)
        distribution = []
        for local_data in local_datas:
            distribution.append(data.get_distribution(local_data))
        X = [[data[idx][0] for idx in idxes] for c, idxes in enumerate(local_datas)]
        Y = [[data[idx][1] for idx in idxes] for c, idxes in enumerate(local_datas)]
        # 返回 local_datas 和统计量
        return (X, Y), distribution
    return wrapper


class AbstractPartitioner(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class BasicPartitioner(AbstractPartitioner):
    """This  is the basic class of data partitioner. The partitioner will be directly called by the
    task generator of different benchmarks. By overwriting __call__ method, different partitioners
    can be realized. The input of __call__ is usually a dataset.
    """
    def __init__(self, num_clients=100, num_per=None):
        self.num_per = num_per
        self.num_clients = num_clients
        return

    def get_info(self):
        class_name = self.__class__.__name__
        attributes = {}
        for attr in vars(self):
            value = getattr(self, attr)
            if value:
                attributes[attr](str(value))
        attributes_str = '_'.join(f'{k}={v}' for k, v in attributes.items())
        return f"{class_name}_{attributes_str}"

    def data_imbalance_generator(self, num_clients, datasize, imbalance=0, minvol=1):
        r"""
        Split the data size into several parts

        Args:
            num_clients (int): the number of clients
            datasize (int): the total data size
            imbalance (float): the degree of data imbalance across clients
            minvol (int): the minimal size of dataset
        Returns:
            a list of integer numbers that represents local data sizes
        """
        if imbalance == 0:
            samples_per_client = [int(datasize / num_clients) for _ in range(num_clients)]
            for _ in range(datasize % num_clients): samples_per_client[_] += 1
        else:
            imbalance = max(0.1, imbalance)
            sigma = imbalance
            mean_datasize = datasize / num_clients
            mu = np.log(mean_datasize) - sigma ** 2 / 2.0
            samples_per_client = np.random.lognormal(mu, sigma, (num_clients)).astype(int)
            crt_data_size = sum(samples_per_client)
            total_delta = np.abs(crt_data_size-datasize)
            thresold = max(int(total_delta/10), 1)
            delta = max(min(int(0.1 * thresold), 10), 1)
            # force current data size to match the total data size
            while crt_data_size != datasize:
                if crt_data_size - datasize >= thresold:
                    maxid = np.argmax(samples_per_client)
                    maxvol = samples_per_client[maxid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while min(new_samples) > maxvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[maxid] + s - datasize) for s in new_samples])
                    samples_per_client[maxid] = new_samples[new_size_id]
                elif crt_data_size - datasize >= delta:
                    maxid = np.argmax(samples_per_client)
                    if samples_per_client[maxid]>=delta:
                        samples_per_client[maxid] -= delta
                    elif samples_per_client[maxid]>1:
                        samples_per_client[maxid] -= 1
                elif crt_data_size - datasize > 0:
                    maxid = np.argmax(samples_per_client)
                    crt_delta = (crt_data_size - datasize)
                    if samples_per_client[maxid]>=crt_delta:
                        samples_per_client[maxid] -= crt_delta
                    elif samples_per_client[maxid]>=minvol:
                        samples_per_client[maxid] -= (crt_delta-minvol)
                    else:
                        warnings.warn("Failed to keep the minvol of clients' training data to be larger than {}".format(minvol))
                        if samples_per_client[maxid] > 1:
                            samples_per_client[maxid] -=1
                        else:
                            raise RuntimeError("Failed to generate distribution due to the conflicts of imbalance and num_clients. Please try to decrease the imbalance term or decrease the number of clients. ")
                elif datasize - crt_data_size >= thresold:
                    minid = np.argmin(samples_per_client)
                    minvol = samples_per_client[minid]
                    new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    while max(new_samples) < minvol:
                        new_samples = np.random.lognormal(mu, sigma, (10 * num_clients))
                    new_size_id = np.argmin(
                        [np.abs(crt_data_size - samples_per_client[minid] + s - datasize) for s in new_samples])
                    samples_per_client[minid] = new_samples[new_size_id]
                elif datasize - crt_data_size >= delta:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += delta
                else:
                    minid = np.argmin(samples_per_client)
                    samples_per_client[minid] += (datasize - crt_data_size)
                crt_data_size = sum(samples_per_client)
            # let the minimal data size to be larger than 0
            while min(samples_per_client)==0:
                zero_client_idx = np.argmin(samples_per_client)
                maxid = np.argmax(samples_per_client)
                samples_per_client[maxid] -=1
                samples_per_client[zero_client_idx] += 1
            assert datasize==sum(samples_per_client) and min(samples_per_client)>0
        return samples_per_client

class IIDPartitioner(BasicPartitioner):
    sign = 'IID'
    def __init__(self, imbalance=0, **kwargs):
        super().__init__(**kwargs)
        self.imbalance = imbalance

    def __str__(self):
        name = "iid"
        if self.imbalance > 0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    @calculate_sample_stats
    def __call__(self, data):
        if self.num_per is None:
            self.num_per = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)

        d_idxs = np.random.permutation(len(data))
        local_datas = np.split(d_idxs, np.cumsum(self.num_per))[:-1]
        local_datas = [di.tolist() for di in local_datas]
        return local_datas

class DirichletPartitioner(BasicPartitioner):
    sign = 'Dir'
    def __init__(self, alpha=1.0, error_bar=1e-6, imbalance=0, minvol=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.imbalance = imbalance
        self.index_func = index_func
        self.minvol = minvol
        self.error_bar = error_bar

    def __str__(self):
        name = "dir{:.2f}_err{}".format(self.alpha, self.error_bar)
        if self.imbalance > 0: name += '_imb{:.1f}'.format(self.imbalance)
        return name

    @calculate_sample_stats
    def __call__(self, data):
        global alter_norms
        attrs = self.index_func(data)
        if self.num_per is None:
            self.num_per = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance, minvol=self.minvol)
        # count the label distribution
        lb_counter = collections.Counter(attrs)
        lb_names = list(lb_counter.keys())
        p = np.array([1.0 * v / len(data) for v in lb_counter.values()])
        lb_dict = {}
        attrs = np.array(attrs)
        for lb in lb_names:
            lb_dict[lb] = np.where(attrs == lb)[0]
        proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        while np.any(np.isnan(proportions)):
            proportions = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
        sorted_cid_map = {k: i for k, i in zip(np.argsort(self.num_per), [_ for _ in range(self.num_clients)])}
        error_increase_interval = 500
        max_error = self.error_bar
        loop_count = 0
        crt_id = 0
        crt_error = 100000
        while True:
            if loop_count >= error_increase_interval:
                loop_count = 0
                max_error = max_error * 10
            # generate dirichlet distribution till ||E(proportion) - P(D)||<=1e-5*self.num_classes
            mean_prop = np.sum([pi * di for pi, di in zip(proportions, self.num_per)], axis=0)
            mean_prop = mean_prop / mean_prop.sum()
            error_norm = ((mean_prop - p) ** 2).sum()
            if crt_error - error_norm >= max_error:
                print("Approximation Error: {:.8f}".format(error_norm))
                crt_error = error_norm
            if error_norm <= max_error:
                break
            excid = sorted_cid_map[crt_id]
            crt_id = (crt_id + 1) % self.num_clients
            sup_prop = [np.random.dirichlet(self.alpha * p) for _ in range(self.num_clients)]
            del_prop = np.sum([pi * di for pi, di in zip(proportions, self.num_per)], axis=0)
            del_prop -= self.num_per[excid] * proportions[excid]
            for i in range(error_increase_interval - loop_count):
                alter_norms = []
                for cid in range(self.num_clients):
                    if np.any(np.isnan(sup_prop[cid])):
                        continue
                    alter_prop = del_prop + self.num_per[excid] * sup_prop[cid]
                    alter_prop = alter_prop / alter_prop.sum()
                    error_alter = ((alter_prop - p) ** 2).sum()
                    alter_norms.append(error_alter)
                if min(alter_norms) < error_norm:
                    break
            if len(alter_norms) > 0 and min(alter_norms) < error_norm:
                alcid = np.argmin(alter_norms)
                proportions[excid] = sup_prop[alcid]
            loop_count += 1
        local_datas = [[] for _ in range(self.num_clients)]
        self.dirichlet_dist = []  # for efficiently visualizing
        for lb in lb_names:
            lb_idxs = lb_dict[lb]
            lb_proportion = np.array([pi[lb_names.index(lb)] * si for pi, si in zip(proportions, self.num_per)])
            lb_proportion = lb_proportion / lb_proportion.sum()
            lb_proportion = (np.cumsum(lb_proportion) * len(lb_idxs)).astype(int)[:-1]
            lb_datas = np.split(lb_idxs, lb_proportion)
            self.dirichlet_dist.append([len(lb_data) for lb_data in lb_datas])
            local_datas = [local_data + lb_data.tolist() for local_data, lb_data in zip(local_datas, lb_datas)]
        self.dirichlet_dist = np.array(self.dirichlet_dist).T
        for i in range(self.num_clients): np.random.shuffle(local_datas[i])
        len_dist = [len(d) for d in local_datas]
        while min(len_dist)<=self.minvol:
            min_did = np.argmin(len_dist)
            max_did = np.argmax(len_dist)
            max_d = local_datas[max_did]
            min_d = local_datas[min_did]
            if len(max_d)<=self.minvol:
                raise RuntimeError("The number of clients is too large to distribute enough samples to each client when minvol=={}. Please decrease the number of clients".format(self.minvol))
            min_d.extend(max_d[:1])
            max_d = max_d[1:]
            local_datas[min_did] = min_d
            local_datas[max_did] = max_d
            len_dist = [len(d) for d in local_datas]
        self.local_datas = local_datas
        return local_datas

class SimpleDirichletPartitioner(BasicPartitioner):
    """`Partition the indices of samples in the original dataset according to Dirichlet distribution of the
    particular attribute. This way of partition is widely used by existing works in federated learning.

    Args:
        num_clients (int, optional): the number of clients
        alpha (float, optional): `alpha`(i.e. alpha>=0) in Dir(alpha*p) where p is the global distribution. The smaller alpha is, the higher heterogeneity the data is.
        imbalance (float, optional): the degree of imbalance of the amounts of different local data (0<=imbalance<=1)
        error_bar (float, optional): the allowed error when the generated distribution mismatches the distirbution that is actually wanted, since there may be no solution for particular imbalance and alpha.
        index_func (func, optional): to index the distribution-dependent (i.e. label) attribute in each sample.
    """
    sign = 'SimDir'
    def __init__(self, alpha=1.0, minvol=10, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.minvol = minvol

    def __str__(self):
        name = "dir{:.2f}_err".format(self.alpha)
        return name

    @calculate_sample_stats
    def __call__(self, data, samples_per_client=None):
        global idx_per_client
        attrs = index_func(data)
        num_attrs = len(set(attrs))
        num_labels = len(attrs)
        alpha = self.alpha
        min_size = 0
        attrs = np.array(attrs)
        while min_size < self.minvol:
            idx_per_client = [[] for _ in range(self.num_clients)] # data sample indices per client
            for k in range(num_attrs):
                idx_k = np.where(attrs == k)[0] # data sample indices of class k
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
                # Note: Balance the number of data samples of clients.
                # Don't assign samples to client j when its number of data samples is larger than the average (yipeng, 2023-11-14)
                proportions = np.array([p * (len(idx_j) < num_labels / self.num_clients) for p, idx_j in zip(proportions, idx_per_client)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in zip(idx_per_client, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        local_datas = []
        for j in range(self.num_clients):
            np.random.shuffle(idx_per_client[j])
            local_datas.append(idx_per_client[j])
        self.local_datas = local_datas

        return local_datas

class ExDirichletPartitioner(BasicPartitioner):
    sign = 'ExDir'
    def __init__(self, alpha=1.0, cls_per=-1, minvol=10, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.cls_per = cls_per
        self.index_func = index_func
        self.minvol = minvol

    def __str__(self):
        name = "dir{:.2f}_err".format(self.alpha)
        return name

    def allocate_classes(self, num_clients, num_classes):
        '''Allocate `C` classes to each client
        Returns:
            clientidx_map (dict): { class id (int): client indices (list) }
        '''
        global clientidx_map
        min_size_per_class = 0
        C = num_classes if self.cls_per == -1 else min(num_classes, max(self.cls_per, 1))
        min_require_size_per_class = max(C * num_clients // num_classes // 5, 1)
        while min_size_per_class < min_require_size_per_class:
            clientidx_map = { k: [] for k in range(num_classes) }
            for cid in range(num_clients):
                slected_classes = np.random.choice(range(num_classes), C, replace=False)
                for k in slected_classes:
                    clientidx_map[k].append(cid)
            min_size_per_class = min([len(clientidx_map[k]) for k in range(num_classes)])
        return clientidx_map

    @calculate_sample_stats
    def __call__(self, data, samples_per_client=None):
        global idx_per_client
        attrs = self.index_func(data)
        num_labels = len(attrs)
        num_attrs = len(set(attrs))
        alpha = self.alpha
        min_size = 0
        attrs = np.array(attrs)
        # 先决定每个客户的类别集
        clientidx_map = self.allocate_classes(self.num_clients, num_attrs)
        print("\n*****clientidx_map*****")
        print(clientidx_map)
        print("\n*****Number of clients per label*****")
        print([len(clientidx_map[cid]) for cid in range(num_attrs)])

        while min_size < self.minvol:
            idx_per_client = [[] for _ in range(self.num_clients)]
            for k in range(num_attrs):  # 分别划分每个类别
                idx_k = np.where(attrs == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, self.num_clients))
                # Case 1 (original case in Dir): Balance
                proportions = np.array(
                    [p * (len(idx_j) < num_labels / self.num_clients and j in clientidx_map[k]) for j, (p, idx_j) in
                     enumerate(zip(proportions, idx_per_client))])
                # Case 2: Don't balance
                # proportions = np.array([p * (j in label_netidx_map[k]) for j, (p, idx_j) in enumerate(zip(proportions, idx_per_client))])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

                '''Note: Process the remainder data samples (yipeng, 2023-11-14).
                There are some cases that the samples of class k are not allocated completely, i.e., proportions[-1] < len(idx_k)
                In these cases, the remainder data samples are assigned to the last client in `clientidx_map[k]`.
                '''
                if proportions[-1] != len(idx_k):
                    for w in range(clientidx_map[k][-1], self.num_clients - 1):
                        proportions[w] = len(idx_k)

                idx_per_client = [idx_j + idx.tolist() for idx_j, idx in
                                  zip(idx_per_client, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_per_client])

        local_datas = []
        for j in range(self.num_clients):
            np.random.shuffle(idx_per_client[j])
            local_datas.append(idx_per_client[j])
        self.local_datas = local_datas

        return local_datas

class PathologyPartitioner(BasicPartitioner):
    sign = 'Pathology'
    def __init__(self, cls_per=-1, imbalance=0.0, disturb=1.0, **kwargs):
        super().__init__(**kwargs)
        self.cls_per = cls_per
        self.imbalance = imbalance
        self.disturb = disturb

    @calculate_sample_stats
    def __call__(self, data, samples_per_client=None):
        labels = index_func(data)
        dpairs = [[did, lb] for did, lb in zip(range(len(data)), labels)]
        num_classes = len(set(labels))
        num = num_classes if self.cls_per == -1 else min(num_classes, max(self.cls_per, 1))
        K = num_classes

        if samples_per_client is None:
            samples_per_client = self.data_imbalance_generator(self.num_clients, len(data), self.imbalance)

        # 全局类别分布
        global_count = collections.Counter(labels)
        total_samples = len(labels)
        global_distribution = [global_count[c]/total_samples for c in range(num_classes)]

        # 基于disturbance_factor构造理想分布
        # 当disturbance_factor=0，期望分布为均匀分布(1/K)
        # 当disturbance_factor=1，期望分布为global_distribution
        # 否则在两者之间线性插值
        ideal_distribution = [
            self.disturb * global_distribution[c] + (1 - self.disturb)*(1.0/num_classes)
            for c in range(num_classes)
        ]

        # 按类别组织数据
        class_to_indices = {}
        for did, lb in dpairs:
            class_to_indices.setdefault(lb, []).append(did)

        # 初步多样性分配
        local_datas = [[] for _ in range(self.num_clients)]
        if num == K:
            contain = [list(range(K)) for _ in range(self.num_clients)]
            for k in tqdm(range(K), desc="Initial full diversity allocation"):
                idx_k = class_to_indices[k]
                np.random.shuffle(idx_k)
                splits = np.array_split(idx_k, self.num_clients)
                for cid in range(self.num_clients):
                    local_datas[cid].extend(splits[cid].tolist())
        else:
            times = [0] * K
            contain = []
            for i in range(self.num_clients):
                current = []
                while len(current) < num:
                    mintime = np.min(times)
                    inds = np.where(times == mintime)[0]
                    ind = np.random.choice(inds)
                    if ind not in current:
                        current.append(ind)
                        times[ind] += 1
                contain.append(current)

            for k in tqdm(range(K), desc="Initial partial diversity allocation"):
                idx_k = class_to_indices[k]
                np.random.shuffle(idx_k)
                splits = np.array_split(idx_k, times[k])
                idx_split = 0
                for cid in range(self.num_clients):
                    if k in contain[cid]:
                        local_datas[cid].extend(splits[idx_split].tolist())
                        idx_split += 1

        # 构建全局可用与已分配计数器
        global_assigned_count = {}
        global_available = {}
        for c in range(K):
            total_samples_this_cat = class_to_indices[c]
            assigned_in_cat = [x for cid in range(self.num_clients) for x in local_datas[cid] if labels[x] == c]
            global_assigned_count[c] = len(assigned_in_cat)
            assigned_set_cat = set(assigned_in_cat)
            global_available[c] = [x for x in total_samples_this_cat if x not in assigned_set_cat]

        # 裁剪阶段
        local_datas = self.trim_phase(local_datas, labels, contain, samples_per_client, global_available, global_assigned_count)

        # 计算离散程度函数
        def calc_dispersion(client_data, assigned_cats):
            if len(client_data) == 0:
                return 0
            cat_map = collections.Counter([labels[sid] for sid in client_data])
            total_local = len(client_data)
            dispersion = 0.0
            for c in assigned_cats:
                local_ratio_c = cat_map[c] / total_local if total_local>0 else 0
                target_ratio_c = ideal_distribution[c]
                dispersion += (local_ratio_c - target_ratio_c)**2
            return dispersion

        def update_client_priority(cid):
            needed = samples_per_client[cid] - len(local_datas[cid])
            if needed > 0:
                assigned_cats = contain[cid] if num != K else list(range(K))
                d = calc_dispersion(local_datas[cid], assigned_cats)
                return (-needed, -d, cid)
            else:
                return None

        pq = []
        for cid in range(self.num_clients):
            entry = update_client_priority(cid)
            if entry is not None:
                heapq.heappush(pq, entry)

        def supplement_one_sample(cid):
            assigned_cats = contain[cid] if num != K else list(range(K))
            cat_map = collections.Counter([labels[sid] for sid in local_datas[cid]])
            cat_counts = [(c, cat_map[c]) for c in assigned_cats]
            cat_counts.sort(key=lambda x: x[1])
            for c, _ in cat_counts:
                if len(global_available[c]) > 0:
                    sid = global_available[c].pop()
                    local_datas[cid].append(sid)
                    global_assigned_count[c] += 1
                    return True
            return False

        def get_top_two(pq):
            if len(pq) == 0:
                return None, None
            top1 = pq[0]
            top2 = pq[1] if len(pq) > 1 else None
            return top1, top2

        # 补充进度条
        pbar = tqdm(desc="Supplementing data", total=sum([max(0, spc - len(ld)) for spc, ld in zip(samples_per_client, local_datas)]))

        while pq:
            top1, top2 = get_top_two(pq)
            if top1 is None:
                break
            needed1, disp1, cid1 = top1
            needed1 = -needed1
            heapq.heappop(pq)

            needed2 = 0
            if top2 is not None:
                n2, d2, c2 = top2
                needed2 = -n2

            while True:
                current_needed = samples_per_client[cid1] - len(local_datas[cid1])
                if current_needed <= 0:
                    break
                if current_needed <= needed2:
                    entry = update_client_priority(cid1)
                    if entry is not None:
                        heapq.heappush(pq, entry)
                    break
                success = supplement_one_sample(cid1)
                if not success:
                    break
                else:
                    pbar.update(1)

            entry = update_client_priority(cid1)
            if entry is not None:
                heapq.heappush(pq, entry)

        pbar.close()
        return local_datas

    def trim_phase(self, local_datas, labels, contain, samples_per_client, global_available, global_assigned_count):
        for cid in tqdm(range(len(local_datas)), desc="Trimming data"):
            while len(local_datas[cid]) > samples_per_client[cid]:
                cat_map = {}
                for sid in local_datas[cid]:
                    c = labels[sid]
                    cat_map.setdefault(c, []).append(sid)
                cat_sizes = [(c, len(slist)) for c, slist in cat_map.items()]
                cat_sizes.sort(key=lambda x: x[1], reverse=True)
                max_cat, max_size = cat_sizes[0]

                if max_size <= 1:
                    break

                # 裁剪一个样本
                sid_to_remove = cat_map[max_cat].pop()
                local_datas[cid].remove(sid_to_remove)
                global_available[max_cat].append(sid_to_remove)
                global_assigned_count[max_cat] -= 1
        return local_datas

class CustomPartitioner(BasicPartitioner):
    sign = 'Custom'
    def __init__(self, num_map, cls_map, minvol=1, **kwargs):
        """
        Args:
            samples_per_client (list or tuple): 每个客户端需要的样本数列表，长度为num_clients
            labels_per_client (list or tuple): 每个客户端需要的标签描述列表
                - 若为整数n，则表示需要从全局标签集中选取n个标签分配给该客户端
                - 若为列表/元组，则表示该客户端所需的确切标签集合
            index_func (function): 从data中提取标签的函数
            minvol (int): 每个客户端的数据量下限
            alpha (float): Dirichlet分布参数，用来控制该客户端内部各标签样本量的分布不均匀性
        """
        super().__init__(**kwargs)
        assert not cls_map or not num_map
        assert len(cls_map) == len(num_map)
        self.cls_map = cls_map
        self.num_map = num_map
        self.minvol = minvol

    @calculate_sample_stats
    def __call__(self, data):
        global counts
        labels = index_func(data)
        unique_labels = list(set(labels))
        unique_labels.sort()
        num_classes = len(unique_labels)

        # 构建 label -> indices 的映射
        lb_dict = {lb: [] for lb in unique_labels}
        for i, lb in enumerate(labels):
            lb_dict[lb].append(i)
        # 打乱每个标签的样本顺序
        for lb in lb_dict:
            np.random.shuffle(lb_dict[lb])

        # 将labels_per_client分为两类：一类客户端有确定的标签集合，一类只有标签数量需求
        client_label_sets_fixed = {}
        client_label_nums = {}
        for cid, req in enumerate(self.cls_map):
            if isinstance(req, int):
                client_label_nums[cid] = req
            else:
                chosen_labels = list(req)
                for clb in chosen_labels:
                    if clb not in lb_dict:
                        raise ValueError(
                            "Client {} requires label {}, which is not present in the dataset".format(cid, clb))
                client_label_sets_fixed[cid] = chosen_labels

        # 利用已确定的标签信息初始化 times 和 contain
        times = [0 for _ in range(num_classes)]
        contain = [[] for _ in range(self.num_clients)]

        # 先处理固定标签集合的客户端
        for cid, fixed_labels in client_label_sets_fixed.items():
            for lb in fixed_labels:
                lb_idx = unique_labels.index(lb)
                times[lb_idx] += 1
                contain[cid].append(lb_idx)

        # 再处理需要分配标签数量的客户端
        for cid, num in client_label_nums.items():
            if num > num_classes:
                raise ValueError(
                    "Client {} requires {} labels, but only {} are available".format(cid, num, num_classes))
            j = 0
            while j < num:
                mintime = np.min(times)
                inds = np.where(np.array(times) == mintime)[0]
                ind = np.random.choice(inds)
                if ind not in contain[cid]:
                    j += 1
                    contain[cid].append(ind)
                    times[ind] += 1

        # 至此，每个客户端都有了最终的标签集合（包含从已确定或分配得来的标签类）
        local_datas = [[] for _ in range(self.num_clients)]

        for cid in range(self.num_clients):
            needed = self.num_map[cid]
            chosen_label_ids = contain[cid]
            chosen_labels = [unique_labels[i] for i in chosen_label_ids]
            num_c_labels = len(chosen_labels)
            if num_c_labels == 0:
                if needed > 0:
                    raise ValueError("Client {} has no labels assigned but needs {} samples.".format(cid, needed))
                else:
                    continue

            # 直接平均分配每个标签的样本
            if num_c_labels > 0:
                counts = [needed // num_c_labels] * num_c_labels
                # 处理余数
                remainder = needed % num_c_labels
                for i in range(remainder):
                    counts[i] += 1

            allocated = 0
            assignment = []
            # 根据counts为每个标签尝试分配样本
            for i, lb in enumerate(chosen_labels):
                want = counts[i]
                available = len(lb_dict[lb])
                if available < want:
                    # 如果该标签不够则分配全部available
                    want = available
                assigned_samples = lb_dict[lb][:want]
                lb_dict[lb] = lb_dict[lb][want:]
                assignment.extend(assigned_samples)
                allocated += len(assigned_samples)

            if allocated < needed:
                # 分配不足，尝试从这些标签中继续补
                short = needed - allocated
                for lb in chosen_labels:
                    if short <= 0:
                        break
                    if len(lb_dict[lb]) > 0:
                        can_take = min(short, len(lb_dict[lb]))
                        assigned_samples = lb_dict[lb][:can_take]
                        lb_dict[lb] = lb_dict[lb][can_take:]
                        assignment.extend(assigned_samples)
                        short -= can_take
                        allocated += can_take

                if allocated < needed:
                    raise ValueError(
                        "Not enough samples for client {}. Required {}, allocated {}.".format(cid, needed, allocated))

            local_datas[cid].extend(assignment)

        # 检查 minvol
        len_dist = [len(d) for d in local_datas]
        while min(len_dist) < self.minvol:
            min_did = np.argmin(len_dist)
            max_did = np.argmax(len_dist)
            if len(local_datas[max_did]) <= self.minvol:
                raise RuntimeError(
                    "The number of clients is too large or minvol too high, cannot ensure minvol={}".format(
                        self.minvol))
            sample_to_move = local_datas[max_did].pop()
            local_datas[min_did].append(sample_to_move)
            len_dist = [len(d) for d in local_datas]

        # 最终检查是否有重复样本
        all_samples = sum(local_datas, [])
        if len(set(all_samples)) != len(all_samples):
            raise RuntimeError(
                "Some samples are assigned to multiple clients, which violates the uniqueness constraint.")

        # 打乱数据
        for cid in range(self.num_clients):
            np.random.shuffle(local_datas[cid])

        return local_datas

class LongTailPartitioner:
    """Partition the indices of samples in the original dataset with optional ordering of classes.

    Args:
        partitioner_het (Partitioner): partitioner for heterogeneity
        type (str): imbalance type: 'exp', 'step' or others
        imb_factor (float): imbalance factor of attr distribution globally
        index_func (callable): function to extract label from data
        class_order_mode (str): the mode of class ordering, in {'normal', 'reverse', 'random', 'given'}
        given_class_order (list[int]): a user-specified class order if class_order_mode='given'
    """
    sign = 'LongTail'
    def __init__(self, partitioner, imb_type='exp', imb_factor=0.01, cls_order='random'):
        super().__init__()
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.partitioner = partitioner
        self.cls_order = cls_order

    def __str__(self):
        name = "imb_type{} imb_factor{:.1f}".format(self.imb_type, self.imb_factor)
        return name

    def _get_class_order(self, num_classes):
        # 根据模式来确定类的顺序
        if self.cls_order == 'positive':
            class_order = list(range(num_classes))
        elif self.cls_order == 'reverse':
            class_order = list(range(num_classes))[::-1]
        elif self.cls_order == 'random':
            class_order = list(range(num_classes))
            np.random.shuffle(class_order)
        elif self.cls_order is list and len(self.cls_order) == num_classes:
            class_order = self.cls_order
        else:
            raise ValueError("Unsupported class_order_mode: {}".format(self.cls_order))
        return class_order

    @calculate_sample_stats
    def __call__(self, data):
        labels = index_func(data)
        num_classes = len(set(labels))
        # 获取标签-样本字典
        list_label2indices = classify_label(data, num_classes)
        img_max = len(labels) / num_classes
        img_num_per_cls = []
        # 根据不平衡类型计算各类的样本数
        if self.imb_type == 'exp':  # 指数型长尾
            for _classes_idx in range(num_classes):
                num = img_max * (self.imb_factor ** (_classes_idx / (num_classes - 1.0)))
                img_num_per_cls.append(int(num))
        elif self.imb_type == 'step':  # 阶梯型不平衡，前半类数目较多，后半类数目较少
            half_num = num_classes // 2
            for cls_idx in range(half_num):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(num_classes - half_num):
                img_num_per_cls.append(int(img_max * self.imb_factor))
        else:
            # 若无特别定义，则所有类均保持相同样本数
            img_num_per_cls.extend([int(img_max)] * num_classes)
        # 获取类的顺序
        class_order = self._get_class_order(num_classes)
        # 按照类的顺序和对应的样本数截取数据
        new_dpairs = []
        for ordered_cls_idx, real_cls_idx in enumerate(class_order):
            cls_indices = list_label2indices[real_cls_idx]
            np.random.shuffle(cls_indices)
            selected_indices = cls_indices[:img_num_per_cls[ordered_cls_idx]]
            for idx in selected_indices:
                # dpair 仍然存原始的类标签 real_cls_idx，以保证label一致性
                new_dpairs.append([idx, real_cls_idx])

        # new_labels = self.index_func(new_dpairs)
        # for cls in range(num_classes):
        #     print(f"类别{cls}分配量为{len([None for l in new_labels if l == cls])}")
        # 打乱重新选择后的数据对
        np.random.shuffle(new_dpairs)
        # 利用 partitioner_het 对 new_dpairs 进行异构划分
        local_datas = self.partitioner(new_dpairs)

        return local_datas

# 边缘-本地两阶段划分
class HierarchPartitioner:
    def __init__(self, partitioner_cloud, partitioner_edge):
        self.p1 = partitioner_cloud
        self.p2 = partitioner_edge

    def get_info(self):
        return f"Hierarch-{self.p1.get_info()}-{self.p2.get_info()}"

    @calculate_sample_stats
    def __call__(self, data):
        edge_servers_data = self.p1(data)
        res = []
        for edge_data_idx in edge_servers_data:
            edge_data = [data[did] for did in edge_data_idx]
            edge_local_datas = self.p2(edge_data)
            for cid in range(len(edge_local_datas)):
                for k in range(len(edge_local_datas[cid])):
                    edge_local_datas[cid][k] = edge_data_idx[edge_local_datas[cid][k]]
            res.append(edge_local_datas)
        return res


# 样本量异构肯定是无法保证的，除非将其视作第三种异构属性
class LabelDomainPartitioner:
    def __init__(self, partitioner_label, partitioner_domain):
        self.p1 = partitioner_label
        self.p2 = partitioner_domain # 下层划分器残参数已指定

    def get_info(self):
        return f"LabelDomain-{self.p1.get_info()}-{self.p2.get_info()}"

    @calculate_sample_stats
    def __call__(self, data):
        local_data = {}

        # Step 1: 按类别划分
        print("Step 1: 按类别划分开始")
        data.set_mode('label')
        labels = index_func(data)
        num_class = len(set(labels))
        print(f"总类别数: {num_class}")

        ld_by_label = self.p1(data)
        local_num_by_cls = []  # 记录每个标签在客户上的样本量分布
        sample_per_client_domain = [[0 for _ in range(len(ld_by_label))] for _ in range(num_class)]
        for cid, ld in enumerate(ld_by_label):
            num_by_cls = [0 for _ in range(num_class)]
            for l in ld:
                num_by_cls[labels[l]] += 1
                sample_per_client_domain[labels[l]][cid] += 1
            local_num_by_cls.append(num_by_cls)
            print(f"客户端 {cid} 的类别分布: {num_by_cls} (总计: {sum(num_by_cls)})")

        for cls in range(num_class): # 确保类别分配只欠
            num_cls = 0
            for cid in range(len(local_num_by_cls)):
                num_cls += local_num_by_cls[cid][cls]
            print(f"类别 {cls} 已分配 {num_cls} 实际拥有 {sum([data.sample_count[d, cls] for d in range(data.num_domains)])}")

        # Step 2: 按领域异构划分
        print("Step 2: 按领域异构划分开始")
        for cls in range(num_class): # 确保每个类别至少有 domain个
            data.set_mode('domain')
            data.set_retain({'label': [cls]})
            domains = index_func(data)
            num_domain = len(set(domains))
            print(f"类别 {cls} 包含领域数: {num_domain} 总样本数 {len(domains)}")
            print(sample_per_client_domain[cls])
            ld_by_label_in_domain = self.p2(data, sample_per_client_domain[cls])  # 获得按领域划分的数据
            num_by_domain = [[] for _ in range(num_domain)]  # 统计每个领域的客户分配情况
            diff_by_domain_global = [0 for _ in range(num_domain)]  # 统计每个领域的客户分配情况
            diff_by_domain_local = [0 for _ in ld_by_label_in_domain]
            local_prob_by_domain = []
            local_num_by_domain = []
            # 由于领域中的每个类别是分配到每个客户，因此需要考虑领域-类别跨客户的完整性
            sum_by_domain = 0 # 数据平衡性是针对全局分布而言，因此需要取全局容量上的概率，而不是本地分布
            for cid, ld in enumerate(ld_by_label_in_domain):
                ld_num_by_domain = [0 for _ in range(num_domain)]
                # 统计该客户端每个领域的数量
                for l in ld: # 逐客户进行模型
                    ld_num_by_domain[domains[l]] += 1
                sum_by_domain += sum(ld_num_by_domain)
                ld_prob_by_domain = [ln / sum_by_domain for ln in ld_num_by_domain]
                local_prob_by_domain.append(ld_prob_by_domain)  # 存放该本地的领域分布
                local_num_by_domain.append(ld_num_by_domain)
                print(f"客户端 {cid} 在类别 {cls} 的领域概率分布: {ld_prob_by_domain}")
                print(f"客户端 {cid} 类别 {cls} 领域数量分布: {local_num_by_domain[cid]} 总计: {sum(local_num_by_domain[cid])}"
                      f" 原始: {local_num_by_cls[cid][cls]}")
                diff_by_domain_local[cid] += (sum(local_num_by_domain[cid]) - local_num_by_cls[cid][cls])

                # 逐领域统计
                for d in range(num_domain):
                    diff_by_domain_global[d] += local_num_by_domain[cid][d]
                    num_by_domain[d].append(local_num_by_domain[cid][d])

            for d in range(num_domain):
                diff_by_domain_global[d] -= data.sample_count[d, cls]

            # 当前类别下的领域分配(存在问题：为何领域不平衡，且普遍存在缺失？)
            print(f"类别 {cls} 领域总数: {[sum(nbd) for nbd in num_by_domain]}")
            print(f"类别 {cls} 实际领域总数: {[data.sample_count[d, cls] for d in range(num_domain)]}")

            # 确保解决客户样本 diff，解决领域样本 diff
            for d in range(num_domain):
                data.set_retain({'domain': [d], 'label': [cls]})
                h1 = sum(num_by_domain[d])
                idxes = list(range(h1))
                np.random.shuffle(idxes)
                pre_num = 0
                print(f'类别{cls} 领域{d} 容量{len(data)} 分配量{h1}')
                for cid, num in enumerate(num_by_domain[d]):
                    if cid not in local_data:
                        local_data[cid] = []
                    this_data = []
                    for local_idx in idxes[pre_num: pre_num + num]:
                        global_idx = data.get_global_idx(local_idx)
                        if global_idx is None:
                            print(local_idx)
                        this_data.append(global_idx)
                    local_data[cid].extend(this_data)
                    # local_data[cid].extend(
                    #     [data.get_global_idx(local_idx) for local_idx in idxes[pre_num: pre_num + num]])
                    pre_num += num

        data.set_retain()
        local_data = list(local_data.values())
        print("划分完成，返回客户端本地数据")

        # 检查类别异构性
        data.set_mode('label')
        for cid, ld in enumerate(local_data):
            labels = []
            for local_idx in ld:
                try:
                    labels.append(data[local_idx][-1])
                except:
                    print(local_idx)
            total_samples = len(labels)
            print(total_samples)
            label_counter = collections.Counter(labels)
            label_probabilities = {label: count / total_samples for label, count in label_counter.items()}
            print(f"客户端 {cid} 的类别概率分布: {label_probabilities}")

        # 检查领域异构性
        data.set_mode('domain')
        for cid, ld in enumerate(local_data):
            domains = [data[local_idx][-1] for local_idx in ld]
            total_samples = len(domains)
            print(total_samples)
            domain_counter = collections.Counter(domains)
            domain_probabilities = {domain: count / total_samples for domain, count in domain_counter.items()}
            print(f"客户端 {cid} 的领域概率分布: {domain_probabilities}")

        return local_data