import numpy as np
import copy

def get_long_tail(dataset_image, dataset_label, num_classes, imb_factor, imb_type):
    list_label2indices = classify_label(dataset_label, num_classes)
    total_class_num, list_label2indices_train_new = train_long_tail(list_label2indices, num_classes, imb_factor, imb_type)
    new_indices = np.concatenate(list_label2indices_train_new)
    return dataset_image[new_indices], dataset_label[new_indices], len(total_class_num)

def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

# 0.01:[5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]
# 0.02:[5000, 3237, 2096, 1357, 878, 568, 368, 238, 154, 100]
# 0.05:[5000, 3584, 2569, 1842, 1320, 946, 678, 486, 348, 250]
def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes#5000
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))
    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type = 'exp'):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('Original number of samples of each label:')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes)) #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))
    return img_num_list, list_clients_indices


def classify_label(labels, num_classes: int):
    list1 = [[] for _ in range(num_classes)]
    for idx, label in enumerate(labels):
        list1[label].append(idx)
    return list1

def show_clients_data_distribution(dataset_label, clients_indices: list, num_classes):
    dict_per_client = []

    for client, indices in enumerate(clients_indices):
        nums_data = [0 for _ in range(num_classes)]
        for idx in indices:
            label = dataset_label[idx]
            nums_data[label] += 1
        new_num_data = []
        total = sum(nums_data)
        for i in range(num_classes):
            new_num_data.append((i,nums_data[i]))
        dict_per_client.append((total,new_num_data))
        print(f'client:{client}:  {total} {new_num_data}')
    return dict_per_client


