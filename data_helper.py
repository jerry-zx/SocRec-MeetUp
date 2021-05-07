import os
import json
import torch
from collections import defaultdict
from sklearn.model_selection import train_test_split

def get_file_list(dir_path):
    file_list = []
    for root, path, filename_list in os.walk(dir_path):
        for file_name in filename_list:
            file_list.append(os.path.join(root, file_name))
    return file_list

def load_data():
    if not os.path.exists('./temp_data'):
        os.mkdir('./temp_data')
    # if os.path.exists('./temp_data/event2idx.json') and os.path.exists('./temp_data/mem2idx.json'):
    #     with open('./temp_data/event2idx.json', 'r', encoding='utf-8') as in_f:
    #         event2idx = json.loads(in_f.readline())
    #     with open('./temp_data/mem2idx.json', 'r', encoding='utf-8') as in_f:
    #         mem2idx = json.loads(in_f.readline())
    #     LOAD_DICT = True
    event2idx = defaultdict(int)
    e_idx = 0
    mem2idx = defaultdict(int)
    m_idx = 0
    data_dict = {}
    # LOAD_DICT = False

    if os.path.exists('./temp_data/event2idx.json') and \
        os.path.exists('./temp_data/mem2idx.json') and \
        os.path.exists('./temp_data/data.json'):
        with open("./temp_data/data.json", 'r', encoding='utf-8') as in_f:
            data_dict = json.loads(in_f.readline())
        return data_dict

    event_data_list = get_file_list('./data/GroupEvent')
    for event_file in event_data_list:
        with open(event_file, 'r', encoding='utf-8') as in_f:
            for idx, line in enumerate(in_f):
                # print(line)
                if idx % 5 == 0:
                    event, n_mem, time = line.split()
                    if event not in event2idx.keys():
                        event2idx[event] = e_idx
                        e_idx += 1
                    data_dict[event2idx[event]] = {}
                    data_dict[event2idx[event]]['n_mem'] = int(n_mem)
                    data_dict[event2idx[event]]['time'] = time
                elif idx % 5 == 1:  # for organizer
                    org = line.strip()
                    if org not in mem2idx.keys():
                        mem2idx[org] = m_idx
                        m_idx += 1
                    data_dict[event2idx[event]]['org'] = mem2idx[org]
                else:
                    mem_list = line.split()
                    _ = list()
                    for mem in mem_list:
                        if mem not in mem2idx.keys():
                            mem2idx[mem] = m_idx
                            m_idx += 1
                        _.append(mem2idx[mem])
                    if idx % 5 == 2:
                        data_dict[event2idx[event]]['yes'] = _
                    elif idx % 5 == 3:
                        data_dict[event2idx[event]]['no'] = _
                    elif idx % 5 == 4:
                        data_dict[event2idx[event]]['maybe'] = _
    with open("./temp_data/event2idx.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(event2idx, ensure_ascii=False))
    with open("./temp_data/mem2idx.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(mem2idx, ensure_ascii=False))
    with open('./temp_data/data.json', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(data_dict, ensure_ascii=False))

    return data_dict


def data_split(data: dict, dev_ratio: float=0.1, test_ratio: float=0.2):

    def _make_a_data(data_key):
        d = data[data_key]
        yes_m_list = d['yes']
        no_m_list = d['no']
        maybe_m_list = d['maybe']
        return yes_m_list, no_m_list, maybe_m_list


    import numpy as np

    np.random.seed(0)
    total = len(data)
    print("total",total)
    amount_dev = int(total * dev_ratio)
    amount_test = int(total * test_ratio)
    data_key = list(data.keys())
    np.random.shuffle(data_key)

    dev_keys = data_key[:amount_dev]
    test_keys = data_key[amount_dev: amount_dev+amount_test]
    train_keys = data_key[amount_dev+amount_test: ]

    train_data, dev_data, test_data = [], [], []
    train_label, dev_label, test_label = [], [], []
    for k in train_keys:
        yes, no, maybe = _make_a_data(k)
        k = int(k)
        train_data.extend([(k, _) for _ in yes])
        train_label.extend([0 for _ in yes])
        train_data.extend([(k, _) for _ in no])
        train_label.extend([1 for _ in no])
        train_data.extend([(k, _) for _ in maybe])
        train_label.extend([2 for _ in maybe])

    for k in dev_keys:
        yes, no, maybe = _make_a_data(k)
        k = int(k)
        dev_data.extend([(k, _) for _ in yes])
        dev_label.extend([0 for _ in yes])
        dev_data.extend([(k, _) for _ in no])
        dev_label.extend([1 for _ in no])
        dev_data.extend([(k, _) for _ in maybe])
        dev_label.extend([2 for _ in maybe])

    for k in test_keys:
        yes, no, maybe = _make_a_data(k)
        k = int(k)
        test_data.extend([(k, _) for _ in yes])
        test_label.extend([0 for _ in yes])
        test_data.extend([(k, _) for _ in no])
        test_label.extend([1 for _ in no])
        test_data.extend([(k, _) for _ in maybe])
        test_label.extend([2 for _ in maybe])

    return train_data, train_label, dev_data, dev_label, test_data, test_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label) :
        super(Dataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, idx) :
        batch = dict()
        batch['input'] = torch.tensor(self.data[idx], dtype=torch.int64)
        batch['label'] = torch.tensor(self.label[idx], dtype=torch.int64)
        return batch

    def __len__(self) :
        return len(self.data)

if __name__ == '__main__':
    print(get_file_list("./data/GroupEvent"))
    load_data()
    pass