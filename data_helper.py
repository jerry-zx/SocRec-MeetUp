import os
import re
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
    idx2mem = dict()
    m_idx = 0
    data_dict = {}
    # LOAD_DICT = False

    if os.path.exists('./temp_data/event2idx.json') and \
            os.path.exists('./temp_data/mem2idx.json') and \
            os.path.exists('./temp_data/data.json') and \
            os.path.exists('./temp_data/group2topic.json') and \
            os.path.exists('./temp_data/mem2topic.json') and \
            os.path.exists('./temp_data/topic2idx.json') and \
            os.path.exists('./temp_data/idx2mem.json') and\
            os.path.exists('./temp_data/idx2topic.json'):
        with open("./temp_data/data.json", 'r', encoding='utf-8') as f_in:
            data_dict = json.loads(f_in.readline())
        with open('./temp_data/group2topic.json', 'r', encoding='utf-8') as f_in :
            group2topic = json.loads(f_in.readline())
        with open("./temp_data/mem2topic.json", 'r', encoding='utf-8') as f_in :
            mem2topic = json.loads(f_in.readline())

        return data_dict, [group2topic, mem2topic]

    event_data_list = get_file_list('./data/GroupEvent')
    for event_file_id, event_file in enumerate(event_data_list):
        with open(event_file, 'r', encoding='utf-8') as in_f:
            group_id = re.findall(r'\d+', event_file)[0]
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
                    data_dict[event2idx[event]]['group_id'] = int(group_id)
                elif idx % 5 == 1:  # for organizer
                    org = line.strip()
                    if org not in mem2idx.keys():
                        mem2idx[org] = m_idx
                        idx2mem[m_idx] = org
                        m_idx += 1
                    data_dict[event2idx[event]]['org'] = mem2idx[org]
                else:
                    mem_list = line.split()
                    _ = list()
                    for mem in mem_list:
                        if mem not in mem2idx.keys():
                            mem2idx[mem] = m_idx
                            idx2mem[m_idx] = mem
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

    #
    topic2idx = defaultdict(int)
    idx2topic = dict()
    group2topic = defaultdict(list)
    # leaving t_idx = 0 as dummy id, because some users' interesting topic is unknown
    t_idx = 1
    with open('./data/GroupTopic.txt', 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            if idx % 2 == 0:
                group_id, founder = line.split()
                group_id = int(re.findall(r'\d+', group_id)[0])
            else:
                line = line.split()
                topic_list = list()
                for topic in line:
                    if topic not in topic2idx.keys():
                        topic2idx[topic] = t_idx
                        idx2topic[t_idx] = topic
                        t_idx += 1
                    topic_list.append(topic2idx[topic])
                group2topic[group_id] = topic_list

    mem2topic = defaultdict(list)
    with open('./data/MemberTopic.txt', 'r', encoding='utf-8') as f_in:
        for idx, line in enumerate(f_in):
            if idx % 2 == 0:
                mem = line.strip()
            elif idx % 2 == 1:
                mem_topic_list = line.strip().split()
                topic_id_list = []
                for topic in mem_topic_list:
                    if topic in topic2idx.keys():
                        topic_id_list.append(topic2idx[topic])
                    else:
                        topic2idx[topic] = t_idx
                        t_idx += 1
                        topic_id_list.append(topic2idx[topic])
                if len(topic_id_list) == 0:
                    topic_id_list.append(0)
                if mem not in mem2idx.keys():
                    continue
                else:
                    mem2topic[mem2idx[mem]] = topic_id_list
        for mem_idx in list(mem2idx.values()):
            if mem_idx not in mem2topic.keys():
                mem2topic[mem_idx] = [0]

    with open('./temp_data/topic2idx.json', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(topic2idx, ensure_ascii=False))
    with open('./temp_data/group2topic.json', 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(group2topic, ensure_ascii=False))
    with open("./temp_data/mem2topic.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(mem2topic, ensure_ascii=False))
    with open("./temp_data/idx2mem.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(idx2mem, ensure_ascii=False))
    with open("./temp_data/idx2topic.json", 'w', encoding='utf-8') as f_out:
        f_out.write(json.dumps(idx2topic, ensure_ascii=False))





    return data_dict, [group2topic, mem2topic]


def data_split(data: dict, other_dict: dict, dev_ratio: float = 0.1, test_ratio: float = 0.2, topic_padding: int = 5):
    def _make_a_data(data_key):
        d = data[data_key]
        yes_m_list = d['yes']
        no_m_list = d['no']
        maybe_m_list = d['maybe']
        return yes_m_list, no_m_list, maybe_m_list

    def _topic_padding(topic_list):
        if len(topic_list) >= topic_padding:
            topic_list = topic_list[:topic_padding]
        else:
            topic_list.extend([0] * (topic_padding - len(topic_list)))
        return topic_list

    import numpy as np

    group2topic, mem2topic = other_dict

    np.random.seed(0)
    total = len(data)
    amount_dev = int(total * dev_ratio)
    amount_test = int(total * test_ratio)
    data_key = list(data.keys())
    np.random.shuffle(data_key)

    dev_keys = data_key[:amount_dev]
    test_keys = data_key[amount_dev: amount_dev + amount_test]
    train_keys = data_key[amount_dev + amount_test:]

    train_data, dev_data, test_data = [], [], []
    train_label, dev_label, test_label = [], [], []

    # data:(org,group,member)
    for k in train_keys:
        yes, no, maybe = _make_a_data(k)
        org, group = data[k]['org'], data[k]['group_id']
        train_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in yes])
        train_label.extend([0 for _ in yes])
        train_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in no])
        train_label.extend([1 for _ in no])
        train_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in maybe])
        train_label.extend([1 for _ in maybe])

    for k in dev_keys:
        yes, no, maybe = _make_a_data(k)
        org, group = data[k]['org'], data[k]['group_id']
        dev_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in yes])
        dev_label.extend([0 for _ in yes])
        dev_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in no])
        dev_label.extend([1 for _ in no])
        dev_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in maybe])
        dev_label.extend([1 for _ in maybe])

    for k in test_keys:
        yes, no, maybe = _make_a_data(k)
        org, group = data[k]['org'], data[k]['group_id']
        test_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in yes])
        test_label.extend([0 for _ in yes])
        test_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in no])
        test_label.extend([1 for _ in no])
        test_data.extend([(org, group, _, _topic_padding(mem2topic[str(_)]), _topic_padding(group2topic[str(group)])) for _ in maybe])
        test_label.extend([1 for _ in maybe])

    return train_data, train_label, dev_data, dev_label, test_data, test_label


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        super(Dataset, self).__init__()
        self.data = data
        self.label = label

    def __getitem__(self, idx):
        batch = dict()
        batch['input'] = torch.tensor(self.data[idx][:3], dtype=torch.int64)
        batch['mem_topic'] = torch.tensor(self.data[idx][3], dtype=torch.int64)
        batch['group_topic'] = torch.tensor(self.data[idx][4], dtype=torch.int64)
        batch['label'] = torch.tensor(self.label[idx], dtype=torch.int64)
        return batch

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # print(get_file_list("./data/GroupEvent"))
    load_data()
    pass
