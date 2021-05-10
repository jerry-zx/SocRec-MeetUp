import os
import math
import utils
import torch
import argparse
import prettytable
import collections
import numpy as np
import data_helper as dh
import torch.nn.functional as F

from model import DNN
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    data_dict, topic_dict, neighbor_dict = dh.load_data()  # data_dict, [group2topic, mem2topic]
    print("Data Preprocess Completed")
    train_data, train_label, dev_data, dev_label, test_data, test_label = dh.data_split(data_dict, topic_dict,
                                                                                        neighbor_dict)
    train_dataset = dh.Dataset(train_data, train_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataset = dh.Dataset(dev_data, dev_label)
    dev_loader = DataLoader(dev_dataset, batch_size=128, shuffle=True)
    print("Dataloader Completed")

    lambda1 = lambda epoch: (epoch / args.warm_up_step) if epoch < args.warm_up_step else 0.5 * (
            math.cos(
                (epoch - args.warm_up_step) / (args.n_epoch * len(train_dataset) - args.warm_up_step) * math.pi) + 1)

    model = DNN(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader) * args.n_epoch)

    global_step = 0
    best_f1 = 0.
    loss_deq = collections.deque([], args.report_step)
    for epoch in range(args.n_epoch):
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = batch['input'].to(device)
            mem_neighbor = batch['mem_neighbor'].to(device)
            group_topic = batch['group_topic'].to(device)
            mem_topic = batch['mem_topic'].to(device)
            labels = batch['label'].to(device)
            output = model(inputs, mem_neighbor, mem_topic, group_topic, label=labels)
            loss = output[0]
            loss.backward()
            loss_deq.append(loss.item())
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % args.report_step == 0:
                logger.info('loss: {}, lr: {}, epoch: {}'.format(np.average(loss_deq).item(),
                                                                 optimizer.param_groups[0]['lr'],
                                                                 global_step / len(train_dataset)))
            if global_step % args.eval_step == 0:
                model.eval()
                eval_result = evaluation(model, data_loader=dev_loader, device=device)
                logger.info(eval_result)
                if eval_result['f1'] > best_f1:
                    torch.save(model, './model/{}/torch.pt'.format(args.task_name))
                    best_f1 = eval_result['f1']
                model.train()


def evaluation(model, data_loader, device):
    def compute_metrics(preds, labels, loss):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, np.round(preds).tolist(), average='binary')
        acc = accuracy_score(labels, np.round(preds).tolist())
        # auc = roc_auc_score(labels, preds, multi_class='ovo')
        return {
            'loss': np.average(loss),
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            # 'auc':  auc,
        }

    accu_loss = []
    accu_pred = []
    accu_label = []
    for batch in tqdm(data_loader):
        inputs = batch['input'].to(device)
        mem_neighbor = batch['mem_neighbor'].to(device)
        group_topic = batch['group_topic'].to(device)
        mem_topic = batch['mem_topic'].to(device)
        labels = batch['label'].to(device)
        output = model(inputs, mem_neighbor, mem_topic, group_topic, label=labels)
        loss, logits = output
        pred = F.softmax(logits, dim=-1)
        _, pred = torch.max(pred, dim=-1)
        accu_loss.append(loss.item())
        accu_pred.extend(pred.tolist())
        accu_label.extend(labels.tolist())

    return compute_metrics(accu_pred, accu_label, accu_loss)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task_name', type=str, default='test05101126')
    argparser.add_argument('--warm_up_step', type=int, default=100)
    argparser.add_argument('--report_step', type=int, default=3000)
    argparser.add_argument('--eval_step', type=int, default=3000)
    argparser.add_argument('--n_epoch', type=int, default=20)
    argparser.add_argument('--init_lr', type=float, default=1e-3)
    argparser.add_argument('--batch_size', type=int, default=128)
    argparser.add_argument('--embed_dim', type=int, default=32)
    argparser.add_argument('--hidden_size', type=int, default=32)
    argparser.add_argument('--n_mem', type=int, default=75136)
    argparser.add_argument('--n_ene', type=int, default=55396)
    argparser.add_argument('--n_group', type=int, default=482)
    argparser.add_argument('--n_topic', type=int, default=17129)
    argparser.add_argument('--n_output', type=int, default=2)
    args = argparser.parse_args()
    dir_check_list = [
        './log',
        './model',
        './model/{}'.format(args.task_name),
    ]
    for dir in dir_check_list:
        if not os.path.exists(dir):
            os.mkdir(dir)
    logger = utils.init_logger('./log/{}.log'.format(args.task_name))

    pt = prettytable.PrettyTable()
    pt.field_names = ['arg', 'val']
    for k, v in vars(args).items():
        pt.add_row([k, v])
    logger.info("\n" + str(pt))

    train()
