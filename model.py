import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.m_embedding = nn.Embedding(args.n_mem, args.embed_dim)
        self.g_embedding = nn.Embedding(args.n_group, args.embed_dim)
        self.t_embedding = nn.Embedding(args.n_topic, args.embed_dim)

        self.fc1 = nn.Linear(args.embed_dim * 5, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.n_output)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch, m_topic, g_topic, label=None):
        org_embedding = self.m_embedding(batch[:, 0])
        g_embedding = self.g_embedding(batch[:, 1])
        m_embedding = self.m_embedding(batch[:, 2])

        m_topic_embedding = self.t_embedding(m_topic) #[batch, pad_length, embed]
        g_topic_embedding = self.t_embedding(g_topic) #[batch, pad_length, embed]
        m_topic_embedding = torch.mean(m_topic_embedding, dim=1)
        g_topic_embedding = torch.mean(g_topic_embedding, dim=1)


        concat_embed = torch.cat((org_embedding, g_embedding, m_embedding, m_topic_embedding, g_topic_embedding), dim=-1)
        hidden = self.fc1(concat_embed)
        hidden = F.relu(hidden)
        hidden = self.fc2(hidden)
        hidden = F.relu(hidden)
        logits = self.fc3(hidden)

        if label is not None:
            loss = self.loss_fct(logits, label)
            output = (loss, logits)
        else:
            output = (logits,)

        return output
