import torch
import torch.nn as nn
import torch.nn.functional as F

class DNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.m_embedding = nn.Embedding(args.n_mem, args.embed_dim)
        self.e_embedding = nn.Embedding(args.n_ene, args.embed_dim)

        self.fc1 = nn.Linear(args.embed_dim * 2, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.n_output)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, batch, label=None):
        e_embedding = self.e_embedding(batch[:, 0])
        m_embedding = self.m_embedding(batch[:, 1])

        concat_embed = torch.cat((e_embedding, m_embedding), dim=-1)
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