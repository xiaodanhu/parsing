from numpy import gradient
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, embedding_size, n=10000):
        super().__init__()

        i = torch.arange(embedding_size // 2)
        k = torch.arange(max_seq_len).unsqueeze(dim=1)

        self.pos_embeddings = torch.zeros(
            max_seq_len, embedding_size, requires_grad=False
        ).cuda()
        self.pos_embeddings[:, 0::2] = torch.sin(k / (n ** (2 * i / embedding_size)))
        self.pos_embeddings[:, 1::2] = torch.cos(k / (n ** (2 * i / embedding_size)))

    def forward(self, x):
        return self.pos_embeddings[: x.shape[1], :]
