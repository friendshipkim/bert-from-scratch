# TODO

import math
from torch import nn, Tensor


class TokenTypeEmbedding(nn.Module):
    def __init__(self, emb_size: int, type_vocab_size: int):
        super(TokenTypeEmbedding, self).__init__()
        self.embedding = nn.Embedding(type_vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor) -> Tensor:
        """
        :param tokens: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len, emb_size)
        """
        # TODO - how does it work?
        out = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        # multiply embedding weights by sqrt(d_model)
        return out
