import typing

import torch
from torch import nn, Tensor

from bert.layer.embedding import Embedding
from bert.encoder import Encoder
from bert.pooler import Pooler


class BertModel(nn.Module):
    def __init__(
            self,
            n_layers: int,
            d_model: int,
            h: int,
            ffn_hidden: int,
            p_drop: float,
            vocab_size: int,
            pad_idx: int,
            type_vocab_size: int,
            eps: float,
            device: str
    ):
        super(BertModel, self).__init__()

        self.pad_idx = pad_idx
        self.device = device

        # embedding
        self.embedding = Embedding(emb_size=d_model,
                                   vocab_size=vocab_size,
                                   type_vocab_size=type_vocab_size,
                                   pad_idx=pad_idx,
                                   p_drop=p_drop,
                                   eps=eps)

        # encoder
        self.encoder = Encoder(n_layers=n_layers,
                               d_model=d_model,
                               h=h,
                               ffn_hidden=ffn_hidden,
                               p_drop=p_drop,
                               eps=eps
                               )

        # pooler
        self.pooler = Pooler(d_model=d_model)

    def forward(self, src: Tensor) -> typing.Tuple:
        """
        :param src: torch.Tensor, shape: (batch_size, src_seq_len)
        :param tgt: torch.Tensor, shape: (batch_size, tgt_seq_len)

        :return: torch.Tensor, shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # mask
        src_pad_mask = self.create_pad_mask(src)

        # embedding
        src_emb = self.embedding(src)

        # encoder
        enc_output = self.encoder(x=src_emb,
                                  src_pad_mask=src_pad_mask)

        # pooler
        pool_output = self.pooler(enc_output)

        return src_emb, enc_output, pool_output  # F.log_softmax(model_out, dim=-1) # TODO

    def create_pad_mask(self, x: Tensor) -> Tensor:
        """
        Create a mask to hide padding

        :param x: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len)
        """
        return x != self.pad_idx

    def create_autoregressive_mask(self, x: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len, seq_len)
        """
        seq_len = x.size(1)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=self.device).tril(diagonal=0)
        return mask