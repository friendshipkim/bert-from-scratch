import typing

from torch import nn, Tensor
from bert.bert_model import BertModel


class BertForSequenceClassification(nn.Module):
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
            num_labels: int,
            eps: float,
            device: str
    ):
        super(BertForSequenceClassification, self).__init__()

        # bertmodel
        self.bert = BertModel(n_layers=n_layers,
                              d_model=d_model,
                              h=h,
                              ffn_hidden=ffn_hidden,
                              p_drop=p_drop,
                              vocab_size=vocab_size,
                              pad_idx=pad_idx,
                              type_vocab_size=type_vocab_size,
                              eps=eps,
                              device=device)

        # dropout
        self.dropout = nn.Dropout(p=p_drop)

        # final classifier
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, src: Tensor) -> typing.Tuple:
        """
        :param src: torch.Tensor, shape: (batch_size, src_seq_len)
        :param tgt: torch.Tensor, shape: (batch_size, tgt_seq_len)

        :return: torch.Tensor, shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        # bertmodel
        src_emb, enc_output, pool_out = self.bert(src)

        # dropout, classifier
        model_out = self.dropout(pool_out)
        model_out = self.classifier(model_out)
        return src_emb, enc_output, pool_out, model_out

