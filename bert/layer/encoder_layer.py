from torch import nn
from torch import Tensor


from bert.sublayer.multihead_attention import MultiHeadAttention
from bert.sublayer.pointwise_feedforward import PointwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, h: int, ffn_hidden: int, p_drop: float, eps: float):
        super(EncoderLayer, self).__init__()
        self.h = h

        # BertAttention + BertSelfOutput
        self.self_attn = MultiHeadAttention(d_model, h, p_drop)
        self.norm1 = nn.LayerNorm(d_model, eps)
        self.dropout1 = nn.Dropout(p_drop)

        # BertIntermediate + BertOutput
        self.ffn = PointwiseFeedForward(d_model, ffn_hidden, p_drop)
        self.norm2 = nn.LayerNorm(d_model, eps)
        self.dropout2 = nn.Dropout(p_drop)

    def forward(self, x: Tensor, src_pad_mask: Tensor = None) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, src_seq_len, d_model)
        :param src_pad_mask: torch.Tensor, shape (batch_size, src_seq_len)
        :return: torch.Tensor, shape (batch_size, src_seq_len, d_model)
        """

        # 1. BertAttention + BertSelfOutput
        attn_out, attn_score = self.self_attn(q=x, k=x, v=x,
                                              mask=None if src_pad_mask == None
                                                        else self.create_enc_mask(pad_mask=src_pad_mask))
        out = self.norm1(x + attn_out)
        out = self.dropout1(out)

        # tmp: saving attention outputs
        self.attn_score = attn_score
        self.attn_out = attn_out

        # 2. BertIntermediate + BertOutput
        cp_out = out
        out = self.ffn(out)
        out = self.norm2(out + cp_out)  # TODO: check if norm is first
        out = self.dropout2(out)

        return out

    def create_enc_mask(self, pad_mask: Tensor):
        # TODO maybe put this into multihead
        """
        transform source padding mask into proper shape

        :param pad_mask: torch.Tensor, shape: (batch_size, src_seq_len)
        :return: enc_mask: torch.Tensor, shape: (batch_size * h, src_seq_len, src_seq_len)
        """
        batch_size, src_seq_len = pad_mask.size()
        enc_mask = pad_mask.view(batch_size, 1, 1, src_seq_len). \
            expand(-1, self.h, src_seq_len, -1).reshape(batch_size * self.h, src_seq_len, src_seq_len)
        return enc_mask
