from torch import nn, Tensor

from bert.sublayer.token_embedding import TokenEmbedding
from bert.sublayer.positional_encoding import PositionalEncoding
from bert.sublayer.token_type_embedding import TokenTypeEmbedding


class Embedding(nn.Module):
    def __init__(self, emb_size: int, vocab_size: int, type_vocab_size: int, pad_idx: int,
                 p_drop: float, eps: float, max_pos_emb: int = 512):
        super(Embedding, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.positional_encodings = nn.Embedding(max_pos_emb, emb_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, emb_size)  # 0 - 1st sentence, 1 - 2nd sentence
        self.layer_norm = nn.LayerNorm(emb_size, eps=eps)
        self.dropout = nn.Dropout(p=p_drop)

    def forward(self, x: Tensor, token_type_ids: Tensor) -> Tensor:
        """
        :param x: torch.Tensor, shape: (batch_size, seq_len)
        :return: torch.Tensor, shape: (batch_size, seq_len, d_model)
        """
        input_emb = self.token_embeddings(x)
        token_type_emb = self.token_type_embeddings(token_type_ids)
        emb = input_emb + token_type_emb
        x = self.positional_encoding(x)

        x = self.layer_norm(x)
        x = self.dropout(x)
        return x



