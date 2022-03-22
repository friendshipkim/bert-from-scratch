from torch import nn
from torch import Tensor


class Pooler(nn.Module):
    def __init__(self, d_model: int):
        super(Pooler, self).__init__()

        self.dense = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        :param hidden_states: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        :return: torch.Tensor, shape: (batch_size, max_seq_len, d_model)
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        x = self.dense(first_token_tensor)
        x = self.tanh(x)

        return x
