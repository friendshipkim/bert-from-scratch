import torch

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

# data hyperparameters - to be removed
vocab_size = 30522
pad_idx = 1
num_labels = 3
type_vocab_size = 2

# model architecture hyperparameters
n_layers = 12
d_model = emb_size = 768
ffn_hidden = d_model * 4
h = 12
p_drop = 0.1
eps = 1e-12

# training hyperparameters
batch_size = 32
eval_batch_size = 16  # TODO: check if it is necessary
