from sae import Sae
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import torch
import pandas as pd
import pickle

# Originaly this script did all layers, but now it only does one layer at a time (because some of the seeds I only trained for 1 layer)
# If you have all layers, you can switch the [args.layer] to [0, 1, 2, ...]

parser = ArgumentParser()
parser.add_argument("--decoder", action="store_true")
parser.add_argument("--location", type=str, default="mlp")
parser.add_argument("--sae_dir", type=str, default="EleutherAI/sae-pythia-160m-32k")
parser.add_argument("--sae_2_seed_dir", type=str, default="/mnt/ssd-1/nora/sae/k32-sae-mlp-32k-seed2")
parser.add_argument("--name", type=str, default="")
parser.add_argument("--layer", type=int, default=6)
args = parser.parse_args()

def load_sae(dir, hookpoint, device):
    if "mnt" in dir:
        return Sae.load_from_disk(dir+"/"+hookpoint, device=device)
    else:
        return Sae.load_from_hub(dir, hookpoint=hookpoint, device=device)

sae_dir = args.sae_dir
sae_2_seed_dir = args.sae_2_seed_dir
scores = []
indices = []
for layer in tqdm([args.layer]):
    if args.location == "mlp":
        submodule = f"layers.{layer}.mlp"
    else:
        submodule = f"layers.{layer}"
    DEVICE = "cuda"
    sae = load_sae(sae_dir, submodule, DEVICE)
    sae_2 = load_sae(sae_2_seed_dir, submodule, DEVICE)
    if args.decoder:
        sae_weight = sae.W_dec.data / sae.W_dec.data.norm(dim=1, keepdim=True)
        sae_2_weight = sae_2.W_dec.data / sae_2.W_dec.data.norm(dim=1, keepdim=True)
    else:
        sae_weight = sae.encoder.weight.data / sae.encoder.weight.data.norm(dim=1, keepdim=True)
        sae_2_weight = sae_2.encoder.weight.data / sae_2.encoder.weight.data.norm(dim=1, keepdim=True)
    
    base_weight = sae_weight
    other_weights = sae_2_weight
    score = {}
    index = {}
    name = "SAE 2"
    # We have to do this because for the bigger models, the cost matrix is too big to fit in memory
    batch_size = 4096  # Adjust batch size based on available memory
    n_batches = (base_weight.shape[0] + batch_size - 1) // batch_size
    cost_matrix_skips = torch.zeros(base_weight.shape[0], other_weights.shape[0], device="cpu")
        
    for i in tqdm(range(n_batches)):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, base_weight.shape[0])
        value = base_weight[start_idx:end_idx] @ other_weights.T
        cost_matrix_skips[start_idx:end_idx] = value.cpu()
            
    cost_matrix_skips = torch.nan_to_num(cost_matrix_skips, nan=0)
    row_ind_skips, col_ind_skips = linear_sum_assignment(cost_matrix_skips.numpy(), maximize=True)
    score[name] = cost_matrix_skips[row_ind_skips, col_ind_skips].mean().item()
    index[name] = (row_ind_skips, col_ind_skips)
    print(f"{name}: {score[name]}")
    scores.append(score)
    indices.append(index)

df = pd.DataFrame(scores)
if args.decoder:
    file_name_indices = f"indices_decoder_{args.name}.pkl"
    file_name_scores = f"scores_decoder_{args.name}.csv"
else:
    file_name_indices = f"indices_encoder_{args.name}.pkl"
    file_name_scores = f"scores_encoder_{args.name}.csv"
with open(file_name_indices, "wb") as f:
    pickle.dump(indices, f)

df.to_csv(file_name_scores)




