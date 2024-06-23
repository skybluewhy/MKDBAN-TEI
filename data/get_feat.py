import torch
import esm
import numpy as np


d = dict()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model = model.eval().cuda()  # disables dropout for deterministic results

f = open("./all_seqs.txt", 'r')
all_l = []
cnt = 0
for line in f:
    line = line.replace("\n", "")
    all_l.append(line)
index = 0
for p in all_l:
    data = [("protein", p)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    d[p] = results["representations"][33].cpu().numpy().tolist()
    index += 1
    print(index)
np.save("./esm_features.npy", d)
