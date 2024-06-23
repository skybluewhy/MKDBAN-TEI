import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np


blosum62 = {
    "A": np.array(
        (4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0)
    ),
    "R": np.array(
        (-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3)
    ),
    "N": np.array(
        (-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3)
    ),
    "D": np.array(
        (-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3)
    ),
    "C": np.array(
        ( 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1)
    ),
    "Q": np.array(
        (-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2)
    ),
    "E": np.array(
        (-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2)
    ),
    "G": np.array(
        ( 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3)
    ),
    "H": np.array(
        (-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3)
    ),
    "I": np.array(
        (-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3, 1, 0, -3, -2, -1, -3, -1,  3)
    ),
    "L": np.array(
        (-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1,  1)
    ),
    "K": np.array(
        (-1,  2, 0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1, 0, -1, -3, -2, -2)
    ),
    "M": np.array(
        (-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1)
    ),
    "F": np.array(
        (-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1)
    ),
    "P": np.array(
        (-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2)
    ),
    "S": np.array(
        (1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2)
    ),
    "T": np.array(
        (0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0)
    ),
    "W": np.array(
        (-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11,  2, -3)
    ),
    "Y": np.array(
        (-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1)
    ),
    "V": np.array(
        (0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4)
    ),
}

atchley = {
    "A": np.array(
        (-0.591, -1.302, -0.733, 1.57, -0.146)
    ),
    "R": np.array(
        (1.538, -0.055, 1.502, 0.44, 2.897)
    ),
    "N": np.array(
        (0.945, 0.828, 1.299, -0.169, 0.933)
    ),
    "D": np.array(
        (1.05, 0.302, -3.656, -0.259, -3.242)
    ),
    "C": np.array(
        (-1.343, 0.465, -0.862, -1.02, -0.255)
    ),
    "Q": np.array(
        (0.931, -0.179, -3.005, -0.503, -1.853)
    ),
    "E": np.array(
        (1.357, -1.453, 1.477, 0.113, -0.837)
    ),
    "G": np.array(
        (-0.384, 1.652, 1.33, 1.045, 2.064)
    ),
    "H": np.array(
        (0.336, -0.417, -1.673, -1.474, -0.078)
    ),
    "I": np.array(
        (-1.239, -0.547, 2.131, 0.393, 0.816)
    ),
    "L": np.array(
        (-1.019, -0.987, -1.505, 1.266, -0.912)
    ),
    "K": np.array(
        (1.831, -0.561, 0.533, -0.277, 1.648)
    ),
    "M": np.array(
        (-0.663, -1.524, 2.219, -1.005, 1.212)
    ),
    "F": np.array(
        (-1.006, -0.59, 1.891, -0.397, 0.412)
    ),
    "P": np.array(
        (0.189, 2.081, -1.628, 0.421, -1.392)
    ),
    "S": np.array(
        (-0.228, 1.399, -4.76, 0.67, -2.647)
    ),
    "T": np.array(
        (-0.032, 0.326, 2.213, 0.908, 1.313)
    ),
    "W": np.array(
        (-0.595, 0.009, 0.672, -2.128, -0.184)
    ),
    "Y": np.array(
        (0.26, 0.83, 3.097, -0.838, 1.512)
    ),
    "V": np.array(
        (-1.337, -0.279, -0.544, 1.242, -1.262)
    ),
}

alphabet_num = {
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
}


class mydataset(Dataset):
    def __init__(self, filename, all_prot_feat):
        self.data = pd.read_csv(filename, sep=',')
        all_tcr_ori_feat = []
        all_tcr_l_feat = []
        all_tcr_pc_feat = []
        all_tcr_evo_feat = []
        all_tcr_masks = []
        all_peptide_ori_feat = []
        all_peptide_l_feat = []
        all_peptide_pc_feat = []
        all_peptide_evo_feat = []
        all_peptide_masks = []
        all_labels = []
        for i in range(len(self.data)):
            TCR = self.data["CDR3"][i].replace(";", "")
            peptide = self.data["Epitope"][i].replace(";", "")
            label = self.data["label"][i]
            peptide_ori_feat = []
            peptide_l_feat = []
            peptide_pc_feat = []
            peptide_evo_feat = []
            peptide_mask = []
            for k in range(20):
                if k < len(peptide):
                    peptide_ori_feat.append(alphabet_num[peptide[k]])
                    peptide_l_feat.append(all_prot_feat[peptide][0][k])
                    peptide_pc_feat.append(list(atchley[peptide[k]]))
                    peptide_evo_feat.append(list(blosum62[peptide[k]]))
                    peptide_mask.append(1)
                else:
                    peptide_ori_feat.append(0)
                    peptide_l_feat.append([0 for j in range(1280)])
                    peptide_pc_feat.append([0 for j in range(5)])
                    peptide_evo_feat.append([0 for j in range(20)])
                    peptide_mask.append(0)
            tcr_ori_feat = []
            tcr_l_feat = []
            tcr_pc_feat = []
            tcr_evo_feat = []
            tcr_mask = []
            for k in range(len(TCR)):
                if TCR[k] == "O" or TCR[k] == 'X':
                    continue
                tcr_ori_feat.append(alphabet_num[TCR[k]])
                tcr_l_feat.append(all_prot_feat[TCR][0][k])
                tcr_pc_feat.append(list(atchley[TCR[k].upper()]))
                tcr_evo_feat.append(list(blosum62[TCR[k].upper()]))
                tcr_mask.append(1)
            for k in range(38 - len(TCR)):
                tcr_ori_feat.append(0)
                tcr_l_feat.append([0 for j in range(1280)])
                tcr_pc_feat.append(([0 for j in range(5)]))
                tcr_evo_feat.append(([0 for j in range(20)]))
                tcr_mask.append(0)

            all_tcr_ori_feat.append(tcr_ori_feat)
            all_tcr_l_feat.append(tcr_l_feat)
            all_tcr_pc_feat.append(tcr_pc_feat)
            all_tcr_evo_feat.append(tcr_evo_feat)
            all_tcr_masks.append(tcr_mask)
            all_peptide_ori_feat.append(peptide_ori_feat)
            all_peptide_l_feat.append(peptide_l_feat)
            all_peptide_pc_feat.append(peptide_pc_feat)
            all_peptide_evo_feat.append(peptide_evo_feat)
            all_peptide_masks.append(peptide_mask)
            all_labels.append(int(label))
        self.tcr_ori_feat = torch.FloatTensor(all_tcr_ori_feat)
        self.tcr_l_feat = torch.FloatTensor(all_tcr_l_feat)
        self.tcr_pc_feat = torch.FloatTensor(all_tcr_pc_feat)
        self.tcr_evo_feat = torch.FloatTensor(all_tcr_evo_feat)
        self.tcr_tokens_masks = torch.LongTensor(all_tcr_masks)
        self.peptide_ori_feat = torch.FloatTensor(all_peptide_ori_feat)
        self.peptide_l_feat = torch.FloatTensor(all_peptide_l_feat)
        self.peptide_pc_feat = torch.FloatTensor(all_peptide_pc_feat)
        self.peptide_evo_feat = torch.FloatTensor(all_peptide_evo_feat)
        self.peptide_tokens_masks = torch.LongTensor(all_peptide_masks)
        self.labels = torch.LongTensor(all_labels)
        self.cnt = self.labels.shape[0]

    def __len__(self):
        return self.cnt

    def __getitem__(self, index):
        return index, self.tcr_ori_feat[index], self.tcr_l_feat[index], self.tcr_pc_feat[index], self.tcr_evo_feat[index],\
               self.peptide_ori_feat[index], self.peptide_l_feat[index], self.peptide_pc_feat[index], self.peptide_evo_feat[index],\
               self.labels[index]
