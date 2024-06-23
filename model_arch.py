import torch
import torch.nn.functional as F
import torch.nn as nn
from ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm


class Mlp(nn.Module):
    """MLP layer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = int(hidden_features) or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ProteinCNN(nn.Module):
    """Protein sequence encoder"""
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0], padding='same')
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1], padding='same')
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2], padding='same')
        self.bn3 = nn.BatchNorm1d(in_ch[3])

    def forward(self, v, cnt, ori=False):
        if ori:
            v = self.embedding(v.long())
        v = v.transpose(2, 1)
        v1 = self.bn1(F.relu(self.conv1(v)))
        v2 = self.bn2(F.relu(self.conv2(v1)))
        v3 = self.bn3(F.relu(self.conv3(v2)))
        v3 = v3.view(v3.size(0), v3.size(2), -1)
        return v3, v2


class TPIBAN_agg(nn.Module):
    """TCR-peptide prediction model"""
    def __init__(self, input_dim, hidden_dim):
        super(TPIBAN_agg, self).__init__()
        self.peptide_extractor = ProteinCNN(input_dim, [hidden_dim, hidden_dim, hidden_dim], [3, 3, 3], True)
        self.tcr_extractor = ProteinCNN(input_dim, [hidden_dim, hidden_dim, hidden_dim], [3, 3, 3], True)
        self.BN_interact_layer = weight_norm(BANLayer(v_dim=hidden_dim, q_dim=hidden_dim, h_dim=hidden_dim*2, h_out=2), name='h_mat', dim=None)
        self.mlp_classifier_layer = Mlp(in_features=hidden_dim*2, hidden_features=hidden_dim, out_features=2, act_layer=nn.GELU, drop=0.)

    def forward(self, v_peptide, v_tcr, ori=False, eval=False):
        feat_list = []
        v_peptide, v_peptide_l2 = self.peptide_extractor(v_peptide, 0, ori)
        v_tcr, v_tcr_l2 = self.tcr_extractor(v_tcr, 0, ori)
        f, att = self.BN_interact_layer(v_peptide, v_tcr)
        out = self.mlp_classifier_layer(f)
        if ori:
            feat_list.append(v_peptide_l2)
            feat_list.append(v_tcr_l2)
        else:
            feat_list.append(v_peptide)
            feat_list.append(v_tcr)
        feat_list.append(f)
        if eval:
            return feat_list, out, att
        return feat_list, out

    def get_logits(self, v_peptide, v_tcr):
        f, att = self.BN_interact_layer(v_peptide, v_tcr)
        out = self.mlp_classifier_layer(f)
        return out


class ConvReg(nn.Module):
    """Convolutional regression for FitNet (feature map layer)"""
    def __init__(self):
        super(ConvReg, self).__init__()
        self.conv = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3, padding='same')
        self.bn = nn.BatchNorm1d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = x.view(x.size(0), x.size(2), -1)
        return x


class CAMKD(nn.Module):
    """Protein sequence encoder"""
    def __init__(self):
        super(CAMKD, self).__init__()
        self.crit_ce = nn.CrossEntropyLoss(reduction='none')
        self.crit_mse = nn.MSELoss(reduction='none')

    def forward(self, trans_feat_s_list, mid_feat_t_list, output_feat_t_list, target):
        bsz = target.shape[0]
        loss_t = [self.crit_ce(logit_t, target) for logit_t in output_feat_t_list]
        num_teacher = len(trans_feat_s_list)
        loss_t = torch.stack(loss_t, dim=0)
        weight = (1.0 - F.softmax(loss_t, dim=0)) / (num_teacher - 1)
        loss_st = []
        for mid_feat_s, mid_feat_t in zip(trans_feat_s_list, mid_feat_t_list):
            tmp_loss_st = self.crit_mse(mid_feat_s, mid_feat_t).reshape(bsz, -1).mean(-1)
            loss_st.append(tmp_loss_st)
        loss_st = torch.stack(loss_st, dim=0)
        loss = torch.mul(weight, loss_st).sum()
        # loss = torch.mul(attention, loss_st).sum()
        loss /= (1.0 * bsz * num_teacher)
        return loss, weight
