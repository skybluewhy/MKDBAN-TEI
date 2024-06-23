import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from model_arch import TPIBAN_agg, ConvReg, CAMKD
from sklearn.metrics import roc_auc_score
from data import mydataset
import torch.nn.functional as F


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)


def training_t(net, optimizer, train_loader, cnt, device):
    """Training multi-teacher models"""
    net.train()
    train_loss = 0
    total = 0
    for i, (index, TCR_batch_ori_feat, TCR_batch_l_feat, TCR_batch_pc_feat, TCR_batch_evo_feat,
            peptide_batch_ori_feat, peptide_batch_l_feat, peptide_batch_pc_feat, peptide_batch_evo_feat,
            Y) in enumerate(train_loader):
        pep_feat_list = [peptide_batch_l_feat.to(device), peptide_batch_pc_feat.to(device), peptide_batch_evo_feat.to(device)]
        TCR_feat_list = [TCR_batch_l_feat.to(device), TCR_batch_pc_feat.to(device), TCR_batch_evo_feat.to(device)]
        Y = Y.to(device)
        feat_t, output = net(pep_feat_list[cnt], TCR_feat_list[cnt])
        loss_fct = torch.nn.CrossEntropyLoss(size_average=False)
        agg_xent_loss = loss_fct(output, Y)
        optimizer.zero_grad()
        loss = agg_xent_loss
        loss.backward()
        train_loss += agg_xent_loss.item() * Y.shape[0]
        total = total + Y.shape[0]
        optimizer.step()
    return train_loss / total


class DistillKL(torch.nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t, is_ca=False):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        if is_ca:
            loss = (torch.nn.KLDivLoss(reduction='none')(p_s, p_t) * (self.T**2)).sum(-1)
        else:
            loss = torch.nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss


def training_distillation_normal(teacher_list, net_agg, optimizer, train_loader, device, epoch, pep_module_list, tcr_module_list):
    """Multi-teacher knowledge distillation"""
    net_ds1, net_ds2, net_ds3 = teacher_list[0], teacher_list[1], teacher_list[2]
    net_ds1.eval()
    net_ds2.eval()
    net_ds3.eval()
    model_t_list = [net_ds1, net_ds2, net_ds3]
    teacher_num = len(model_t_list)
    net_agg.train()
    criterion_div = DistillKL(4)
    criterion_kd = CAMKD()
    train_loss = 0
    total = 0
    for i, (index, TCR_batch_ori_feat, TCR_batch_l_feat, TCR_batch_pc_feat, TCR_batch_evo_feat,
            peptide_batch_ori_feat, peptide_batch_l_feat, peptide_batch_pc_feat, peptide_batch_evo_feat, Y) in enumerate(train_loader):
        f_l, output = net_agg(peptide_batch_ori_feat.to(device), TCR_batch_ori_feat.to(device), ori=True)
        pep_feat_list = [peptide_batch_l_feat.to(device), peptide_batch_pc_feat.to(device), peptide_batch_evo_feat.to(device)]
        TCR_feat_list = [TCR_batch_l_feat.to(device), TCR_batch_pc_feat.to(device), TCR_batch_evo_feat.to(device)]
        Y = Y.to(device)
        feat_t_list = []
        logit_t_list = []
        with torch.no_grad():
            index = 0
            for model_t in model_t_list:
                feat_t, logit_t = model_t(pep_feat_list[index], TCR_feat_list[index])
                feat_t = [f.detach() for f in feat_t]
                feat_t_list.append(feat_t)
                logit_t_list.append(logit_t)
                index += 1
        # Cross-entropy loss
        loss_fct = torch.nn.CrossEntropyLoss(size_average=False)
        agg_xent_loss = loss_fct(output, Y)

        # Teacher prediction loss
        criterion_cls_lc = torch.nn.CrossEntropyLoss(reduction='none')
        loss_t_list = [criterion_cls_lc(logit_t, Y) for logit_t in logit_t_list]
        loss_t = torch.stack(loss_t_list, dim=0)
        attention = (1.0 - F.softmax(loss_t, dim=0)) / (teacher_num - 1)
        loss_div_list = [criterion_div(output, logit_t, is_ca=True)
                         for logit_t in logit_t_list]
        loss_div = torch.stack(loss_div_list, dim=0)
        bsz = loss_div.shape[1]
        loss_div = (torch.mul(attention, loss_div).sum()) / (1.0 * bsz * teacher_num)

        # Intermediate teacher features loss
        pep_f_s_list = [regress_s(f_l[0]) for regress_s in pep_module_list]
        tcr_f_s_list = [regress_s(f_l[1]) for regress_s in tcr_module_list]
        pep_f_t_list = [f_t[0] for f_t in feat_t_list]
        tcr_f_t_list = [f_t[1] for f_t in feat_t_list]
        tran_logit_t_list = []
        with torch.no_grad():
            index = 0
            for model_t in model_t_list:
                tran_logit = model_t.get_logits(pep_f_s_list[index], tcr_f_s_list[index])
                tran_logit_t_list.append(tran_logit)
                index += 1
        pep_loss_kd, weight = criterion_kd(pep_f_s_list, pep_f_t_list, logit_t_list, Y)
        tcr_loss_kd, weight = criterion_kd(tcr_f_s_list, tcr_f_t_list, logit_t_list, Y)
        loss_kd = pep_loss_kd + tcr_loss_kd

        optimizer.zero_grad()
        alpha = 1
        beta = 10
        gamma = 0
        if epoch > 20:
            alpha = int(100 - epoch) / \
                        int(100 - 20) * alpha
            gamma = 1 - alpha
        loss = (gamma * agg_xent_loss + alpha * loss_kd + beta * loss_div)
        loss.backward()
        train_loss += loss.item() * Y.shape[0]
        total = total + Y.shape[0]
        optimizer.step()
    return train_loss / total


def evaluate(net, test_loader, device):
    """Evaluate model"""
    net.eval()
    pred_val = []
    y_true_s = []
    with torch.no_grad():
        for i, (index, TCR_batch_ori_feat, TCR_batch_l_feat, TCR_batch_pc_feat, TCR_batch_evo_feat,
                peptide_batch_ori_feat, peptide_batch_l_feat, peptide_batch_pc_feat, peptide_batch_evo_feat,
                Y) in enumerate(test_loader):
            f_l, output = net(peptide_batch_ori_feat.to(device), TCR_batch_ori_feat.to(device), ori=True)
            p = F.softmax(output, dim=1)[:, 1]
            y_true_s = y_true_s + Y.tolist()
            pred_val = pred_val + p.tolist()
        AUC = roc_auc_score(y_true_s, pred_val)
    return AUC


def run_model(args):
    """Training/testing model"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_prot_feat = np.load("./data/esm_features.npy", allow_pickle=True).item()

    if not args.only_test:
        train_filename = args.train_dataset
        train_dataset = mydataset(train_filename, all_prot_feat)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  drop_last=False)
    test_filename = args.test_dataset
    test_dataset = mydataset(test_filename, all_prot_feat)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             drop_last=False)

    teacher_list = []
    teacher_optimizer = []
    teacher_scheduler = []
    teacher_input_dim = [1280, 5, 20]
    pep_module_list = []
    tcr_module_list = []
    stu_trainable_list = torch.nn.ModuleList([])

    # initialize teacher models
    for i in range(3):
        net = TPIBAN_agg(teacher_input_dim[i], 256)
        net.to(device)
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=False, threshold=0.0001, eps=1e-08)
        teacher_list.append(net)
        teacher_optimizer.append(optimizer)
        teacher_scheduler.append(scheduler)

        pep_reg = ConvReg().to(device)
        tcr_reg = ConvReg().to(device)
        stu_trainable_list.append(pep_reg)
        stu_trainable_list.append(tcr_reg)
        pep_module_list.append(pep_reg)
        tcr_module_list.append(tcr_reg)

    # initialize student model
    net = TPIBAN_agg(256, 256)
    net.to(device)
    stu_trainable_list.append(net)
    optimizer = torch.optim.SGD(stu_trainable_list.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20,
                                                           verbose=False, threshold=0.0001, eps=1e-08)

    if args.only_test:
        net.load_state_dict(torch.load(args.save_model, map_location=device))
        AUC = evaluate(net, test_loader, device)
        print('AUC {:.6f}'.format(AUC))
        return

    # Training multi-teacher models
    for epoch in range(30):
        for i in range(3):
            train_loss = training_t(teacher_list[i], teacher_optimizer[i], train_loader, i, device)
            teacher_scheduler[i].step(train_loss)

    # Multi-teacher knowledge distillation
    best_AUC = 0
    counter = 0
    if os.path.exists(args.save_dir + '/checkpoint/checkpoint.pt'):
        print('Load model weights from /checkpoint/checkpoint.pt')
        net.load_state_dict(
            torch.load(args.save_dir + '/checkpoint/checkpoint.pt', map_location=device))
    for epoch in range(args.epoch):
        train_loss = training_distillation_normal(teacher_list, net, optimizer, train_loader, device, epoch,
                                                  pep_module_list, tcr_module_list)
        AUC_val = evaluate(net, test_loader, device)
        AUC = evaluate(net, test_loader, device)
        scheduler.step(AUC)
        print('Epoch {:d} | Train loss {:.6f} | AUC_val {:.6f} | AUC {:.6f}'.format(epoch, train_loss, AUC_val, AUC))
        if (AUC > best_AUC):
            best_AUC = AUC
            torch.save(net.state_dict(), args.save_dir + '/checkpoint/checkpoint.pt')
            counter = 0
        else:
            counter += 1


def parser():
    ap = argparse.ArgumentParser(description='TCR-peptide model')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epoch', type=int, default=50, help='Number of epochs. Default is 100.')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size. Default is 8.')
    ap.add_argument('--repeat', type=int, default=1, help='Repeat the training and testing for N times. Default is 1.')
    ap.add_argument('--lr', default=0.001)
    ap.add_argument('--weight_decay', default=1e-4)
    ap.add_argument('--num_workers', default=0, type=int)
    ap.add_argument('--nFold', default=5, type=int)
    ap.add_argument('--train_dataset', default="./data/train0.csv", type=str)
    ap.add_argument('--test_dataset', default="./data/test0.csv", type=str)
    ap.add_argument('--data_dir', default='../../data/{}/')
    ap.add_argument('--save_dir_format', default='./results_dpp/{}/repeat{}/')
    ap.add_argument('--only_test', default=False, type=bool)
    ap.add_argument('--save_model', default="./checkpoint.pt", type=str)
    args = ap.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    args.dataset = 'Dataset'
    args.data_dir = args.data_dir.format(args.dataset)
    for rp in range(args.repeat):
        print('This is repeat ', rp)
        args.rp = rp
        args.save_dir = args.save_dir_format.format(args.dataset, args.rp)
        print('Save path ', args.save_dir)
        make_dir(args.save_dir)
        make_dir(args.save_dir + '/checkpoint')
        sys.stdout = Logger(args.save_dir + 'log.txt')
        run_model(args)

