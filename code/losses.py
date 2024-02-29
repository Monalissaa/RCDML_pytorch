import torch, os
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from pytorch_metric_learning import miners, losses

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # set random seed for all gpus
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes=range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, prc=0.02, src=0.01):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.prc = prc
        self.src = src

    def forward(self, X, T):#
        P = self.proxies
        cos = F.linear(l2_norm(X), l2_norm(P))
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        ## chuangxin_P
        NumSam_C = torch.sum(P_one_hot, dim=0)
        assist_iterm = torch.exp(torch.where(P_one_hot == 1, cos, torch.zeros_like(cos)).sum(dim=1))
        for i in range(self.nb_classes):
            if NumSam_C[i]==0:
                continue
            w = torch.sum(torch.where(T==i, assist_iterm, torch.zeros_like(assist_iterm)))
            assist_iterm = torch.where(T==i, assist_iterm/w, assist_iterm)
        X_weight = X.mul(torch.unsqueeze(assist_iterm, 1))
        P_update = torch.zeros_like(P)
        for i in range(self.nb_classes):
             if NumSam_C[i] == 0:
                 continue
             P_update[i] = 0.3 * X_weight[T == i].sum(dim=0) + 0.7 * P[i]
        cos = F.linear(l2_norm(X), l2_norm(P_update))

        #chuangxin_R1
        # cos_P = F.linear(l2_norm(P), l2_norm(P))
        cos_P = F.linear(l2_norm(P_update), l2_norm(P_update))
        diag = torch.diag(cos_P)
        a_cos_P = torch.diag_embed(diag)
        cos_P = cos_P - a_cos_P
        m = torch.sum(torch.where(cos_P >= 0, cos_P, torch.zeros_like(cos_P)))
        
        
        # m = torch.sum(torch.topk(cos_P,33).values[:,1:])
        #
        N_one_hot = 1 - P_one_hot
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))
        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)  # The number of positive proxies

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        ## chuangxin_R2
        regul = 0
        for i in range(self.nb_classes):
            data = F.linear(l2_norm(X[T==i]),l2_norm(X[T==i]))
            if min(data.shape) == 0 or min(data.shape) == 1:
                continue
            molecle = torch.exp(-torch.min(data))
            regul = regul + molecle

        # loss = pos_term + neg_term #+ 0.025 * regul + 0.01 * m# + 1.5 * r  +
        # loss = pos_term + neg_term + 0.02 * regul + 0.01 * m
        loss = pos_term + neg_term + self.prc * m + self.src * regul
        return loss


# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes=self.nb_classes, embedding_size=self.sz_embed,
                                             softmax_scale=self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50

        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets='semihard')
        self.loss_func = losses.TripletMarginLoss(margin=self.margin)

    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss


class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings=False)

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss