# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F




def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C


class TMC(nn.Module):

    def __init__(self, classes, lambda_epochs=1):
        """
        :param classes: Number of classification categories
        :param views: Number of views
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMC, self).__init__()
        #self.views = views
        self.classes = classes
        self.lambda_epochs = lambda_epochs
        self.sp=nn.Softplus()
        #self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.classes) for i in range(self.views)])

    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """
        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v]-1
                b[v] = E[v]/(S[v].expand(E[v].shape))
                u[v] = self.classes/S[v]

            # b^0 @ b^(0+1)
            bb = torch.bmm(b[0].view(-1, self.classes, 1), b[1].view(-1, 1, self.classes))
            # b^0 * u^1
            uv1_expand = u[1].expand(b[0].shape)
            bu = torch.mul(b[0], uv1_expand)
            # b^1 * u^0
            uv_expand = u[0].expand(b[0].shape)
            ub = torch.mul(b[1], uv_expand)
            # calculate C
            bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            C = bb_sum - bb_diag

            # calculate b^a
            b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
            # calculate u^a
            u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))

            # calculate new S
            S_a = self.classes / u_a
            # calculate new e_k
            e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            alpha_a = e_a + 1
            return alpha_a
        # #add start
        # if len(alpha) == 0:
        #     raise ValueError("alpha cannot be empty")
        #
        # if len(alpha) == 1:
        #     return alpha[0]
        #
        # alpha_a = alpha[0]
        # for v in range(1, len(alpha)):
        #     alpha_a = DS_Combin_two(alpha_a, alpha[v])
        # #add end

        for v in range(len(alpha)-1):
            if v==0:
                alpha_a = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
        return alpha_a

    def forward(self, evidence, y, global_step):
        #evidence = self.infer(X)
        loss = dict()
        alpha = dict()
        for v_num in range(len(evidence)):
            alpha[v_num] = self.sp(evidence[v_num]) + 1
            #if v_num == len(evidence)-1:
            loss['loss_twc'+str(v_num)]= ce_loss(y, alpha[v_num], self.classes, global_step, self.lambda_epochs).mean()
        # alpha_a = self.DS_Combin(alpha)
        #evidence_a = alpha_a - 1
        #loss['loss_twc_tot']= ce_loss(y, alpha_a, self.classes, global_step, self.lambda_epochs).mean()
        #loss = torch.mean(loss)
        return loss


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    dist = 2 - 2 * torch.mm(x, y.t())
    return dist

def multiceloss(embsx,embsy, gtx,gty):
    
    dist_mat = -euclidean_dist(embsx, embsy)
    N, M = dist_mat.size()  # (bsz, memory)
    
    is_pos = gtx.view(N, 1).expand(N, M).eq(
        gty.expand(N, M)).float()
    
    is_neg = gtx.view(N, 1).expand(N, M).ne(
        gty.expand(N, M)).float()
    
    log_probs = F.log_softmax(dist_mat, dim=1)
    loss = - (is_pos*log_probs).sum(dim=1)
    loss = F.softplus(loss).mean()
    return loss


class APLoss (nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        # Initialize quantizer as non-trainable convolution
        self.quantizer = q = nn.Conv1d(1, 2*nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq-1) / gap
        # First half equal to lines passing to (min+x,1) and (min+x+1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a*min + np.arange(nq, 0, -1))  # b = 1 + a*(min+x)
        # First half equal to lines passing to (min+x,1) and (min+x-1/a,0) with x = {nq-1..0}*gap/(nq-1)
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2-nq, 2, 1) - a*min)  # b = 1 - a*(min+x)
        # First and last one as a horizontal straight line
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        nbs = q.sum(dim=-1)  # number of samples  N x Q = c
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)  # number of correct samples = c+ N x Q
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))  # precision
        rec /= rec.sum(dim=-1).unsqueeze(1)  # norm in [0,1]

        ap = (prec * rec).sum(dim=-1)  # per-image AP

        if ret == '1-mAP':
            if qw is not None:
                ap *= qw  # query weights
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}


class TAPLoss (APLoss):
    """ Differentiable tie-aware AP loss, through quantization. From the paper:
        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589
        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}
        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """
    def __init__(self, nq=25, min=0, max=1, simplified=False):
        APLoss.__init__(self, nq=nq, min=min, max=max)
        self.simplified = simplified

    def forward(self, x, label, qw=None, ret='1-mAP'):
        '''N: number of images;
           M: size of the descs;
           Q: number of bins (nq);
        '''
        assert x.shape == label.shape  # N x M
        N, M = x.shape
        label = label.float()
        Np = label.sum(dim=-1, keepdim=True)

        # Quantize all predictions
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)  # N x Q x M

        c = q.sum(dim=-1)  # number of samples  N x Q = nbs on APLoss
        cp = (q * label.view(N, 1, M)).sum(dim=-1)  # N x Q number of correct samples = rec on APLoss
        C = c.cumsum(dim=-1)
        Cp = cp.cumsum(dim=-1)

        zeros = torch.zeros(N, 1).to(x.device)
        C_1d = torch.cat((zeros, C[:, :-1]), dim=-1)
        Cp_1d = torch.cat((zeros, Cp[:, :-1]), dim=-1)

        if self.simplified:
            aps = cp * (Cp_1d+Cp+1) / (C_1d+C+1) / Np
        else:
            eps = 1e-8
            ratio = (cp - 1).clamp(min=0) / ((c-1).clamp(min=0) + eps)
            aps = cp * (c * ratio + (Cp_1d + 1 - ratio * (C_1d + 1)) * torch.log((C + 1) / (C_1d + 1))) / (c + eps) / Np
        aps = aps.sum(dim=-1)

        assert aps.numel() == N

        if ret == '1-mAP':
            if qw is not None:
                aps *= qw  # query weights
            return 1 - aps.mean()
        elif ret == 'AP':
            assert qw is None
            return aps
        else:
            raise ValueError("Bad return type for APLoss(): %s" % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_tap'+('s' if self.simplified else ''): float(loss)}


class TripletMarginLoss(nn.TripletMarginLoss):
    """ PyTorch's margin triplet loss
    TripletMarginLoss(margin=1.0, p=2, eps=1e-06, swap=False, size_average=True, reduce=True)
    """
    def eval_func(self, dp, dn):
        return max(0, dp - dn + self.margin)


class TripletLogExpLoss(nn.Module):
    r"""Creates a criterion that measures the triplet loss given an input
    tensors x1, x2, x3.
    This is used for measuring a relative similarity between samples. A triplet
    is composed by `a`, `p` and `n`: anchor, positive examples and negative
    example respectively. The shape of all input variables should be
    :math:`(N, D)`.
    The distance is described in detail in the paper `Improving Pairwise Ranking for Multi-Label
    Image Classification`_ by Y. Li et al.
    .. math::
        L(a, p, n) = log \left( 1 + exp(d(a_i, p_i) - d(a_i, n_i) \right)
    Args:
        anchor: anchor input tensor
        positive: positive input tensor
        negative: negative input tensor
    Shape:
        - Input: :math:`(N, D)` where `D = vector dimension`
        - Output: :math:`(N, 1)`
    >>> triplet_loss = nn.TripletLogExpLoss(p=2)
    >>> input1 = autograd.Variable(torch.randn(100, 128))
    >>> input2 = autograd.Variable(torch.randn(100, 128))
    >>> input3 = autograd.Variable(torch.randn(100, 128))
    >>> output = triplet_loss(input1, input2, input3)
    >>> output.backward()
    .. _Learning shallow convolutional feature descriptors with triplet losses:
        http://www.iis.ee.ic.ac.uk/%7Evbalnt/shallow_descr/TFeat_paper.pdf
    """

    def __init__(self, p=2, eps=1e-6, swap=False):
        super(TripletLogExpLoss, self).__init__()
        self.p = p
        self.eps = eps
        self.swap = swap

    def forward(self, anchor, positive, negative):
        assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
        assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
        assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
        assert anchor.dim() == 2, "Input must be a 2D matrix."

        d_p = F.pairwise_distance(anchor, positive, self.p, self.eps)
        d_n = F.pairwise_distance(anchor, negative, self.p, self.eps)

        if self.swap:
            d_s = F.pairwise_distance(positive, negative, self.p, self.eps)
            d_n = torch.min(d_n, d_s)

        dist = torch.log(1 + torch.exp(d_p - d_n))
        loss = torch.mean(dist)
        return loss

    def eval_func(self, dp, dn):
        return np.log(1 + np.exp(dp - dn))


def sim_to_dist(scores):
    return 1 - torch.sqrt(2.001 - 2*scores)


class APLoss_dist (APLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return APLoss.forward(self, d, label, **kw)


class TAPLoss_dist (TAPLoss):
    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return TAPLoss.forward(self, d, label, **kw)