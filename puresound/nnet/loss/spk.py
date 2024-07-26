import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCH_PI = torch.acos(torch.zeros(1)).item() * 2


class AAMsoftmax(nn.Module):
    """
    Additive Angular Margin Softmax

    Args:
        embedding_dim: input embedding dimension
        n_classes: number of classes
        margin: loss margin in AAM softmax
        scale: loss scale in AAM softmax
        mp: margin penalty of hard samples
        sub_center: if larger than 1, enable sub-center loss
        sub_center_topk: if > 0, top-k would be used [3]
        sub_center_type: choose in "max"[1] and "avg"[2]

    References:
        [1] https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf
        [2] https://arxiv.org/pdf/2407.04291v1
        [3] https://arxiv.org/pdf/2110.05042
    """

    def __init__(
        self,
        embedding_dim: int,
        n_classes: int,
        margin: float = 0.2,
        scale: int = 30,
        mp: float = 0.,
        sub_center: int = 1,
        sub_center_topk: Optional[int] = 0,
        sub_center_type: str = "max",
    ) -> None:
        super().__init__()
        self.m = margin
        self.s = scale
        self.n_classes = n_classes
        self.sub_center = sub_center
        self.sub_center_topk = sub_center_topk
        self.sub_center_type = sub_center_type.lower()
        assert self.sub_center_type in ["max", "avg"]
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(n_classes * sub_center, embedding_dim), requires_grad=True
        )
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.weight, gain=1)

        self.cos_m = torch.cos(torch.tensor(self.m))
        self.sin_m = torch.sin(torch.tensor(self.m))

        if margin > 0.001:
            mp = mp * (margin / 0.2)
        else:
            mp = 0.
        
        self.cos_mp = torch.cos(torch.tensor(mp))
        self.sin_mp = torch.sin(torch.tensor(mp))

        # make the function cos(theta+m) monotonic decreasing while theta in [0°, 180°]
        self.th = torch.cos(TORCH_PI - torch.tensor(self.m))
        self.mm = torch.sin(TORCH_PI - torch.tensor(self.m)) * self.m

        print("Initialised AAMSoftmax margin %.3f scale %.3f" % (self.m, self.s))

    def forward(self, x: torch.Tensor, label: torch.Tensor):
        if label.dim() == 2:
            label = label.squeeze(1)  # [N]

        assert x.shape[0] == label.shape[0]

        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight.to(x.device)))
        if self.sub_center != 1:
            cosine = torch.reshape(cosine, (-1, self.n_classes, self.sub_center))
            if self.sub_center_type == "max":
                cosine = torch.max(cosine, 2)[0]
            elif self.sub_center_type == "avg":
                cosine = cosine * (torch.softmax(cosine, dim=2))
                cosine = cosine.sum(dim=-1)
        
        # cos(theta + m)
        sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        phi_mp = cosine * self.cos_mp + sine * self.sin_mp

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        if self.sub_center_topk > 0:
            topk_idx = torch.topk(cosine - 2 * one_hot, self.sub_center_topk)[1]
            topk_one_hot = torch.zeros_like(cosine).scatter_(1, topk_idx, 1)
            output = (one_hot * phi) + (topk_one_hot * phi_mp) + ((1 - one_hot - topk_one_hot) * cosine)
        
        else:
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        output = output * self.s

        loss = self.ce(output, label)
        return loss


class SphereFace2(nn.Module):
    """
    Implement of sphereface2 for speaker verification:
    Reference:
        [1] Exploring Binary Classification Loss for Speaker Verification https://ieeexplore.ieee.org/abstract/document/10094954
        [2] Sphereface2: Binary classification is all you need for deep face recognition https://arxiv.org/pdf/2108.01513

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale: norm of input feature
        margin: margin
        lanbuda: weight of positive and negative pairs
        t: parameter for adjust score distribution
        margin_type: A:cos(theta+margin) or C:cos(theta)-margin
    Recommend margin:
        training: 0.2 for C and 0.15 for A
        LMF: 0.3 for C and 0.25 for A
    """

    def __init__(
        self,
        in_features,
        out_features,
        scale=32.0,
        margin=0.2,
        lanbuda=0.7,
        t=3,
        margin_type="C",
        sub_center: int = 1,
    ):
        super(SphereFace2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.sub_center = sub_center
        self.weight = nn.Parameter(
            torch.FloatTensor(out_features * sub_center, in_features)
        )
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.t = t
        self.lanbuda = lanbuda
        self.margin_type = margin_type

        ########
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)
        ########

    def update(self, margin=0.2):
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin)
        self.mmm = 1.0 + math.cos(math.pi - margin)

    def fun_g(self, z, t: int):
        gz = 2 * torch.pow((z + 1) / 2, t) - 1
        return gz

    def forward(self, input: torch.Tensor, label: torch.Tensor):
        # compute similarity
        cos = F.linear(F.normalize(input), F.normalize(self.weight))
        if self.sub_center != 1:
            cos = torch.reshape(cos, (-1, self.out_features, self.sub_center))
            cos = torch.max(cos, 2)[0]

        if self.margin_type == "A":  # arcface type
            sin = torch.sqrt(1.0 - torch.pow(cos, 2))
            cos_m_theta_p = (
                self.scale
                * self.fun_g(
                    torch.where(
                        cos > self.th,
                        cos * self.cos_m - sin * self.sin_m,
                        cos - self.mmm,
                    ),
                    self.t,
                )
                + self.bias[0][0]
            )
            cos_m_theta_n = (
                self.scale * self.fun_g(cos * self.cos_m + sin * self.sin_m, self.t)
                + self.bias[0][0]
            )
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))
        else:  # cosface type
            cos_m_theta_p = (
                self.scale * (self.fun_g(cos, self.t) - self.margin) + self.bias[0][0]
            )
            cos_m_theta_n = (
                self.scale * (self.fun_g(cos, self.t) + self.margin) + self.bias[0][0]
            )
            cos_p_theta = self.lanbuda * torch.log(1 + torch.exp(-1.0 * cos_m_theta_p))
            cos_n_theta = (1 - self.lanbuda) * torch.log(1 + torch.exp(cos_m_theta_n))

        target_mask = input.new_zeros(cos.size())
        target_mask.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget_mask = 1 - target_mask
        # cos1 = (cos - self.margin) * target_mask + cos * nontarget_mask
        # output = self.scale * cos1  # for computing the accuracy
        loss = (target_mask * cos_p_theta + nontarget_mask * cos_n_theta).sum(1).mean()

        # return output, loss
        return loss
