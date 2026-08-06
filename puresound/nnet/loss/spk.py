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


class GE2ELoss(nn.Module):
    """
    This code comes from cvqluu/GE2E-Loss

    References:
        https://github.com/cvqluu/GE2E-Loss/blob/master/ge2e.py
    """

    def __init__(
        self,
        nspks: int,
        putts: int,
        init_w: float = 10.0,
        init_b: float = -5.0,
        loss_method: str = "softmax",
        add_norm: bool = True,
    ):
        """
        Implementation of the Generalized End-to-End loss defined in https://arxiv.org/abs/1710.10467 [1]
        Accepts an input of size (N, M, D)
            where N is the number of speakers in the batch,
            M is the number of utterances per speaker,
            and D is the dimensionality of the embedding vector (e.g. d-vector)
        
        Args:
            - nspks: Number of speakers in each bath (PureSound added)
            - putts: Number of utterances per speaker (PureSound added)
            - init_w (float): defines the initial value of w in Equation (5) of [1]
            - init_b (float): definies the initial value of b in Equation (5) of [1]
            - add_norm (bool): add 2-norm on input dvec
        """
        super(GE2ELoss, self).__init__()
        self.nspks = nspks
        self.putts = putts
        self.add_norm = add_norm

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        """
        Calculates the new centroids excluding the reference utterance
        """
        excl = torch.cat((dvecs[spkr, :utt], dvecs[spkr, utt + 1 :]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        """
        Make the cosine similarity matrix with dims (N,M,N)
        """
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(
                    dvecs, centroids, spkr_idx, utt_idx
                )
                # vector based cosine similarity for speed
                cs_row.append(
                    torch.clamp(
                        torch.mm(
                            utterance.unsqueeze(1).transpose(0, 1),
                            new_centroids.transpose(0, 1),
                        )
                        / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)),
                        1e-6,
                    )
                )
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j, i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """ 
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        """
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(
        self, dvecs: torch.Tensor, label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        
        Args:
            Input dvecs has shape [N, D], where N = nspks * putts
            #TODO: add label to compute accuracy like voxceleb_trainer
        """
        if self.add_norm:
            dvecs = F.normalize(dvecs, p=2, dim=1)  # 22/06/07 add 2-norm

        dvecs = dvecs.reshape(self.nspks, self.putts, -1)  # [N, D] -> [nspks, putts, D]

        # Calculate centroids
        centroids = torch.mean(dvecs, 1)

        # Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


class TripletLoss(nn.Module):
    def __init__(
        self, margin: float = 0.0, add_norm: bool = True, distance: str = "Euclidean"
    ):
        super().__init__()
        self.margin = margin
        self.add_norm = add_norm
        self.distance = distance

    def cosine_similarity(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return (s1 * s2).sum(dim=-1) / torch.sqrt(
            (s1 * s1).sum(dim=-1) * (s2 * s2).sum(dim=-1)
        )

    def euclidean_distance(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((s1 - s2).pow(2).sum(dim=-1) + 1e-8)

    def forward(self, x: torch.Tensor, reduction: bool = True) -> torch.Tensor:
        """
        Args:
            input x tensor has shape [N, 3, C] of meaning [Anchor, Postive, Negative] sample separately.
        
        Returns:
            loss tensor
        """
        assert x.shape[1] == 3

        if self.add_norm:
            x = F.normalize(x, p=2, dim=-1)

        x_anchor = x[:, 0, :]
        x_pos = x[:, 1, :]
        x_neg = x[:, 2, :]

        if self.distance.lower() == "euclidean":
            dist_pos = self.euclidean_distance(x_anchor, x_pos)
            dist_neg = self.euclidean_distance(x_anchor, x_neg)

        elif self.distance.lower() == "cosine":
            # The hinge below is written for a distance (small = similar), so
            # the similarity has to be converted; feeding cosine similarity in
            # directly would invert the objective and push the anchor away
            # from its positive. 1 - cos lands in [0, 2], 0 when identical.
            dist_pos = 1.0 - self.cosine_similarity(x_anchor, x_pos)
            dist_neg = 1.0 - self.cosine_similarity(x_anchor, x_neg)

        else:
            raise NameError

        if reduction:
            return torch.mean(
                torch.max(
                    torch.zeros(x_anchor.shape[0]).to(dist_pos.device),
                    dist_pos - dist_neg + self.margin,
                )
            )

        else:
            return torch.max(
                torch.zeros(x_anchor.shape[0]), dist_pos - dist_neg + self.margin
            )
