from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    """
    This code comes from cvqluu/GE2E-Loss

    References:
        https://github.com/cvqluu/GE2E-Loss/blob/master/ge2e.py
    """
    def __init__(self, nspks: int, putts: int, init_w: float = 10.0, init_b: float = -5.0, loss_method: str = 'softmax', add_norm: bool = True):
        '''
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
        '''
        super(GE2ELoss, self).__init__()
        self.nspks = nspks
        self.putts = putts
        self.add_norm = add_norm

        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.loss_method = loss_method

        assert self.loss_method in ['softmax', 'contrast']

        if self.loss_method == 'softmax':
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == 'contrast':
            self.embed_loss = self.embed_loss_contrast

    def calc_new_centroids(self, dvecs, centroids, spkr, utt):
        '''
        Calculates the new centroids excluding the reference utterance
        '''
        excl = torch.cat((dvecs[spkr,:utt], dvecs[spkr,utt+1:]))
        excl = torch.mean(excl, 0)
        new_centroids = []
        for i, centroid in enumerate(centroids):
            if i == spkr:
                new_centroids.append(excl)
            else:
                new_centroids.append(centroid)
        return torch.stack(new_centroids)

    def calc_cosine_sim(self, dvecs, centroids):
        '''
        Make the cosine similarity matrix with dims (N,M,N)
        '''
        cos_sim_matrix = []
        for spkr_idx, speaker in enumerate(dvecs):
            cs_row = []
            for utt_idx, utterance in enumerate(speaker):
                new_centroids = self.calc_new_centroids(dvecs, centroids, spkr_idx, utt_idx)
                # vector based cosine similarity for speed
                cs_row.append(torch.clamp(torch.mm(utterance.unsqueeze(1).transpose(0,1), new_centroids.transpose(0,1)) / (torch.norm(utterance) * torch.norm(new_centroids, dim=1)), 1e-6))
            cs_row = torch.cat(cs_row, dim=0)
            cos_sim_matrix.append(cs_row)
        return torch.stack(cos_sim_matrix)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        '''
        Calculates the loss on each embedding $L(e_{ji})$ by taking softmax
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                L_row.append(-F.log_softmax(cos_sim_matrix[j,i], 0)[j])
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        ''' 
        Calculates the loss on each embedding $L(e_{ji})$ by contrast loss with closest centroid
        '''
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j,i])
                excl_centroids_sigmoids = torch.cat((centroids_sigmoids[:j], centroids_sigmoids[j+1:]))
                L_row.append(1. - torch.sigmoid(cos_sim_matrix[j,i,j]) + torch.max(excl_centroids_sigmoids))
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs: torch.Tensor, label: Optional[torch.Tensor] = None) -> torch.Tensor:
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        
        Args:
            Input dvecs has shape [N, D], where N = nspks * putts
            #TODO: add label to compute accuracy like voxceleb_trainer
        '''
        if self.add_norm:
            dvecs = F.normalize(dvecs, p=2, dim=1) # 22/06/07 add 2-norm

        dvecs = dvecs.reshape(self.nspks, self.putts, -1) # [N, D] -> [nspks, putts, D]

        #Calculate centroids
        centroids = torch.mean(dvecs, 1)

        #Calculate the cosine similarity matrix
        cos_sim_matrix = self.calc_cosine_sim(dvecs, centroids)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


class TripletLoss(nn.Module):
    def __init__(self, margin: float = 0., add_norm: bool = True, distance: str = 'Euclidean'):
        super().__init__()
        self.margin = margin
        self.add_norm = add_norm
        self.distance = distance
    
    def cosine_similarity(self, s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
        return (s1*s2).sum(dim=-1) / torch.sqrt((s1*s1).sum(dim=-1)*(s2*s2).sum(dim=-1))
    
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

        if self.distance.lower() == 'euclidean':
            dist_pos = self.euclidean_distance(x_anchor, x_pos)
            dist_neg = self.euclidean_distance(x_anchor, x_neg)
        
        elif self.distance.lower() == 'consine':
            dist_pos = self.cosine_similarity(x_anchor, x_pos)
            dist_neg = self.cosine_similarity(x_anchor, x_neg)
        
        else:
            raise NameError
        
        if reduction:
            return torch.mean(torch.max(torch.zeros(x_anchor.shape[0]).to(dist_pos.device), dist_pos - dist_neg + self.margin))

        else:
            return torch.max(torch.zeros(x_anchor.shape[0]), dist_pos - dist_neg + self.margin)
