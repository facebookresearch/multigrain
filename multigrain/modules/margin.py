# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch
import numpy as np


class DistanceWeightedSampling(nn.Module):
    r"""Distance weighted sampling.
    See "sampling matters in deep embedding learning" paper for details.
    Implementation similar to https://github.com/chaoyuaw/sampling_matters
    """
    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4):
        super().__init__()
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    @staticmethod
    def get_distance(x):
        """
        Helper function for margin-based loss. Return a distance matrix given a matrix.
        Returns 1 on the diagonal (prevents numerical errors)
        """
        n = x.size(0)
        square = torch.sum(x ** 2.0, dim=1, keepdim=True)
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        return torch.sqrt(distance_square + torch.eye(n, dtype=x.dtype, device=x.device))

    def forward(self, embedding, target):
        """
        Inputs:
            - embedding: embeddings of images in batch
            - target: id of instance targets

        Outputs:
            - a dict with
               * 'anchor_embeddings'
               * 'negative_embeddings'
               * 'positive_embeddings'
               with sampled embeddings corresponding to anchors, negatives, positives
        """

        B, C = embedding.size()[:2]
        embedding = embedding.view(B, C)

        distance = self.get_distance(embedding)
        distance = torch.clamp(distance, min=self.cutoff)

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(C)) * torch.log(distance)
                       - (float(C - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))
        weights = torch.exp(log_weights - log_weights.max())

        unequal = target.view(-1, 1)
        unequal = (unequal != unequal.t())

        weights = weights * (unequal & (distance < self.nonzero_loss_cutoff)).float()
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.detach().cpu().numpy()
        unequal_np = unequal.cpu().numpy()

        for i in range(B):
            same = (1 - unequal_np[i]).nonzero()[0]

            if np.isnan(np_weights[i].sum()):  # 0 samples within cutoff, sample uniformly
                np_weights_ = unequal_np[i].astype(float)
                np_weights_ /= np_weights_.sum()
            else:
                np_weights_ = np_weights[i]
            try:
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_, replace=False).tolist()
            except ValueError:  # cannot always sample without replacement
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_).tolist()

            for j in same:
                if j != i:
                    a_indices.append(i)
                    p_indices.append(j)

        return {'anchor_embeddings': embedding[a_indices],
                'negative_embeddings': embedding[n_indices],
                'positive_embeddings': embedding[p_indices]}


class MarginLoss(nn.Module):
    r"""Margin based loss.

    Parameters
    ----------
    beta_init: float
        Initial beta
    margin : float
        Margin between positive and negative pairs.
    """
    def __init__(self, beta_init=1.2, margin=0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self._margin = margin

    def forward(self, anchor_embeddings, negative_embeddings, positive_embeddings, eps=1e-8):
        """

        Inputs:
            - input_dict: 'anchor_embeddings', 'negative_embeddings', 'positive_embeddings'

        Outputs:
            - Loss.
        """

        d_ap = torch.sqrt(torch.sum((positive_embeddings - anchor_embeddings) ** 2, dim=1) + eps)
        d_an = torch.sqrt(torch.sum((negative_embeddings - anchor_embeddings) ** 2, dim=1) + eps)

        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        pair_cnt = float(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).item())

        # Normalize based on the number of pairs
        loss = (torch.sum(pos_loss + neg_loss)) / max(pair_cnt, 1.0)

        return loss


class SampledMarginLoss(nn.Module):
    """
    Combines DistanceWeightedSampling + Margin Loss
    """
    def __init__(self, sampling_args={}, margin_args={}):
        super().__init__()
        self.sampling = DistanceWeightedSampling(**sampling_args)
        self.margin = MarginLoss(**margin_args)

    def forward(self, embedding, target):
        sampled_dict = self.sampling(embedding, target)
        loss = self.margin(**sampled_dict)
        return loss

