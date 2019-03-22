# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from sklearn.decomposition import PCA
import torch
import numpy as np


def get_whiten(X):
    pca = PCA(whiten=True)
    pca.fit(X.detach().cpu().numpy())
    m = torch.tensor(pca.mean_, dtype=torch.float)
    P = torch.tensor(pca.components_.T / np.sqrt(pca.explained_variance_), dtype=torch.float)
    return m, P
