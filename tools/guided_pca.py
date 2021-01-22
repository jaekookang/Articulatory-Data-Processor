'''
Guided PCA

2019-08-15
2020-03-22 jaekoo, variable names modified
2020-03-27 jaekoo, fixed for standalone version
2021-01-20 jaekoo, copied from https://github.com/jaekookang/ucm_gem_analysis/tree/master/tools
'''

import os
import numpy as np
from sklearn.decomposition import PCA

# Define factor matrix
# ['T1x','T1y','T2x','T2y','T3x','T3y','T4x','T4y','ULx','ULy','LLx','LLy','MNIx','MNIy']
_factor_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # factor1: JAWy
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # factor2: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # factor3: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],  # factor4: ULx,ULy,LLx,LLy
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # factor5: T1x,T1y,T2x,T2y,T3x,T3y,T4x,T4y
])


class _PCA:
    def __init__(self, n_components):
        '''Standalone/Simplified PCA using sklearn PCA'''
        self.n_components = n_components
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.mean_ = None
        self._pca = PCA(self.n_components)

    def fit(self, data, delete_sklearn_pca=True):
        self._pca.fit(data)
        self.components_ = self._pca.components_  # (n_comp,n_dim)
        self.explained_variance_ = self._pca.explained_variance_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.mean_ = data.mean(axis=0, keepdims=True)
        if delete_sklearn_pca:
            del self._pca  # for pickle saving

    def transform(self, data):
        # (N,n_dim) x (n_dim,n_comp)
        data = data - self.mean_
        return data.dot(self.components_.T)

    def inverse_transform(self, data):
        # (N,n_comp) x (n_comp,n_dim)
        return data.dot(self.components_) + self.mean_


class GuidedPCA:
    def __init__(self, factor_matrix, n_components=3):
        self.n_components = n_components
        self.explained_variance_ratio_ = None
        self.components_ = []
        self.mean_ = None
        self.mat = factor_matrix

        # Make PCA objects
        self.P = [_PCA(n_components=1) for _ in range(self.n_components)]

    def fit(self, data):
        '''Run guided pca iteratively by n_components'''
        self.nrow, self.ncol = data.shape  # NxD
        self._explained_variance_ratio_ = []
        self.mean_ = data.mean(axis=0, keepdims=True)
        # Get PCs iteratively
        resid = data.copy()
        for i in range(self.n_components):
            selected = resid[:, self.mat[i, :] > 0].copy()  # Nxd
            self.P[i].fit(selected)  # PCA
            reduced = self.P[i].transform(selected)  # (Nxd)(dx1)=Nx1
            inverse = self.P[i].inverse_transform(reduced)  # (Nx1)(1xd)=Nxd
            pad = np.zeros(resid.shape)
            pad[:, self.mat[i, :] > 0] = inverse
            inverse = pad
            # Get residual
            resid = resid - inverse
            # Save
            self.components_.append(self.P[i].components_)
        # Compute explained variance ratio (0~1)
        data = data - self.mean_
        SS = np.diag(data.T.dot(data)).sum()
        data_ = self.inverse_transform(self.transform(data))
        data_ = data_ - data_.mean(axis=0, keepdims=True)
        SSreg = np.diag(data_.T.dot(data_)).sum()
        self.explained_variance_ratio_ = round(SSreg/SS, 4)

    def transform(self, data):
        '''Transform data into reduced dimensions'''
        resid = data.copy()
        reduced_data = np.array([]).reshape(resid.shape[0], 0)
        for i in range(self.n_components):
            selected = resid[:, self.mat[i, :] > 0].copy()  # Nxd
            reduced = self.P[i].transform(selected)  # (Nxd)(dx1)=Nx1
            inverse = self.P[i].inverse_transform(
                reduced)  # (Nx1)(1xd)=Nxd
            pad = np.zeros(resid.shape)
            pad[:, self.mat[i, :] > 0] = inverse
            inverse = pad
            # Get residual
            resid = resid - inverse
            reduced_data = np.hstack([reduced_data, reduced])
        return reduced_data

    def inverse_transform(self, data):
        '''Inverse transform of reduced dimensions into original dimensions'''
        reduced_data = data.copy()
        nrow, _ = data.shape
        ncol = self.mean_.shape[1]
        resid = np.zeros((nrow, ncol))
        # Reverse transform
        for i in reversed(range(self.n_components)):
            reduced = reduced_data[:, [i]]
            inverse = self.P[i].inverse_transform(reduced)
            pad = np.zeros(resid.shape)
            pad[:, self.mat[i, :] > 0] = inverse
            inverse = pad
            resid += inverse
        return resid
