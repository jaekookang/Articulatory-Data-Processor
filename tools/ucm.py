'''
UCM tools

2019-11-04
2019-12-15
2020-03-25 jadkoo, updated

Ref: 
- https://scipy-cookbook.readthedocs.io/items/RankNullspace.html
'''

import numpy as np
from numpy.linalg import svd


def get_weights(X, Y, show_score=False):
    # Linear Regression
    #
    # X: (examples, inp_dim) <= should be centered
    # Y: (examples, out_dim) <= should be centered
    #
    # W: (inp_dim, out_dim)
    # -----------------------
    #  X*w = y
    #  w = X^-1*y
    W = np.dot(np.linalg.pinv(X), Y)
    if show_score:
        ypred = X.dot(W)
        SS_tot = ((Y - Y.mean())**2).sum()
        SS_res = ((Y - ypred)**2).sum()
        # Sum of Squares Regression (=coef of determination)
        SS_reg = 1 - SS_res/SS_tot
        return W, SS_reg
    return W  # 3x2 --> transpose of W is Jacobian matrix


def get_score_lr(W, X, Y):
    # Get Sum-of-Squared Regression for Linear Regression
    # given X, Y
    # W: (inp_dim, out_dim)
    ypred = X.dot(W)
    SS_tot = ((Y - Y.mean())**2).sum()
    SS_res = ((Y - ypred)**2).sum()
    # Sum of Squares Regression (=coef of determination)
    SS_reg = 1 - SS_res/SS_tot
    return SS_reg


def get_score(Y, Y_pred):
    # Get Sum-of-Squared Regression for general
    # given Y, Y_pred
    SS_tot = ((Y - Y.mean())**2).sum()
    SS_res = ((Y - Y_pred)**2).sum()
    # Sum of Squares Regression (=coef of determination)
    SS_reg = 1 - SS_res/SS_tot
    return SS_reg


def get_ucm_cm(weights, normalize_dim=True):
    # weights: (inp_dim, out_dim)
    # out: 
    # - ucm_vec (dim x n)
    # - cm_vec (dim x n)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    ucm_vec, cm_vec = nullspace(weights.T, rangespace=True)  # 5x2, 5x3
    if normalize_dim:
        ucm_vec = normalize(ucm_vec)
        cm_vec = normalize(cm_vec)
    return ucm_vec, cm_vec


def normalize(vec):
    '''Normalize vector norm to 1
    vec: (dim, n)
    '''
    vec = vec.copy()
    for i in range(vec.shape[1]):
        vec[:,i] /= np.linalg.norm(vec[:,i])
    return vec


def _get_ucm_cm(weights, normalize_dim=True):
    # DONT USE THIS
    # weights: (inp_dim, out_dim)
    # out: 
    # - ucm_vec (dim x n)
    # - cm_vec (dim x n)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    ucm_vec, cm_vec = nullspace(weights.T, rangespace=True)  # 5x2, 5x3
    if normalize_dim:
        ucm_vec = normalize(ucm_vec.T).T # <-- this part is wrong
        cm_vec = normalize(cm_vec.T).T
    return ucm_vec, cm_vec


def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.
    The algorithm used by this function is based on the singular value
    decomposition of `A`.
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    Return value
    ------------
    r : int
        The estimated rank of the matrix.
    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0, rangespace=False):
    """Compute an approximate basis for the nullspace of A.
    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    2019-10-21 edited by Jaekoo: rangespace option was added
    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.
    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.
    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.

    rs: (optional) ndarray
        rangespace of `A`
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    if rangespace:
        rs = vh[:nnz].conj().T
        return ns, rs
    else:
        return ns
