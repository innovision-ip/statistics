"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2024-05-28 21:14:14
Last Edited : 2024-08-15 12:22:38
Last Author : Jie Li
File Path   : /undefined/Users/Jie/Documents/dcm_IP/dcm/utils.py
Description :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

import copy
import time

import numpy as np
import scipy.io as sio
from numpy import cos, sin, sqrt
from numpy.linalg import LinAlgError, cholesky, svd
from scipy.linalg import expm, inv, toeplitz
from scipy.signal import detrend
from scipy.sparse import issparse
from scipy.special import gamma
from scipy.stats import norm

from defaults import defaults


def vectorise_object(*x):
    """
    Vectorise a python object. The function is equivalent to `spm_vec' in SPM12.
    By default, the 2D arrays are vectorised in column-major order.

    Parameters
    ----------
    *x : tuple
        Variable length argument list. The objects to be vectorised.

    Returns
    -------
    np.ndarray
        The vectorised object.

    Examples
    --------
    >>> vectorise_object(np.array([[1, 2], [3, 4]]), np.array([4, 98]), False)
    >>> vectorise_object(np.array([[1, 2], [3, 4]]), [4, 98], False)

    >>> a = np.array(np.arange(12)).reshape((2, 2, 3), order="F") + 1
    >>> a = np.transpose(a, (1, 0, 2))
    >>> vectorise_object(a)
    """
    # rest of the function...
    # If multiple arguments are provided, convert them into a list
    x = copy.deepcopy(x)
    if len(x) > 1:
        x = list(x)
    else:
        x = x[0]

    # vectorize numerical arrays
    if isinstance(x, np.ndarray):
        return x.flatten("F")

    if isinstance(x, (int, float, complex)):
        return x

    # vectorize logical arrays
    elif isinstance(x, np.bool_):
        return x.flatten("F")

    elif isinstance(x, bool):
        return x

    # vectorize dictionary into list arrays
    elif isinstance(x, dict):
        vx = np.array([])
        for key in x:
            vx = np.concatenate((vx, vectorise_object(x[key])), axis=None)
        return vx

    # vectorize lists into numerical arrays
    elif isinstance(x, list):
        vx = np.array([])
        for i in x:
            vx = np.concatenate((vx, vectorise_object(i)), axis=None)
        return vx

    else:
        return np.array([])


def unvectorise_object(vX, *X):
    """
    Unvectorise an object that was previously vectorised.

    This function takes a vectorised object and a template object, and returns the object in its original, unvectorised form.

    Parameters
    ----------
    vX : array_like
        The vectorised object.
    *X : object
        The template object. This should be the same type and shape as the original object before it was vectorised.

    Returns
    -------
    object
        The unvectorised object. This will be the same type and shape as the template object.

    Examples
    --------
    >>> a = np.eye(2)
    >>> a_vec = vectorise_object(a)
    >>> a_rec = unvectorise_object(a_vec, a)

    >>> b = vectorise_object(np.array([[1, 2], [3, 4]]), [4, 98], False)
    >>> b_rec = unvectorise_object(b, np.array([[1, 2], [3, 4]]), [4, 98], False)

    >>> b_dict = {
    ...     "a1": 3,
    ...     "b1": 4,
    ...     "c1": np.array([[1.0, 2.0], [3.0, 4.0]]),
    ...     "d1": False,
    ...     "e1": [4, 5],
    ...     "f1": {"a": 3, "b": 4},
    ... }
    >>> b_dict_vec = vectorise_object(b_dict)
    >>> b_dict_rec = unvectorise_object(b_dict_vec, b_dict)

    >>> c = [[3, 4], b_dict]
    >>> c_dict_vec = vectorise_object(c)
    >>> c_dict_rec = unvectorise_object(c_dict_vec, c)
    """

    # If multiple arguments are provided, convert them into a list
    X = copy.deepcopy(X)
    vX = copy.deepcopy(vX)
    if len(X) > 1:
        X = list(X)
    else:
        X = X[0]

    if isinstance(X, int):
        return int(vX)

    if isinstance(X, (float, complex)):
        return vX

    if isinstance(X, bool):
        return bool(vX)

    def process_element(x, vX):
        if isinstance(x, bool) and len(vX) > 0:
            return bool(vX[0]), vX[1:]
        elif isinstance(x, int) and len(vX) > 0:
            return int(vX[0]), vX[1:]
        elif isinstance(x, (float, complex)) and len(vX) > 0:
            return vX[0], vX[1:]
        elif isinstance(x, list):
            result = []
            for item in x:
                processed_item, vX = process_element(item, vX)
                result.append(processed_item)
            return result, vX
        elif isinstance(x, dict):
            result = {}
            for key, value in x.items():
                processed_value, vX = process_element(value, vX)
                result[key] = processed_value
            return result, vX
        elif isinstance(x, np.ndarray):
            n = x.size
            reshaped, vX = vX[:n], vX[n:]
            return reshaped.reshape(x.shape, order="F"), vX
        return x, vX

    # Start the processing
    result, _ = process_element(X, vX)
    return result


def concatenate_dod_lol(x):
    """
    Convert a dictionary of dictionaries or a list of lists into a matrix.
    Empty list elements are replaced by zero partitions. Each element of dictionary of dictionaries (or list of lists) must be a 2d numpy array or empty list ‘[]’., if it is a 1d numpy array, then it must be converted to a 2d numpy array.

    Parameters
    ----------
    x : dict or list
        The input data structure to be converted. It can be a dictionary of dictionaries or a list of lists.

    Returns
    -------
    numpy.ndarray
        The resulting matrix after the conversion.

    Raises
    ------
    ValueError
        If the input is a dictionary but not a dictionary of dictionaries, or if the input is a list but not a list of lists.

    Examples
    --------
    >>> # dictionary of dictionaries
    >>> x = {}
    >>> x[0] = {0: np.array([1]).reshape((1, 1)), 1: np.array([[1, 2]]), 2: []}
    >>> x[1] = {0: [], 1: np.eye(2) * 2, 2: np.zeros((2, 3))}
    >>> x[2] = {0: [], 1: [], 2: np.ones((3, 3))}
    >>> spm_cat(x)

    >>> # list of lists
    >>> x1 = []
    >>> x1.append([np.array([[1]]), np.array([[1, 2]]), []])
    >>> x1.append([np.zeros((2, 1)), np.eye(2) * 2, np.zeros((2, 3))])
    >>> x1.append([np.zeros((3, 1)), np.zeros((3, 2)), np.ones((3, 3))])
    >>> spm_cat(x1)
    """
    # rest of the function...
    # x = copy.deepcopy(x)  # create a deep copy of x
    # check x is not already a matrix
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (dict, list)):
        if isinstance(x, dict):
            for i in x:
                if not isinstance(x[i], dict):
                    raise ValueError(
                        "The input dictionary must be a dictionary of dictionaries"
                    )
        elif isinstance(x, list):
            for i in x:
                if not isinstance(i, list):
                    raise ValueError("The input list must be a list of lists")

    if isinstance(x, dict):
        I = {}
        for i in x:
            max_val_i = 0
            for j in x[i]:
                if isinstance(x[i][j], np.ndarray):
                    shape = np.shape(x[i][j])
                    if len(shape) > 1 and shape[0] > max_val_i:
                        max_val_i = shape[0]
            I[i] = max_val_i
        J = {}
        first_dict = x[next(iter(x))]
        for j in first_dict:
            max_val_j = 0
            for i in x:
                if isinstance(x[i][j], np.ndarray):
                    shape = np.shape(x[i][j])
                    if len(shape) > 1 and shape[1] > max_val_j:
                        max_val_j = shape[1]
            J[j] = max_val_j
    elif isinstance(x, list):
        I = []
        for i, row in enumerate(x):
            max_val_i = 0
            for j, item in enumerate(row):
                if isinstance(item, np.ndarray):
                    shape = np.shape(item)
                    if len(shape) > 1 and shape[0] > max_val_i:
                        max_val_i = shape[0]
            I.append(max_val_i)
        J = []
        for j, _ in enumerate(x[0]):
            max_val_j = 0
            for i, row in enumerate(x):
                if isinstance(row[j], np.ndarray):
                    shape = np.shape(row[j])
                    if len(shape) > 1 and shape[1] > max_val_j:
                        max_val_j = shape[1]
            J.append(max_val_j)

    # empty partitions
    # pylint: disable=consider-using-enumerate
    for i in range(len(x)):
        for j in range(len(x[0])):
            if not isinstance(x[i][j], np.ndarray) or x[i][j].size == 0:
                x[i][j] = np.zeros((I[i], J[j]))
    # pylint: enable=consider-using-enumerate

    # concatenate
    if isinstance(x, dict):
        y = {i: np.concatenate(list(x[i].values()), axis=1) for i in x}
        return np.concatenate(list(y.values()), axis=0)

    if isinstance(x, list):
        y = [np.concatenate(x[i], axis=1) for i in range(len(x))]
        return np.concatenate(y, axis=0)

    return None


def compute_dfdx(f, f0, dx):
    """
    Compute numerical differences.

    Parameters
    ----------
    f : array_like, struct or cell
        The function values at the current point.
    f0 : array_like, struct or cell
        The function values at the previous point.
    dx : float
        The difference in x between the current and previous point.

    Returns
    -------
    dfdx : array_like, struct or cell
        The numerical differences.

    """
    if isinstance(f, list):
        dfdx = [compute_dfdx(f_i, f0_i, dx) for f_i, f0_i in zip(f, f0)]
    elif isinstance(f, dict):
        dfdx = (vectorise_object(f) - vectorise_object(f0)) / dx
    else:
        dfdx = (f - f0) / dx
    return dfdx


def get_default(key):
    return defaults.get(key, None)


def specify_dcm_fmri_priors(A, B, C, D, options):
    """
    Returns the priors for a two-state DCM for fMRI.

    Parameters
    ----------
    A, B, C, D : ndarray
        Constraints on connections (1 - present, 0 - absent)
    options : dict
        Options for the model. Keys:
            'two_state': (0 or 1) one or two states per region
            'stochastic': (0 or 1) exogenous or endogenous fluctuations
            'precision': log precision on connection rates

    Returns
    -------
    pE : dict
        Prior expectations (connections and hemodynamic)
    pC : dict
        Prior covariances (connections and hemodynamic)
    x : ndarray
        Prior (initial) states
    C : ndarray
        Prior variances (in struct form)
    """
    # number of regions
    n = A.shape[0]

    # check options and D (for nonlinear coupling)
    options.setdefault("stochastic", 0)
    options.setdefault("induced", 0)
    options.setdefault("two_state", 0)
    options.setdefault("backwards", 0)
    D = np.zeros((n, n, 0)) if D is None else D

    # connectivity priors and initial states
    if options["two_state"]:
        # (6) initial states
        x = np.zeros((n, 6))
        A = np.array(A, dtype=bool)

        # precision of log-connections (two-state)
        try:
            pA = np.exp(options["precision"])
        except KeyError:
            pA = 16

        # prior expectations and variances
        pE = {"A": A * 32 - 32, "B": B * 0, "C": C * 0, "D": D * 0}

        # prior covariances
        pC = {"A": A / pA, "B": B / 4, "C": C * 4, "D": D / 4}

        # excitatory proportion
        if options.get("backwards"):
            pEA = np.zeros((3, 3, 2))
            pCA = np.zeros((3, 3, 2))
            pEA[:, :, 0] = pE["A"]
            pCA[:, :, 0] = pC["A"]
            pEA[:, :, 1] = A * 0
            pCA[:, :, 1] = A / pA
            pE["A"] = pEA
            pC["A"] = pCA

    else:
        # one hidden state per node
        # (6 - 1) initial states
        x = np.zeros((n, 5))

        # precision of connections (one-state)
        try:
            pA = np.exp(options["precision"])
        except KeyError:
            pA = 64
        dA = options.get("decay", 1)

        # prior expectations_deg
        A = np.array(A, dtype=bool)
        if A.ndim == 1:
            pE = {"A": (A - 1) * dA, "B": B * 0, "C": C * 0, "D": D * 0}
        else:
            pE = {"A": A / 128, "B": B * 0, "C": C * 0, "D": D * 0}

        # prior covariances
        if A.ndim == 1:
            pC = {"A": A, "B": B, "C": C, "D": D}
        else:
            pC = {"A": A / pA, "B": B, "C": C, "D": D}

    # and add hemodynamic priors

    pE.update({"transit": np.zeros(n), "decay": np.zeros(1), "epsilon": np.zeros(1)})
    pC.update(
        {
            "transit": np.ones(n) / 256,
            "decay": np.ones(1) / 256,
            "epsilon": np.ones(1) / 256,
        }
    )

    # add prior on spectral density of fluctuations (amplitude and exponent)
    if options["induced"]:
        pE.update({"a": np.zeros(2), "b": np.zeros(2), "c": np.zeros(n)})
        pC.update({"a": np.ones(2) / 64, "b": np.ones(2) / 64, "c": np.ones(n) / 64})

    # prior covariance matrix
    pC = np.diag(vectorise_object(pC))

    return pE, pC, x


def truncate_svd(X, tol=1e-6):
    if tol >= 1:
        tol = tol - 1e-6
    if tol <= 0:
        tol = 64 * np.finfo(float).eps

    # Preprocess the matrix X
    M, N = X.shape
    p = np.where(np.any(X, axis=1))[0]
    q = np.where(np.any(X, axis=0))[0]
    X = X[np.ix_(p, q)]

    # Perform SVD
    i, j = np.nonzero(X)
    s = X[i, j]
    m, n = X.shape
    if np.any(i - j):
        # Full SVD for off-leading diagonal elements
        if m > n:
            v, S_diag, vT = np.linalg.svd(X.T @ X, full_matrices=False)
            S = np.diag(np.sqrt(S_diag))
            j = np.where(S_diag * len(S_diag) / np.sum(S_diag) > tol)[0]
            v = v[:, j]
            u = X @ v / np.sqrt(S_diag[j])
            S = np.sqrt(S[j, j])
        elif m < n:
            u, S_diag, _ = np.linalg.svd(X @ X.T, full_matrices=False)
            S = np.diag(np.sqrt(S_diag))
            j = np.where(S_diag * len(S_diag) / np.sum(S_diag) > tol)[0]
            u = u[:, j]
            v = X.T @ u / np.sqrt(S_diag[j])
            S = np.sqrt(S[j, j])
        else:
            u, S_diag, vT = np.linalg.svd(X, full_matrices=False)
            S = np.diag(S_diag)
            j = np.where(S_diag**2 * len(S_diag) / np.sum(S_diag**2) > tol)[0]
            v = vT.T[:, j]
            u = u[:, j]
            S = S[j, j]
    else:
        S = np.diag(s)
        u = np.eye(m, n)
        v = np.eye(m, n)
        j = np.argsort(-s, kind="stable")
        S = S[j, :][:, j]
        v = v[:, j]
        u = u[:, j]
        s = np.diag(S) ** 2
        j = np.where(s * len(s) / np.sum(s) > tol)[0]
        v = v[:, j]
        u = u[:, j]
        S = S[j, :][:, j]

    # Replace in full matrices
    j = len(j)
    U = np.zeros((M, j))
    V = np.zeros((N, j))
    if j:
        U[p, :] = u
        V[q, :] = v

    return U, S, V


def get_object_length(X):
    """
    Length of a vectorised numeric, list or dictionary
    :param X: numeric, list or dictionary
    :return: length of vectorised X
    """
    if isinstance(X, (int, float, bool, np.ndarray)):
        # vectorise numerical or logical arrays
        return np.size(X)
    elif isinstance(X, dict):
        # vectorise dictionary into list
        n = 0
        for key in X:
            n += get_object_length(X[key])
        return n
    elif isinstance(X, list):
        # vectorise list into numerical arrays
        n = 0
        for i in range(len(X)):
            n += get_object_length(X[i])
        return n
    else:
        return 0


def get_dfdx_cat(J):
    """
    Concatenate into a matrix.

    Parameters
    ----------
    J : dictionary, 1D
        The input dictionary. Each value in the inner dictionaries should be a numpy array.

    Returns
    -------
    numpy.ndarray
        The concatenated matrix.

    Examples
    --------
    >>> J = {}
    >>> J[0] = 1 * np.ones((1, 2))
    >>> J[1] = 2 * np.ones((1, 2))
    >>> get_dfdx_cat(J)

    """
    # get the first value of the first dictionary in J
    first_element = list(J.values())[0]

    # check if the first element is a vector
    if first_element.ndim == 1 or any(dim == 1 for dim in first_element.shape):
        values = [value.reshape(-1, 1) for value in J.values()]
        matrix = np.hstack(values)
        return matrix
    else:
        return J


def get_dfdx(f, f0, dx):
    """
    numerical differences
    """
    if isinstance(f, (list, dict)):
        dfdx = f.copy()
        # pylint: disable=consider-using-enumerate
        for i in range(len(f)):
            dfdx[i] = get_dfdx(f[i], f0[i], dx)
        # pylint: enable=consider-using-enumerate
    else:
        dfdx = (f - f0) / dx
    return dfdx


def get_diff(*args):
    """
    Matrix high-order numerical differentiation, rewrite based on spm_diff in SPM12.
    Parameters
    ----------
    f : callable
        Function to differentiate.
    x : array_like
        Input arguments to the function `f`.
    n : int or array_like
        Arguments to differentiate with respect to.
    V : list, optional
        Array of matrices for differentiation with respect to a linear transformation of the parameters.
    q : str, optional
        Flag to preclude default concatenation of dfdx, use `nocat`.

    Returns
    -------
    dfdx : array_like
        Derivative of `f` with respect to `x`.
            dfdx  = (f(x + dx)- f(x))/dx

    Examples
    --------
    >>> def myexp(a, b):
        return np.exp(a) + np.exp(b)
    >>> [aa, bb, cc] = get_diff(myexp, np.array([3, 4]), np.array([1, 2]), np.array([1, 2]))
    """

    # step size for numerical derivatives
    dx = np.exp(-8)
    args = list(args)
    # create function handle
    f = args[0]

    # parse input arguments
    if isinstance(args[-1], list):
        x = args[1:-2]
        n = np.asarray(args[-2], dtype=int)
        V = args[-1]
        q = True
    elif isinstance(args[-1], np.ndarray):
        x = args[1:-1]
        n = args[-1]
        V = [None] * len(x)
        q = True
    elif isinstance(args[-1], (int, float)):
        x = args[1:-1]
        n = np.array([args[-1]])
        V = [None] * len(x)
        q = True
    elif isinstance(args[-1], str):
        x = args[1:-2]
        if isinstance(args[-2], (int, float)):
            n = np.array([args[-2]])
        else:
            n = args[-2]
        V = [None] * len(x)
        q = True
    else:
        raise ValueError("Improper call.")

    # check transform matrices V = dxdy
    V.extend([None] * (len(x) - len(V)))  # Extend V to match the length of x if needed
    for i, xi in enumerate(x):
        if V[i] is None and any(i == (n - 1)):
            V[i] = np.eye(get_object_length(xi))
    # for i, xi in enumerate(x):
    #     try:
    #         V[i]
    #     except Exception:
    #         V[i] = None
    #     if V[i] is None and any(i == (n - 1)):
    #         V[i] = np.eye(get_object_length(xi))

    # initialise
    if isinstance(n, int):
        m = n - 1
    elif isinstance(n, (list, np.ndarray)):
        m = n[-1] - 1
    xm = vectorise_object(x[m])
    J = {i: None for i in range(V[m].shape[1])}

    # proceed to derivatives
    if isinstance(n, (int, float, complex)) or n.size == 1:
        # dfdx
        f0 = f(*x)
        for i in J.keys():
            xi = copy.deepcopy(x[:])
            xi[m] = unvectorise_object(xm + V[m][:, i] * dx, x[m])
            J[i] = get_dfdx(f(*xi), f0, dx)

        # return numeric array for first-order derivatives
        f = vectorise_object(f0)

        # if there are no arguments to differentiate w.r.t. ...
        if xm.size == 0:
            J = np.zeros((len(f), 0))
        # or there are no arguments to differentiate
        elif f.size == 0:
            J = np.zeros((0, len(xm)))

        # differentiation of a scalar or vector
        if isinstance(f0, (int, float, np.ndarray)) and isinstance(J, dict) and q:
            J = get_dfdx_cat(J)

        # assign output argument and return
        return [J, f0]

    else:
        # dfdxdxdx....
        # f0 = [None] * len(n)
        f0 = get_diff(f, *x, n[:-1], V)
        p = True

        for i in J.keys():
            xi = copy.deepcopy(x[:])
            xmi = xm + V[m][:, i] * dx
            xi[m] = unvectorise_object(xmi, x[m])
            fi = get_diff(f, *xi, n[:-1], V)
            J[i] = get_dfdx(fi[0], f0[0], dx)
            p = p and isinstance(J[i], (int, float, np.ndarray))

        # or differentiation of a scalar or vector
        if p and q:
            J = get_dfdx_cat(J)

        # assign output argument and return
        return [J] + f0


def generate_2d_dict(m, n):
    """
    Initialize a 2D dictionary of dictionaries with empty numpy arrays as inner values.
    :param m: Number of rows
    :param n: Number of columns
    :return: 2D dictionary of dictionaries
    """
    data = {}
    for i in range(m):
        data[i] = {}
        for j in range(n):
            data[i][j] = np.array([])
    return data


def get_diag(X, K=0):
    """
    Diagonal matrices and diagonals of a matrix

    get_diag generalises the function "np.diag" to also work with dictionaries and dictionary of dictionaries.
    Parameters
    ----------
    X : np.ndarray, list, dict
        The input from which to extract the diagonal. This can be a numpy array, a list (of lists), a dictionary, or a dictionary of dictionaries.
    K : int, optional
        The diagonal in question. The default, 0, will return the main diagonal. A positive value returns an upper diagonal, and a negative value returns a lower diagonal.

    Returns
    -------
    np.ndarray or dictionary
        If `X` is a numpy array, returns a numpy array containing the diagonal elements.
        If `X` is a list, list of lists, dictionary, or dictionary of dictionaries, returns a dictionary containing the diagonal elements.

    Raises
    ------
    ValueError
        If `X` is not a supported data type.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4]])
    >>> get_diag(X)
    array([1, 4])

    >>> X = {0: {0: 1, 1: 2}, 1: {0: 3, 1: 4}}
    >>> get_diag(X)
    [1, 4]

    >>> X = [[1, 2], [3, 4]]
    >>> get_diag(X)
    [1, 4]
    """

    # use built-in np.diag for most data types
    if isinstance(X, np.ndarray):
        return np.diag(X, k=K)
    # or use the following for dictionaries and dictionary of dictionaries
    else:
        if isinstance(X, list) and isinstance(X[0], np.ndarray):
            X = {i: X[i] for i in range(len(X))}

        if isinstance(X, dict) and isinstance(X[0], np.ndarray):
            m = len(X)
            n = 1
            max_dim = max(m, n) + abs(K)
            D = generate_2d_dict(max_dim, max_dim)
            for i in range(max_dim - K):
                D[i][i + K] = X[i]
            return D
        if isinstance(X, list) and isinstance(X[0], list):
            X = {i: {j: X[i][j] for j in range(len(X[i]))} for i in range(len(X))}

        if isinstance(X, dict) and isinstance(X[0], dict):
            m = len(X)
            n = len(X[0])
            min_dim = min(m, n) - abs(K)
            D = {}
            if K >= 0:
                for i in range(min_dim):
                    D[i] = X[i][i + K]
            else:
                for i in range(min_dim):
                    D[i] = X[i - K][i]
            return D
        else:
            return None


def convert_sparse_to_dense(data):
    if isinstance(data, dict):
        for key in data:
            data[key] = convert_sparse_to_dense(data[key])
    elif isinstance(data, list):
        for i, item in enumerate(data):
            data[i] = convert_sparse_to_dense(item)
    elif isinstance(data, np.ndarray) and data.dtype == object:
        # Convert each sparse matrix in the numpy array to a dense matrix and return as a list
        if np.ndim(data) == 0:
            return data
        else:
            return [convert_sparse_to_dense(item) for item in data]
    elif issparse(data):
        data = data.toarray()
    return data


def get_phi(x):
    return 1 / (1 + np.exp(-x))


def generate_dcm_fmri_mode(Ev, modes, Cv=None):
    """
    Generate adjacency matrix for spectral DCM from Lyapunov exponents
    :param Ev: Lyapunov exponents or eigenvalues of effective connectivity
    :param modes: modes or eigenvectors
    :param Cv: optional (posterior) covariance matrix
    :return: Ep - Jacobian or (symmetric) effective connectivity matrix,
            Cp - posterior covariance matrix of Jacobian elements
    """
    # outer product
    Ep = modes @ np.diag(-np.exp(-Ev)) @ modes.T

    if Cv is None:
        return Ep

    # covariance
    # dAdv = get_diff(generate_dcm_fmri_mode, Ev, modes, 1)
    # G = np.column_stack([dAdv[i].flatten() for i in range(len(dAdv))])
    # Cp = G @ Cv @ G.T

    # return Ep, Cp


def define_fx_fmri(x, u, P, M=None, first_derivative=False):
    """
    State equation for a dynamic [bilinear/nonlinear/Balloon] model of fMRI responses.

    Parameters
    ----------
    x : np.ndarray
        State vector with the following components:
        x[:, 0] - excitatory neuronal activity (ue)
        x[:, 1] - vascular signal (s)
        x[:, 2] - rCBF (ln(f))
        x[:, 3] - venous volume (ln(v))
        x[:, 4] - deoxyHb (ln(q))
        Optionally: x[:, 5] - inhibitory neuronal activity (ui)
    u : np.ndarray
        Input vector.
    P : dict
        Parameters of the model, containing:
        - A: linear parameters
        - B: bilinear parameters
        - C: exogenous parameters
        - D: nonlinear parameters
    M : dict
        Additional options, can contain:
        - symmetry: indicates if symmetry is considered in the model.

    Returns
    -------
    f : np.ndarray
        The derivative of the state vector, dx/dt.
    dfdx : np.ndarray
        The derivative of f with respect to x, df/dx (placeholder, not implemented).
    D : np.ndarray
        Delays (placeholder, not implemented).
    dfdu : np.ndarray
        The derivative of f with respect to u, df/du (placeholder, not implemented).
    """
    # Check for M and symmetry
    symmetry = M.get("symmetry", 0) if M is not None else 0

    # Convert parameters to dense if they are in sparse format in MATLAB
    # In Python, assuming all matrices are already dense
    A = np.array(P["A"])
    B = np.array(P["B"])
    C = np.array(P["C"]) / 16
    D = np.array(P["D"])

    # Implement differential state equation y = dx/dt (neuronal)
    # Placeholder for actual implementation
    f = x.copy()  # This is a simplification, actual dynamics need to be implemented
    x = copy.deepcopy(x)
    # Check for five hidden states per region
    if x.shape[1] == 5:
        if (A.size == A.shape[0] or A.shape[1] == A.size) and A.size > 1:
            # Excitatory connections
            EE = generate_dcm_fmri_mode(P.A, M.modes)

            # Input dependent modulation
            for i in range(P.B.shape[2]):
                EE += u[i] * P.B[:, :, i]

            # Nonlinear (state) terms
            for i in range(P.D.shape[2]):
                EE += x[i, 0] * P.D[:, :, i]
        else:
            # Input dependent modulation
            for i in range(B.shape[2]):
                A += u[i] * B[:, :, i]

            # Nonlinear (state) terms
            if D.ndim == 3:
                for i in range(D.shape[2]):
                    A += x[i, 0] * D[:, :, i]

            # Combine forward and backward connections if necessary
            if A.ndim == 3 and A.shape[2] > 1:
                A = np.exp(A[:, :, 0]) - np.exp(A[:, :, 1])

            # One neuronal state per region
            SE = np.diag(A)
            EE = A - np.diag(np.exp(SE) / 2 + SE)

            # Symmetry constraints
            if symmetry:
                EE = (EE + EE.T) / 2

        # Flow
        f[:, [0]] = EE @ x[:, [0]] + C @ u

    else:
        # Otherwise two neuronal states per region
        for i in range(B.shape[2]):
            A += u[i] * B[:, :, i]

        if D.ndim == 3:
            for i in range(D.shape[2]):
                A += x[i, 0] * D[:, :, i]

        n = A.shape[0]
        EE = np.exp(A) / 8
        IE = np.diag(np.diag(EE))
        EE -= IE
        EI = np.eye(n)
        SE = np.eye(n) / 2
        SI = np.eye(n)

        if A.ndim == 3 and A.shape[2] > 1:
            phi = get_phi(A[:, :, 1] * 2)
            EI += EE * (1 - phi)
            EE = EE * phi - SE
        else:
            EE -= SE

        f[:, [0]] = EE @ x[:, [0]] - IE @ x[:, [5]] + P.C @ u
        f[:, [5]] = EI @ x[:, [0]] - SI @ x[:, [5]]

    # Hemodynamic motion
    H = [0.64, 0.32, 2.00, 0.32, 0.4]
    x[:, 2:5] = np.exp(x[:, 2:5])
    sd = H[0] * np.exp(P["decay"])
    tt = H[2] * np.exp(P["transit"])
    fv = x[:, [3]] ** (1 / H[3])
    ff = (1 - (1 - H[4]) ** (1 / x[:, [2]])) / H[4]

    f[:, [1]] = x[:, [0]] - sd * x[:, [1]] - H[1] * (x[:, [2]] - 1)
    f[:, [2]] = x[:, [1]] / x[:, [2]]
    f[:, [3]] = (x[:, [2]] - fv) / (tt * x[:, [3]])
    f[:, [4]] = (ff * x[:, [2]] - fv * x[:, [4]] / x[:, [3]]) / (tt * x[:, [4]])
    f = f.flatten(order="F")

    if first_derivative:
        # Neuronal Jacobian
        n, m = x.shape
        if m == 5:
            # One neuronal state per region
            dfdx = generate_2d_dict(m, m)
            dfdx[0][0] = EE
            if D.ndim == 3:
                for i in range(D.shape[2]):
                    dd = D[:, :, i] + np.diag((np.diag(EE) - 1) * np.diag(D[:, :, i]))
                    dfdx[0][0][:, [i]] = dfdx[0][0][:, [i]] + dd @ x[:, [0]]
        else:
            # Two neuronal states per region (NB nonlinear (D) effects not implemented)
            dfdx = generate_2d_dict(m + 1, m + 1)
            dfdx[0][0] = EE
            dfdx[0][5] = -IE
            dfdx[5][0] = EI
            dfdx[5][5] = -SI

        # Input Jacobian
        dfdu = generate_2d_dict(2, 1)
        dfdu[0][0] = C
        for i in range(B.shape[2]):
            bb = B[:, :, i] + np.diag((np.diag(EE) - 1) * np.diag(B[:, :, i]))
            dfdu[0][0][:, [i]] = dfdu[0][0][:, [i]] + bb @ x[:, [0]]
        dfdu[1][0] = np.zeros((n * (m - 1), len(u)))
        # Hemodynamic Jacobian
        dfdx[1][0] = np.eye(n)
        dfdx[1][1] = np.eye(n) * (-sd)
        dfdx[1][2] = np.diag(-H[1] * x[:, 2])
        dfdx[2][1] = np.diag(1.0 / x[:, 2])
        dfdx[2][2] = np.diag(-x[:, 1] / x[:, 2])
        dfdx[3][2] = np.diag(x[:, 2] / (tt[:, 0] * x[:, 3]))
        dfdx[3][3] = np.diag(
            -x[:, 3] ** (1 / H[3] - 1) / (tt[:, 0] * H[3])
            - (1.0 / x[:, 3] * (x[:, 2] - x[:, 3] ** (1 / H[3]))) / tt[:, 0]
        )
        dfdx[4][2] = np.diag(
            (
                x[:, 2]
                + np.log(1 - H[4]) * (1 - H[4]) ** (1.0 / x[:, 2])
                - x[:, 2] * (1 - H[4]) ** (1.0 / x[:, 2])
            )
            / (tt[:, 0] * x[:, 4] * H[4])
        )
        dfdx[4][3] = np.diag(
            (x[:, 3] ** (1 / H[3] - 1) * (H[3] - 1)) / (tt[:, 0] * H[3])
        )
        dfdx[4][4] = np.diag(
            (x[:, 2] / x[:, 4])
            * ((1 - H[4]) ** (1.0 / x[:, 2]) - 1)
            / (tt[:, 0] * H[4])
        )

        # Concatenate Jacobians
        dfdx = concatenate_dod_lol(dfdx)
        dfdu = concatenate_dod_lol(dfdu)
        dd = 1

        return f, dfdx, dd, dfdu

    return f


def define_gx_fmri(x, u, P, first_derivative=False):
    """
    Simulated BOLD response to input
    Parameters:
    x          - state vector     (see spm_fx_fmri)
    epsilon          - Parameter vector (see spm_fx_fmri)

    Returns:
    g          - BOLD response (%)
    dgdx       - Derivative of BOLD response with respect to x
    """
    # Biophysical constants for 1.5T
    TE = 0.04  # time to echo (TE) (default 0.04 sec)
    V0 = 4  # resting venous volume (%)
    # estimated region-specific ratios of intra- to extra-vascular signal
    if isinstance(P["epsilon"], np.ndarray):
        if P["epsilon"].ndim == 1:
            ep = np.exp(P["epsilon"])[0]
        elif P["epsilon"].ndim == 2:
            ep = np.exp(P["epsilon"])[0, 0]
    elif isinstance(P["epsilon"], (int, float)):
        ep = np.exp(P["epsilon"])
    r0 = 25  # slope r0 of intravascular relaxation rate R_iv as a function of oxygen saturation S
    nu0 = 40.3  # frequency offset at the outer surface of magnetized vessels (Hz)
    E0 = 0.4  # resting oxygen extraction fraction

    # Coefficients in BOLD signal model
    k1 = 4.3 * nu0 * E0 * TE
    k2 = ep * r0 * E0 * TE
    k3 = 1 - ep

    # Output equation of BOLD signal model
    v = np.exp(x[:, 3])
    q = np.exp(x[:, 4])
    g = V0 * (k1 - k1 * q + k2 - k2 * q / v + k3 - k3 * v)
    if first_derivative:
        # Derivative dgdx
        n, m = x.shape
        dgdx = np.zeros((n, n * m))
        dgdx[:, 3 * n : 4 * n] = np.diag(-V0 * (k3 * v - k2 * q / v))
        dgdx[:, 4 * n : 5 * n] = np.diag(-V0 * (k1 * q + k2 * q / v))

        return g, dgdx
    return g


def integrate_bilinear(P, M, U):
    """
    Integrate a MIMO bilinear system dx/dt = f(x,u) = A*x + B*x*u + Cu + D;
    Parameters:
    P   - model parameters
    M   - model structure
    U   - input structure or matrix

    Returns:
    y   - response y = g(x,u,P)
    """
    if not isinstance(U, dict):
        U = {"u": U}
    dt = U.setdefault("dt", 1)

    u = U["u"].shape[0]
    v = M.get("ns", u)

    x = np.hstack([1, vectorise_object(M["x"])])

    M["f"] = M.get("f", lambda x, u, P, M: np.zeros((0, 1)))
    M["g"] = M.get("g", lambda x, u, P, M: x)

    M0, M1, *_ = get_bireduce(M, P)
    m = len(M1)

    try:
        D = np.maximum(np.round(np.array(M["delays"]) / U["dt"]).astype(int), 0)
    except KeyError:
        D = np.ones(M["l"], dtype=int) * np.round(u / v)

    i = (
        list([0])
        + (np.nonzero(np.any(np.diff(U["u"], axis=0), axis=1))[0] + 1).tolist()
    )
    su = np.zeros(u, dtype=bool)
    su[i] = True

    s = np.ceil(np.arange(v) * u / v).astype(int)
    sy = np.zeros((M["l"], u))
    for j in range(M["l"]):
        sy[j, s + D[j] - 1] = np.arange(1, v + 1)

    t = np.nonzero(su | np.any(sy, axis=0))[0]
    su = su[t]
    sy = sy[:, t]
    dt = np.append(np.diff(t), 0) * U["dt"]

    y = np.zeros((M["l"], v))
    J = copy.deepcopy(M0)
    uu = U["u"]
    for i in range(len(t)):
        if su[i]:
            u = uu[t[i], :]
            J = copy.deepcopy(M0)
            for j in range(m):
                J += u[j] * M1[j]

        if np.any(sy[:, i]):
            q = unvectorise_object(x[1:], M["x"])
            q = vectorise_object(M["g"](q, u, P))
            j = np.nonzero(sy[:, i])[0]
            s = int(sy[j[0], i]) - 1
            y[j, s] = q[j]

        x = expm(J * dt[i]).dot(x)

        if np.linalg.norm(x, 1) > 1e6:
            break

    return y.T


def get_bireduce(M, P):
    """
    Reduction of a fully nonlinear MIMO system to Bilinear form
    """

    # set up the f functions, in python directly assign the function name.
    M = copy.deepcopy(M)
    try:
        funx = M["f"]
    except KeyError:
        M["f"] = lambda x, u, P, M: np.zeros((0, 1))
        M["n"] = 0
        M["x"] = np.zeros((0, 0))
        funx = M["f"]

    # expansion point
    x = vectorise_object(M["x"])
    u = vectorise_object(M.get("u", np.zeros((M["m"], 1)))).reshape((3, 1))

    # Partial derivatives for 1st order Bilinear operators
    if all(key in M for key in ("dfdxu", "dfdx", "dfdu", "f0")):
        dfdxu = M["dfdxu"]
        dfdx = M["dfdx"]
        dfdu = M["dfdu"]
        f0 = M["f0"]
    else:
        dfdxu, dfdx, *_ = get_diff(funx, M["x"], u, P, M, np.array([1, 2]))
        dfdu, f0 = get_diff(funx, M["x"], u, P, M, 2)
    f0 = vectorise_object(f0)
    m = len(dfdxu)  # m inputs
    n = len(f0)  # n states

    # delay operator
    if "D" in M:
        f0 = M["D"] @ f0
        dfdx = M["D"] @ dfdx
        dfdu = M["D"] @ dfdu
        for i in range(m):
            dfdxu[i] = M["D"] @ dfdxu[i]

    # Bilinear operators
    M0 = concatenate_dod_lol(
        [[np.array([[0]]), []], [(f0.reshape((n, 1)) - dfdx @ x.reshape((n, 1))), dfdx]]
    )
    M1 = [
        concatenate_dod_lol(
            [
                [np.array([[0]]), []],
                [(dfdu[:, [i]] - dfdxu[i] @ x.reshape((n, 1))), dfdxu[i]],
            ]
        )
        for i in range(m)
    ]
    if "g" not in M:
        M["g"] = lambda x, u, P, M: vectorise_object(x)
        M["l"] = n
    fung = M["g"]

    dgdx, g0 = get_diff(fung, M["x"], u, P, 1)
    g0 = vectorise_object(g0)
    l = len(g0)

    L1 = concatenate_dod_lol([[(g0.reshape((l, 1)) - dgdx @ x.reshape((n, 1))), dgdx]])

    if "l" not in M:
        return M0, M1, L1

    dgdxx, *_ = get_diff(fung, M["x"], u, P, np.array([1, 1]), "nocat")
    L2 = {}
    for i in range(l):
        D = np.zeros((n, n))
        for j in range(n):
            D[j, :] = dgdxx[j][i, :]
        L2[i] = concatenate_dod_lol(get_diag([np.array([[0]]), D]))
    return M0, M1, L1, L2


def get_normalisation(X, p=None):
    """
    Euclidean normalization.

    Perform a Euclidean normalization setting the column-wise sum of
    squares to unity (leaving columns of zeros as zeros).

    Parameters
    ----------
    X : ndarray
        Matrix to be normalized.
    p : int, optional
        Degree of polynomial used for detrending. If None (default), no detrending is performed. Currently, only 0 and 1 are supported

    Returns
    -------
    X : ndarray
        The normalized matrix.

    Notes
    -----
    This function performs a Euclidean normalization on the input matrix `X`,
    setting the column-wise sum of squares to unity. Columns of zeros remain unchanged.

    Examples
    --------
    >>> X = np.array([[1, 2], [3, 4], [5, 6]])
    >>> get_normalisation(X)
    array([[0.16903085, 0.18257419],
            [0.50709255, 0.54772256],
            [0.84515425, 0.91287093]])

    """
    # Detrend if p is not None
    if p is not None:
        X = detrend(X, type="constant" if p == 0 else "linear", axis=0)

    # Euclidean normalization
    for i in range(X.shape[1]):
        col_norm = np.linalg.norm(X[:, i])
        if col_norm > 0:
            X[:, i] = X[:, i] / col_norm

    return X


def get_inv(A, TOL=None):
    """
    Compute the inverse for ill-conditioned matrices.

    This function computes the inverse of an ill-conditioned matrix by adding
    a small value to its diagonal and then computing the inverse. This can help
    to stabilize the inversion process.

    Parameters
    ----------
    A : ndarray
        The matrix to be inverted.
    TOL : float, optional
        Tolerance for adjusting the diagonal of `A`. If not specified, it is
        automatically determined based on `A`.

    Returns
    -------
    X : ndarray
        The inverse of the matrix `A`.

    Notes
    -----
    The tolerance (TOL) is used to modify the diagonal of the matrix before
    inversion. If not specified, it is calculated as the maximum of
    `eps(norm(A, inf)) * max(m, n)` and `exp(-32)`, where `m` and `n` are the
    dimensions of `A`.

    Examples
    --------
    >>> A = np.array([[1, 2], [3, 4]])
    >>> get_inv(A)
    array([[-2. ,  1. ],
            [ 1.5, -0.5]])

    """
    if np.isscalar(A):
        A = np.eye(1, 1) * A
    # Check if A is empty
    if A.size == 0:
        return np.empty((A.shape[1], A.shape[0]))

    m, n = A.shape

    # Calculate tolerance if not provided
    if TOL is None:
        TOL = max(
            np.finfo(A.dtype).eps * np.linalg.norm(A, np.inf) * max(m, n), np.exp(-32)
        )

    # Compute the inverse
    X = np.linalg.inv(A + np.eye(m, n) * TOL)

    if X.size == 1:
        X = X.item()

    return X


def swap_dod_lol(x):
    """
    Swap columns for dictionaries in matrix arrays stored in a dictionary.

    This function swaps the columns for dictionaries in matrix arrays, where the
    matrix arrays are stored in a dictionary. The keys of the dictionary
    correspond to the cell indices in the MATLAB version.

    Parameters
    ----------
    x : 2D dict or 2D list
        A dictionary of dictionaries and each value of innermost dictionary is a matrix.

    Returns
    -------
    y : dict
        A dictionary with the swapped matrices.

    Notes
    -----
    This function is designed to mimic the behavior of the MATLAB function
    `spm_cell_swap`, converting it to work with Python dictionaries and numpy
    arrays. The input dictionary `x` should be a 2D dictionary.

    Examples
    --------
    >>> x = {}
        x[0] = {}
        x[0][0] = np.array([[1, 2], [3, 4]])
        x[0][1] = np.array([[5, 6], [7, 8]])
        x[1] = {}
        x[1][0] = np.array([[9, 10], [11, 12]])
        x[1][1] = np.array([[13, 14], [15, 16]])
        xx = swap_dod_lol(x)

    """
    if not x:
        return {}

    input_is_list = isinstance(x, list)
    # Convert list of lists to dictionary format if necessary
    if input_is_list:
        x_dict = {
            i: {j: np.array(x[i][j]) for j in range(len(x[i]))} for i in range(len(x))
        }
    else:
        x_dict = x
    x = copy.deepcopy(x_dict)
    # Determine the size of the arrays and the dictionary structure
    k = len(x)
    l = len(x[0])
    m, n = x[0][0].shape

    # Initialize the output dictionary
    y = generate_2d_dict(k, n)
    for i in range(k):
        for j in range(n):
            y[i][j] = np.zeros((m, l))

    # Perform the swap
    for r in range(k):
        for j in range(l):
            for i in range(n):
                y[r][i][:, j] = x[r][j][:, i]

    return y


def get_Ncdf(x, loc=0, scale=1):
    """
    Compute the cumulative distribution function (CDF) for the normal distribution.

    This function computes the CDF for the normal distribution with a given mean
    and standard deviation.

    Parameters
    ----------
    x : float or ndarray
        The value(s) at which to compute the CDF.
    loc : float or ndarray, optional
        The mean of the normal distribution. Default is 0.
    scale : float or ndarray, optional
        The standard deviation of the normal distribution. Default is 1.

    Returns
    -------
    y : float or ndarray
        The CDF value(s) for the normal distribution.

    Notes
    -----
    This function is designed to mimic the behavior of the MATLAB function 'spm_Ncdf', note that 'v' stands for the variance in 'spm_Ncdf'. This function use 'scale', i.e., the standard deviation.

    Examples
    --------
    >>> get_Ncdf(0)
    0.5
    >>> get_Ncdf([0, 1, 2])
    array([0.5       , 0.84134475, 0.97724987])
    """
    # Ensure loc and scale are numpy arrays for uniform handling
    loc = np.array(loc, ndmin=1)
    scale = np.array(scale, ndmin=1)

    # Broadcast loc or scale if they are scalars or 1-element arrays to match the other's size
    if loc.size == 1:
        loc = np.full(scale.shape, loc.item())
    if scale.size == 1:
        scale = np.full(loc.shape, scale.item())

    # Calculate the CDF for each element in x with corresponding loc and scale
    return norm.cdf(x, loc=loc, scale=scale)


def get_invNcdf(q, loc=0, scale=1):
    """
    Calculate the inverse of the normal cumulative distribution function (CDF).

    Parameters
    ----------
    q : array_like
        Quantiles, with the last axis of `q` being the quantiles. `q` must be in the range [0, 1].
    loc : float or array_like, optional
        Mean (“centre”) of the distribution. Default is 0.
    scale : float or array_like, optional
        Standard deviation (spread or “width”) of the distribution. Default is 1.

    Returns
    -------
    ndarray or scalar
        Inverse of the CDF for each element in `q` for the given mean and standard deviation.
        Scalar if `q` is a scalar, otherwise an ndarray with the same shape as `q`.

    Notes
    -----
    This function computes the quantile values for the normal distribution, effectively
    serving as the inverse function for the CDF of the normal distribution. It is useful
    for determining the value at a given percentile.

    Examples
    --------
    >>> get_invNcdf(0.95)
    1.6448536269514722
    >>> get_invNcdf([0.025, 0.975])
    array([-1.95996398,  1.95996398])

    """
    # Ensure loc and scale are numpy arrays for uniform handling
    loc = np.array(loc, ndmin=1)
    scale = np.array(scale, ndmin=1)

    # Broadcast loc or scale if they are scalars or 1-element arrays to match the other's size
    if loc.size == 1:
        loc = np.full(scale.shape, loc.item())
    if scale.size == 1:
        scale = np.full(loc.shape, scale.item())

    # Calculate the CDF for each element in x with corresponding loc and scale
    return norm.ppf(q, loc=loc, scale=scale)


def get_trace(A, B):
    """
    Fast trace for large matrices: C = get_trace(A, B) = trace(A*B)

    Parameters
    ----------
    A : array_like
        First input matrix.
    B : array_like
        Second input matrix, which is multiplied by A.

    Returns
    -------
    C : float
        The trace of the product of A and B.
    """
    C = np.sum(np.multiply(A.T, B))
    return C


def get_pinv(A, tol=None):
    """
    Pseudo-inverse for matrices, with an option for sparse matrices.

    Parameters
    ----------
    A : ndarray
        Matrix for which to compute the pseudo-inverse.
    tol : float, optional
        Tolerance to force singular value decomposition. If not provided,
        it is computed as max(m, n) * eps(max(S)), where S are the singular values.

    Returns
    -------
    X : ndarray
        Generalized inverse of A.
    """
    m, n = A.shape
    if A.size == 0:
        return np.zeros((n, m))

    if tol is None:
        # Suppress warnings and attempt inversion
        try:
            X = get_inv(A.T @ A)
            if np.all(np.isfinite(X)):
                return X @ A.T
        except np.linalg.LinAlgError:
            pass  # Proceed to SVD if inversion fails or results are not finite

    # Compute SVD
    U, S, V = truncate_svd(A, 0)

    if tol <= 0:
        tol = max(m, n) * np.finfo(S.dtype).eps * max(S)

    # Apply tolerance
    r = sum(np.abs(S) > tol)
    if r == 0:
        return np.zeros((n, m))

    # Compute pseudo-inverse using truncated SVD
    S_inv = np.diag(1 / S[:r])
    X = V[:, :r] @ S_inv @ U[:, :r].T
    return X


def get_logdet(C):
    """
    Compute the log of the determinant of a positive (semi-)definite matrix C.

    Parameters
    ----------
    C : ndarray
        A positive (semi-)definite matrix.

    Returns
    -------
    H : float
        The log of the determinant of C. For non-positive definite cases,
        the determinant is considered to be the product of the positive
        singular values.

    Notes
    -----
    This function is a computationally efficient operator that can deal with
    dense matrices. For non-positive definite cases, the determinant is
    considered to be the product of the positive singular values.
    """
    # Remove null variances
    if np.isscalar(C):
        return np.log(C) if C > 0 else np.nan
    i = np.diag(C) != 0
    C = C[i][:, i]
    i, j = np.nonzero(C)
    s = C[i, j]
    if np.any(np.isnan(C)):
        return np.nan

    TOL = 1e-16
    if np.any(i != j):
        # Check if the matrix is asymmetric
        if not np.allclose(C, C.T, atol=TOL):
            # Asymmetric matrix
            s = svd(C, compute_uv=False)
        else:
            # Symmetric matrix
            try:
                R = cholesky(C)
                H = 2 * np.sum(np.log(np.diag(R)))
                return H
            except LinAlgError:
                # Fallback to SVD for non-positive definite cases
                s = svd(C, compute_uv=False)

    # Singular values in s
    valid_s = s[(s > TOL) & (s < 1 / TOL)]
    H = np.sum(np.log(valid_s))
    return H


def get_Q(a, n, q=0):
    """
    Return an (n x n) (inverse) autocorrelation matrix for an AR(p) process.

    Parameters
    ----------
    a : array_like
        Vector of (p) AR coefficients.
    n : int
        Size of Q.
    q : int, optional
        Switch to return inverse autocorrelation (q=1) or precision (default q=0).

    Returns
    -------
    Q : ndarray
        (Inverse) autocorrelation or precision matrix.

    Notes
    -----
    Uses a Yule-Walker approach to compute the matrix such that if y is an AR(p)
    process generated from an i.i.d innovation z, then cov(y) = K*K' where y = K*z.
    If q != 0, a first order process is assumed when evaluating the precision
    (inverse covariance) matrix; i.e., a = a[0].
    """
    if q:
        # Compute P (precision)
        A = np.array([-a[0], (1 + a[0] ** 2), -a[0]])
        Q = np.zeros((n, n))
        diags = np.arange(-1, 2)
        for d in diags:
            Q += np.diag(np.ones(n - abs(d)) * A[d + 1], k=d)
    else:
        # Compute Q (covariance)
        p = min(len(a), n - 1)
        if p == 0:
            A = np.array([1.0])
        else:
            A = np.concatenate(([1], -a[:p]))
        col = np.zeros(n)
        col[: p + 1] = A
        row = np.zeros(n)
        row[0] = 1
        P = toeplitz(col, row)
        K = inv(P)
        K = K * (np.abs(K) > 1e-4)
        Q = K @ K.T
        Q = toeplitz(Q[:, 0])

    return Q


def get_dctmtx(N, K=None, n=None, f=None):
    """
    Create basis functions for Discrete Cosine Transform.

    Parameters
    ----------
    N : int or list of int
        Dimension(s).
    K : int or list of int, optional
        Order(s).
    n : int or list of int
        Points to sample.
    f : str, optional
        'diff' or 'diff2'.

    Returns
    -------
    C or D : ndarray or dict of ndarray
        DCT matrix or its derivative, or a dictionary of these matrices.

    Notes
    -----
    This function creates a matrix for the first few basis functions of a one
    dimensional discrete cosine transform. With the 'diff' argument, it produces
    the derivatives of the DCT.
    """
    if isinstance(N, list) and len(N) > 1:
        if K is None:
            K = N
        elif isinstance(K, list) and len(K) < len(N):
            K = [K[0]] * len(N)

        if n is None:
            C = 1
            for i in range(len(N)):
                c = get_dctmtx(N[i], K[i], n=None, f=f)
                C = np.kron(c, C)
            return C

        if f is None:
            f = [1] * len(N)
        D = generate_2d_dict(len(N), 1)
        for d in range(len(N)):
            C = 1
            for i in range(len(N)):
                if i == d:
                    c = get_dctmtx(N[i], K[i], n=n, f=f)
                else:
                    c = get_dctmtx(N[i], K[i], n=n)
                C = np.kron(c, C)
            D[d][0] = C
        return concatenate_dod_lol(D)

    if K is None:
        K = N
    if n is None:
        n = np.arange(N)
    else:
        n = np.asarray(n).flatten()

    d = 0
    if isinstance(f, str) and f == "diff":
        d = 1
    elif isinstance(f, str) and f == "diff2":
        d = 2

    C = np.zeros((len(n), K))
    if d == 0:
        C[:, 0] = 1 / sqrt(N)
        for k in range(1, K):
            C[:, k] = sqrt(2 / N) * cos(np.pi * (2 * n + 1) * (k) / (2 * N))
    elif d == 1:
        for k in range(1, K):
            C[:, k] = (
                -sqrt(2 / N) * sin(np.pi * (2 * n + 1) * k / N / 2) * np.pi * k / N
            )
    elif d == 2:
        for k in range(1, K):
            C[:, k] = (
                -sqrt(2 / N)
                * cos(np.pi * (2 * n + 1) * k / N / 2)
                * (np.pi * k / N) ** 2
            )

    return C


def get_Ce(t, v, a=None):
    """
    Error covariance constraints (for serially correlated data).

    Parameters
    ----------
    t : str or list
        If a string, specifies the type ('ar' or 'fast'). If a list, it is treated as the 'v' parameter.
    v : list or None
        A list where v[i] = number of observations for i-th block. Used if 't' is a list.
    a : int or None, optional
        AR coefficient expansion point. Default is None, which implies block diagonal identity matrices.

    Returns
    -------
    C : list of numpy.ndarray
        List of covariance matrices.

    Notes
    -----
    This function is a Python adaptation of the MATLAB `spm_Ce` function, tailored for use with NumPy and custom utility functions.
    """
    # Handle input parameters
    if isinstance(v, (int, float)):
        v = [int(v)]

    # Initialize the list to hold covariance matrices
    C = []

    if t.lower() == "ar":
        n = sum(v)
        k = 0
        if len(v) > 1:
            for vi in v:
                dCda = get_Ce(t="ar", v=vi, a=a)
                for dj in dCda:
                    x, y = np.nonzero(dj)
                    q = dj[x, y]
                    tempMatrix = np.zeros((n, n))
                    tempMatrix[x + k, y + k] = q
                    C.append(tempMatrix)
                k += vi
        else:
            if a is not None:
                Q = get_Q(np.array([a]), v[0])
                dQda, *_ = get_diff(get_Q, np.array([a]), v[0], 1)
                C.append(Q - dQda[0] * a)
                C.append(Q + dQda[0] * a)
            else:
                C.append(np.eye(v[0], v[0]))

    elif t.lower() == "fast":
        dt = a
        n = sum(v)
        k = 0
        for vm in v:
            T = np.arange(vm) * dt
            d = 2 ** np.arange(np.floor(np.log2(dt / 4)), np.log2(64))
            for i in range(min(6, len(d))):
                for j in range(3):
                    QQ = toeplitz((T**j) * np.exp(-T / d[i]))
                    tempMatrix = np.zeros((n, n))
                    x, y = np.nonzero(QQ)
                    q = QQ[x, y]
                    tempMatrix[x + k, y + k] = q
                    C.append(tempMatrix)
            k += vm

    else:
        raise ValueError("Unknown error covariance constraints.")

    return C


def get_kernels(*args, nargout=2):
    """
    Return global Volterra kernels for a MIMO Bilinear system.

    Parameters
    ----------
    *args : variable length argument list
        Can be one of the following formats:
        - (M0, M1, N, dt) for output kernels
        - (M0, M1, L1, N, dt) for state kernels
        - (M0, M1, L1, L2, N, dt) for output kernels (1st and 2nd order)
    nargout : int, optional

    Returns
    -------
    K0 : ndarray
        0th order kernel, shape (1, l)
    K1 : ndarray
        1st order kernel, shape (N, l, m)
    K2 : ndarray
        2nd order kernel, shape (N, N, l, m, m)
    H1 : ndarray
        Helper matrix for computing kernels, shape (N, n, m)

    Notes
    -----
    This function returns Volterra kernels for bilinear systems of the form:

        dq/dt = f(q,u) = M0*q + M1{1}*q*u1 + ... M1{m}*q*um
        y(i) = L1(i,:)*q + q'*L2{i}*q

    where q = [1 x(t)] are the states augmented with a constant term.
    """
    # Assign inputs based on the number of arguments
    if len(args) == 4:
        M0, M1, N, dt = args
        L1, L2 = None, None
    elif len(args) == 5:
        M0, M1, L1, N, dt = args
        L2 = None
    elif len(args) == 6:
        M0, M1, L1, L2, N, dt = args

    # Bilinear reduction if necessary
    if isinstance(M0, (dict, list)):
        M0, M1, L1, L2 = get_bireduce(M0, M1)

    # Initialize parameters
    N = int(N)
    n = M0.shape[0]
    m = len(M1)
    l = L1.shape[0] if L1 is not None else n - 1
    H1 = np.zeros((N, n, m))
    K1 = np.zeros((N, l, m))
    K2 = np.zeros((N, N, l, m, m))
    M0 = M0.astype(float)

    # Pre-compute matrix exponentials
    e1 = expm(dt * M0)
    e2 = expm(-dt * M0)
    M = generate_2d_dict(N, m)
    for p in range(m):
        M[0][p] = e1 @ M1[p] @ e2
    ei = copy.deepcopy(e1)
    for i in range(1, N):
        ei = e1 @ ei
        for p in range(m):
            M[i][p] = e1 @ M[i - 1][p] @ e2

    # 0th order kernel
    X0 = np.zeros((n, 1))
    X0[0, 0] = 1
    if nargout > 0:
        H0 = ei @ X0
        K0 = L1 @ H0 if L1 is not None else H0

    # 1st order kernel
    if nargout > 1:
        for p in range(m):
            for i in range(N):
                H1[i, :, p] = M[i][p] @ H0[:, 0]
                K1[i, :, p] = L1 @ H1[i, :, p] if L1 is not None else H1[i, :, p]

    # 2nd order kernels
    if nargout > 2:
        for p in range(m):
            for q in range(m):
                for j in range(N):
                    H = (
                        L1 @ M[j][q] @ H1[j:N, :, p].T
                        if L1 is not None
                        else M[j][q] @ H1[j:N, :, p].T
                    )
                    K2[j, j:N, :, q, p] = H.T
                    K2[j:N, j, :, p, q] = H.T

        if L2 is not None:
            # Add output nonlinearity
            for i in range(m):
                for j in range(m):
                    for p in range(l):
                        K2[:, :, p, i, j] += H1[:, :, i] @ L2[p] @ H1[:, :, j].T

    return K0, K1, K2, H1


def get_dx(dfdx, f, t=np.inf, Q=None):
    """
    Returns dx(t) = (expm(dfdx*t) - I)*inv(dfdx)*f using local linearisation.

    Parameters
    ----------
    dfdx : ndarray
        The derivative of f with respect to x, i.e., df/dx.
    f : ndarray or dict
        The derivative of x with respect to t, i.e., dx/dt. Can be a vector or a structured object.
    t : float or ndarray or cell, optional
        The integration time. If t is a cell (i.e., {t}), then t is set to exp(t - log(diag(-dfdx))).
        The default is np.inf, which implies heavy regularization.
    Q : ndarray, optional
        Solenoidal flow, used for augmenting the flow and Jacobian. The default is None, which disables solenoidal mixing.

    Returns
    -------
    dx : ndarray
        The change in x, i.e., x(t) - x(0).

    Notes
    -----
    This function integrates a dynamic system using local linearisation, accommodating nonlinearities in the state equation.
    It uses an augmented system and the Pade approximation for computing matrix exponentials.
    """
    # Defaults
    nmax = 512  # Threshold for numerical approximation
    xf = copy.deepcopy(f)
    f = vectorise_object(f)  # Vectorise
    n = len(f)  # Dimensionality
    dfdx = copy.deepcopy(dfdx)

    # Handle t as a regulariser
    if isinstance(t, (list, tuple, np.ndarray)):  # Check if t is a cell
        t = t[0]
        if np.isscalar(t):
            t = np.exp(t - get_logdet(dfdx) / n)
        else:
            t = np.exp(t - np.log(np.diag(-dfdx)))
    # Solenoidal mixing
    if Q is not None:
        L = np.tril(dfdx)
        Q = L - L.T
        Q = Q / np.linalg.norm(Q, 2) / 8
        f = f - Q @ f
        dfdx = dfdx - Q @ dfdx

    # Use a [pseudo]inverse if all t > TOL
    if np.min(t) > np.exp(16):
        dx = -get_pinv(dfdx) @ f
    else:
        # Ensure t is a scalar or matrix
        if not np.isscalar(t):
            t = np.diag(t)

        # Augment Jacobian and take matrix exponential
        J = concatenate_dod_lol(
            [[np.zeros((1, 1)), np.zeros((1, n))], [t * f.reshape((n, 1)), t * dfdx]]
        )

        # Solve using matrix expectation
        if n <= nmax:
            dx = expm(J)[:, 0]
        else:
            x = np.zeros(n + 1)
            x[0] = 1
            # dx = expm_multiply(J, x)[:,0]
            dx = expm(J).dot(x)

    # Recover update and ensure it's real
    dx = np.real(dx[1:])
    return unvectorise_object(dx, xf)


def get_nlsi_GN(M, U, Y):
    """
    Bayesian inversion of nonlinear models - Gauss-Newton/Variational Laplace.

    Parameters
    ----------
    M : dict
        Dynamic MIMO models configuration, including functions for the generative model (IS),
        feature selection (FS), starting estimates (P), prior expectations (pE, hE) and
        prior covariances (pC, hC) of model parameters.
    U : dict
        Inputs configuration, including inputs (u) and sampling interval (dt).
    Y : dict
        Outputs configuration, including outputs (y), sampling interval for outputs (dt),
        confounds or null space (X0), and error precision components (Q).

    Returns
    -------
    Ep : ndarray
        Conditional expectation E{P|y} of model parameters.
    Cp : ndarray
        Conditional covariance Cov{P|y} of model parameters.
    Eh : ndarray
        Conditional log-precisions E{h|y}.
    F : float
        Log evidence, i.e., free energy F = log evidence = p(y|f,g,pE,pC) = p(y|m).
    L : ndarray
        Additional outputs as specified.
    dFdp : ndarray
        Derivative of free energy with respect to parameters.
    dFdpp : ndarray
        Second derivative (Hessian) of free energy with respect to parameters.

    Notes
    -----
    Returns the moments of the posterior p.d.f. of the parameters of a
    nonlinear model specified by IS(P,M,U) under Gaussian assumptions.
    Usually, IS is an integrator of a dynamic MIMO input-state-output model:

        dx/dt = f(x,u,P)
        y     = g(x,u,P)  + X0*P0 + e

    A static nonlinear observation model with fixed input or causes u
    obtains when x = []. i.e.,

        y     = g([],u,P) + X0*P0e + e

    but static nonlinear models are specified more simply using

        y     = IS(P,M,U) + X0*P0 + e

    Priors on the free parameters P are specified in terms of expectation pE
    and covariance pC. The E-Step uses a Fisher-Scoring scheme and a Laplace
    approximation to estimate the conditional expectation and covariance of P.
    If the free-energy starts to increase, an abbreviated descent is
    invoked. The M-Step estimates the precision components of e, in terms
    of log-precisions. Although these two steps can be thought of in
    terms of E and M steps they are in fact variational steps of a full
    variational Laplace scheme that accommodates conditional uncertainty
    over both parameters and log precisions (c.f. hyperparameters with hyper
    priors).

    An optional feature selection can be specified with parameters M.FS.

    For generic aspects of the scheme see:

    Friston K, Mattout J, Trujillo-Barreto N, Ashburner J, Penny W.
    Variational free energy and the Laplace approximation.
    NeuroImage. 2007 Jan 1;34(1):220-34.

    This scheme handles complex data along the lines originally described in:

    Sehpard RJ, Lordan BP, and Grant EH.
    Least squares analysis of complex data with applications to permittivity
    measurements.
    J. Phys. D. Appl. Phys 1970 3:1759-1764.
    """

    #
    # Setting default values with Pythonic approach
    M.setdefault("nograph", 0)
    M.setdefault("noprint", 0)
    M.setdefault("Nmax", 128)

    # check integrator or generation scheme
    try:
        _ = M["IS"]
    except KeyError:
        try:
            M["IS"] = M["G"]
        except KeyError:
            M["IS"] = "integrate_bilinear"
    # Check feature selection
    try:
        _ = M["FS"]
    except KeyError:
        M["FS"] = lambda x: x

    # Composition of feature selection and prediction (usually an integrator)
    try:
        y = Y["y"]
    except (TypeError, KeyError):
        y = Y

    if callable(M["IS"]) and callable(M["FS"]):
        # Composition of feature selection and generation functions
        IS = lambda P, M, U: M["FS"](M["IS"](P, M, U))
        y = M["FS"](y)
    else:
        try:
            # Try FS(y, M)
            try:
                y = M["FS"](y, M)
                IS = lambda P, M, U: M["FS"](M["IS"](P, M, U), M)
            except Exception:
                # Try FS(y)
                y = M["FS"](y)
                IS = lambda P, M, U: M["FS"](M["IS"](P, M, U))
        except Exception:
            # Otherwise, FS(y) = y
            try:
                IS = lambda P, M, U: M["IS"](P, M, U)
            except Exception:
                IS = M["IS"]
    #  already python handle and callable, no need to check
    # Check and update M dictionary with checked functions
    # if "f" in M:
    #     M["f"] = spm_funcheck(M["f"])
    # if "g" in M:
    #     M["g"] = spm_funcheck(M["g"])
    # if "h" in M:
    #     M["h"] = spm_funcheck(M["h"])
    #
    if isinstance(y, (dict, list)):
        ns = y[0].shape[0]
    else:
        ns = y.shape[0]
    ny = len(vectorise_object(y))  # Total number of response variables
    nr = ny // ns  # Number of response components
    M["ns"] = ns
    #
    # initial states
    if "x" not in M:
        M.setdefault("n", 0)
        M["x"] = np.zeros((M["n"], 1))

    # input
    U = U if "U" in locals() else {}
    #
    # initial parameters
    try:
        _ = vectorise_object(M["P"]) - vectorise_object(M["pE"])
        print("\nParameter initialisation successful\n")
    except Exception:
        M["P"] = copy.deepcopy(M["pE"])

    #  time-step for plot
    # dt = Y["dt"] if "dt" in Y else 1

    #  precision components Q
    try:
        Q = Y["Q"]
        if isinstance(Q, (int, float)):
            Q = [Q]
    except KeyError:
        Q = get_Ce(t="ar", v=nr * [ns])
    nh = len(Q)  # number of precision components
    nq = ny // Q[0].shape[0]  # for compact Kronecker form of M-step
    #  prior moments (assume uninformative priors if not specified)
    pE = copy.deepcopy(M["pE"])
    try:
        pC = M["pC"]
    except AttributeError:
        num_p = len(vectorise_object(M["pE"]))
        pC = np.eye(num_p) * np.exp(16)

    #  confounds (if specified)
    try:
        nb = Y["X0"].shape[0]  # number of bins
        nx = ny // nb  # number of blocks
        dfdu = np.kron(np.eye(nx), Y["X0"])
    except Exception:
        dfdu = np.zeros((ny, 0))
    if dfdu.size == 0:
        dfdu = np.zeros((ny, 0))

    #  hyperpriors - expectation (and initialize hyperparameters)
    try:
        hE = np.array(M["hE"]).flatten()
        if len(hE) != nh:
            hE = np.pad(hE, (0, nh - len(hE)), "constant")
    except Exception:
        hE = np.zeros(nh) - np.log(np.var(vectorise_object(y), ddof=1)) + 4
    hE = hE.astype(np.float64)
    h = copy.deepcopy(hE)

    #  hyperpriors - covariance
    try:
        ihC = get_inv(M["hC"])
        if len(ihC) != nh:
            ihC = ihC * np.eye(nh)
    except Exception:
        ihC = np.eye(nh) * np.exp(4)

    #  unpack covariance
    if isinstance(pC, dict):
        pC = np.diag(vectorise_object(pC))

    #  dimension reduction of parameter space
    V, *_ = truncate_svd(pC, 0)
    nu = dfdu.shape[1]  # number of parameters (confounds)
    num_p = V.shape[1]  # number of parameters (effective)
    ip = np.arange(num_p)
    iu = np.arange(nu) + num_p

    #  second-order moments (in reduced space)
    pC = V.T @ pC @ V
    uC = np.eye(nu) * 1e8
    ipC = np.linalg.inv(concatenate_dod_lol(get_diag([pC, uC])))

    #  initialize conditional density

    Eu = get_pinv(dfdu) @ vectorise_object(y)
    p = np.concatenate(
        (V.T @ (vectorise_object(M["P"]) - vectorise_object(M["pE"])), Eu)
    )
    Ep = unvectorise_object(vectorise_object(pE) + V @ p[ip], pE)

    #  EM
    criterion = [0, 0, 0, 0]
    C = {}
    C["F"] = -np.inf  # free energy
    v = -4  # log ascent rate
    dFdh = np.zeros(nh)
    dFdhh = np.zeros((nh, nh))

    for k in range(M["Nmax"]):

        # time
        tStart = time.time()

        # E-Step: prediction f, and gradients; dfdp
        try:
            # gradients
            dfdp, f = get_diff(IS, Ep, M, U, np.array([1]), [V])
            dfdp = vectorise_object(dfdp).reshape(ny, num_p, order="F")

            # check for stability
            normdfdp = np.linalg.norm(dfdp, np.inf)
            revert = np.isnan(normdfdp) or normdfdp > 1e32
        except Exception:
            revert = True

        if revert and k > 0:
            for i in range(4):
                # reset expansion point and increase regularization
                v = min(v - 2, -4)

                # E-Step: update
                p = C["p"] + get_dx(dFdpp, dFdp, [v])
                Ep = unvectorise_object(vectorise_object(pE) + V @ p[ip], pE)

                # try again
                try:
                    dfdp, f = get_diff(IS, Ep, M, U, np.array([1]), [V])
                    dfdp = vectorise_object(dfdp).reshape(ny, np, order="F")

                    # check for stability
                    normdfdp = np.linalg.norm(dfdp, np.inf)
                    revert = np.isnan(normdfdp) or normdfdp > np.exp(32)
                except Exception:
                    revert = True

                if not revert:
                    break

        if revert:
            raise ValueError("DCM: get_nlsi_GN, Convergence failure.")

        # prediction error and full gradients
        e = vectorise_object(y) - vectorise_object(f) - dfdu @ p[iu]
        J = -np.concatenate((dfdp, dfdu), axis=1)

        # M-step: Fisher scoring scheme to find h = max{F(p,h)}
        for _ in range(8):
            # precision and conditional covariance
            iS = 0
            for i in range(nh):
                iS += Q[i] * (np.exp(-32) + np.exp(h[i]))

            if nh > 1:
                S = get_inv(iS)

            iS = np.kron(np.eye(nq), iS)
            Pp = np.real(J.T @ iS @ J)
            Cp = get_inv(Pp + ipC)

            if nh > 1:
                # precision operators for M-Step
                P = [None] * nh
                PS = [None] * nh
                JPJ = [None] * nh
                for i in range(nh):
                    P[i] = Q[i] * np.exp(h[i])
                    PS[i] = P[i] @ S
                    P[i] = np.kron(np.eye(nq), P[i])
                    JPJ[i] = np.real(J.T @ P[i] @ J)

                # derivatives: dLdh = dL/dh,...
                for i in range(nh):
                    dFdh[i] = (
                        np.trace(PS[i]) * nq / 2
                        - np.real(e.T @ P[i] @ e) / 2
                        - get_trace(Cp, JPJ[i]) / 2
                    )
                    for j in range(i, nh):
                        dFdhh[i, j] = -get_trace(PS[i], PS[j]) * nq / 2
                        dFdhh[j, i] = dFdhh[i, j]
            else:
                # Simplifications for nh == 1
                dFdh[0] = ny / 2 - np.real(e.T @ iS @ e) / 2 - get_trace(Cp, Pp) / 2
                dFdhh[0, 0] = -ny / 2
            # Add second order terms; noting diS/dh(i)h(i) = diS/dh(i) = P{i}
            dFdhh += np.diag(dFdh)

            # Add hyperpriors
            d = h - hE
            dFdh -= ihC @ d
            dFdhh -= ihC
            Ch = get_pinv(-dFdhh.real)

            # Update ReML estimate
            dh = get_dx(dFdhh, dFdh, [4])
            dh = np.clip(dh, -1, 1)
            h += dh

            # convergence
            dF = dFdh.T @ dh
            if dF < 1e-2:
                break

        # E-Step with Levenberg-Marquardt regularization
        L = [
            get_logdet(iS) * nq / 2
            - np.real(e.T @ iS @ e) / 2
            - ny * np.log(8 * np.arctan(1)) / 2,
            get_logdet(ipC @ Cp) / 2 - p.T @ ipC @ p / 2,
            get_logdet(ihC @ Ch) / 2 - d.T @ ihC @ d / 2,
        ]
        F = sum(L)

        if "F0" not in locals():
            F0 = F
        else:
            if not M["noprint"]:
                print(f" actual: {F - C['F']:.3e} ({time.time() - tStart:.2f} sec)")

        if F > C["F"] or k < 2:
            # accept current estimates
            C["p"] = p
            C["h"] = h
            C["F"] = F
            C["L"] = L
            C["Cp"] = Cp

            # E-Step: Conditional update of gradients and curvature
            dFdp = -np.real(J.T @ iS @ e) - ipC @ p
            dFdpp = -np.real(J.T @ iS @ J) - ipC

            # decrease regularization
            v = min(v + 1 / 2, 4)
            str_EM = "EM:(+)"
        else:
            # reset expansion point
            p = C["p"]
            h = C["h"]
            Cp = C["Cp"]

            # and increase regularization
            v = min(v - 2, -4)
            str_EM = "EM:(-)"

        # E-Step: update
        dp = get_dx(dFdpp, dFdp, [v], "Q")
        p += dp
        Ep = unvectorise_object(vectorise_object(pE) + V @ p[ip], pE)

        # convergence
        dF = dFdp.T @ dp
        if not M["noprint"]:
            print(f"{str_EM}: {k} F: {F - F0:.3e} dF predicted: {dF:.3e}")
        criterion = [(dF < 1e-1)] + criterion[:-1]
        if all(criterion):
            if not M["noprint"]:
                print(" convergence")
            break
    #  outputs

    Ep = unvectorise_object(vectorise_object(pE) + V @ C["p"][ip], pE)
    Cp = V @ C["Cp"][np.ix_(ip, ip)] @ V.T
    Eh = C["h"]
    F = C["F"]
    L = C["L"]

    return Ep, Cp, Eh, F, L, dFdp, dFdpp


def get_dcm_evidence(DCM):
    """
    Compute evidence of DCM model in Python equivalent to MATLAB's spm_dcm_evidence function.

    Parameters:
    DCM (dict): DCM data structure

    Returns:
    dict: evidence with fields region_cost, bic_penalty, bic_overall, aic_penalty, aic_overall
    """
    v = DCM["v"]  # number of samples
    n = DCM["n"]  # number of regions
    wsel = np.nonzero(np.diag(DCM["Cp"]))[0]  # Indices of non-zero posterior covariance

    evidence = {
        "region_cost": np.zeros(n),
        "bic_penalty": 0,
        "bic_overall": 0,
        "aic_penalty": 0,
        "aic_overall": 0,
    }

    for i in range(n):
        try:
            lambda_i = DCM["Ce"][i * v, i * v]  # Ce is error covariance
        except IndexError:
            try:
                lambda_i = DCM["Ce"][i]  # Ce is a hyperparameter
            except IndexError:
                lambda_i = DCM["Ce"]  # Ce is the hyperparameter

        evidence["region_cost"][i] = -0.5 * v * np.log(lambda_i) - 0.5 * np.dot(
            DCM["R"][:, i].T, (1 / lambda_i) * np.eye(v).dot(DCM["R"][:, i])
        )

    evidence["aic_penalty"] = len(wsel)
    evidence["bic_penalty"] = 0.5 * len(wsel) * np.log(v)
    evidence["aic_overall"] = np.sum(evidence["region_cost"]) - evidence["aic_penalty"]
    evidence["bic_overall"] = np.sum(evidence["region_cost"]) - evidence["bic_penalty"]

    return evidence


def get_DEM_M_set(M):
    """
    Processes the input dictionary M to set missing fields and checks for
    the specification of hidden states or their number. It also checks for
    supra-ordinate level and adds one with flat priors if necessary. Finally,
    it sets default fields for static models (hidden states).

    Parameters
    ----------
    M : dict or list
        A dictionary representing the model structure with potential fields:
        - 'f': function for hidden states dynamics,
        - 'n': number of hidden states,
        - 'x': initial hidden states,
        - 'g': function for observation model,
        - 'l': levels,
        - 'm': modes.

    Returns
    -------
    None
        Modifies the input dictionary M in-place.
    """
    M = copy.deepcopy(M)
    g = len(M)

    # Check for specification of hidden states
    if "f" in M and "n" not in M and "x" not in M:
        print("Please specify hidden states or their number")

    # Check supra-ordinate level and add one (with flat priors) if necessary
    if "g" in M[g - 1] and callable(M[g - 1]["g"]):
        g += 1
        M[g] = {"l": M[g - 1]["m"]}
    M[g - 1]["m"] = 0
    M[g - 1]["n"] = 0
    # Default fields for static models (hidden states)
    for m in M:
        if "f" not in m:
            m["f"] = lambda x, v, P: np.zeros((0, 1))
            m["x"] = np.zeros((0, 1))
            m["n"] = 0
    for i in range(g):
        if "f" not in M[i] or not callable(M[i]["f"]):
            M[i]["f"] = lambda x, v, P: np.zeros((0, 1))
            M[i]["x"] = np.zeros((0, 1))
            M[i]["n"] = 0
    # Check for prior expectation of parameters M.pE
    for i in range(g):
        if "pE" not in M[i]:
            # Assume fixed parameters
            M[i]["pE"] = np.zeros((0, 0))
    # Check for priors covariances - pC
    try:
        _ = M[0]["pC"]  # Attempt to access pC to check if it exists
    except KeyError:
        # Assume fixed (zero variance) parameters
        for i in range(g):
            p = len(vectorise_object(M[i]["pE"]))
            M[i]["pC"] = np.zeros((p, p))  # Use dense matrix instead of sparse
    for i in range(g):
        # number of parameters
        nump = len(vectorise_object(M[i]["pE"]))

        # Assume fixed parameters if not specified
        if (
            "pC" not in M[i]
            or M[i]["pC"] is None
            or M[i]["pC"].size == 0
            or len(M[i]["pC"]) == 0
        ):
            M[i]["pC"] = np.zeros((nump, nump))

        # convert variances to covariances if necessary
        if isinstance(M[i]["pC"], (np.ndarray, list)) and np.ndim(M[i]["pC"]) == 1:
            M[i]["pC"] = np.diag(M[i]["pC"])

        # convert variance to covariances if necessary
        if np.isscalar(M[i]["pC"]):
            M[i]["pC"] = np.eye(nump) * M[i]["pC"]

        # check size
        if M[i]["pC"].shape[0] != nump:
            raise ValueError(f"please check: M[{i}].pC")

    # get inputs
    # --------------------------------------------------------------------------
    try:
        v = M[g - 1]["v"]
    except KeyError:
        v = np.array([])

    if v.size == 0:
        try:
            v = np.zeros((M[g - 2]["m"], 1))
        except KeyError:
            pass

    if v.size == 0:
        try:
            v = np.zeros((M[g - 1]["l"], 1))
        except KeyError:
            pass

    M[g - 1]["l"] = len(vectorise_object(v))
    M[g - 1]["v"] = v

    for i in range(g - 2, -1, -1):  # Adjusted for Python's 0-based indexing
        # print(i)
        # Initialize x if not present
        x = M[i].get("x", np.zeros((M[i].get("n", 0), 1)))

        # Ensure x is not empty if n is specified
        if x.size == 0 and M[i].get("n", 0):
            x = np.zeros((M[i]["n"], 1))

        # Check and call f(x,v,P)
        if "f" in M[i] and callable(M[i]["f"]):
            try:
                f = M[i]["f"](x, v, M[i].get("pE", np.array([])), None)
                if len(vectorise_object(x)) != len(vectorise_object(f)):
                    raise ValueError(f"please check: M[{i}].f(x,v,P)")
            except Exception as e:
                print(f"??? evaluation failure: M[{i}].f(x,v,P)")
                raise e

        # Check and call g(x,v,P)
        if "g" in M[i] and callable(M[i]["g"]):
            try:
                M[i]["m"] = len(vectorise_object(v))
                v = M[i]["g"](x, v, M[i].get("pE", np.array([])))
                M[i]["l"] = len(vectorise_object(v))
                M[i]["n"] = len(vectorise_object(x))
                M[i]["v"] = v
                M[i]["x"] = x
            except Exception as e:
                print(f"??? evaluation failure: M[{i}].g(x,v,P)")
                raise e
    # Ensure xP and vP are initialized in M[0] if not present

    for i in range(g):
        M[i].setdefault("xP", [])
        M[i].setdefault("vP", [])
        # Handling hidden states xP

        if np.size(M[i]["xP"]) > 1:
            M[i]["xP"] = np.diag(M[i]["xP"])
        if np.size(M[i]["xP"]) != M[i]["n"]:
            try:
                M[i]["xP"] = np.eye(M[i]["n"]) * M[i]["xP"]
            except Exception:
                M[i]["xP"] = np.zeros((M[i]["n"], M[i]["n"]))

        # Handling hidden states vP
        if np.size(M[i]["vP"]) > 1:
            M[i]["vP"] = np.diag(M[i]["vP"])
        if np.size(M[i]["vP"]) != M[i]["l"]:
            try:
                M[i]["vP"] = np.eye(M[i]["l"]) * M[i]["vP"]
            except Exception:
                M[i]["vP"] = np.zeros((M[i]["l"], M[i]["l"]))

    # Calculate the total number of hidden states across all models
    nx = sum(m["n"] for m in M)

    # Hyperparameters and components initialization

    pP = 1  # Prior precision on log-precisions

    for i in range(g):
        M[i].setdefault("Q", [])
        M[i].setdefault("R", [])
        M[i].setdefault("V", [])
        M[i].setdefault("W", [])
        M[i].setdefault("hE", [])
        M[i].setdefault("gE", [])
        M[i].setdefault("ph", [])
        M[i].setdefault("pg", [])
        # Ensure components are lists
        if not isinstance(M[i].get("Q", None), (list, np.ndarray)):
            M[i]["Q"] = [M[i]["Q"]]
        if not isinstance(M[i].get("R", None), (list, np.ndarray)):
            M[i]["R"] = [M[i]["R"]]

        # Vectorize hE and gE
        M[i]["hE"] = vectorise_object(M[i].get("hE", []))
        M[i]["gE"] = vectorise_object(M[i].get("gE", []))
        # M[i]["hC"] = M[i]["hC"]

        # Initialize or validate hC and gC
        # Check and set hC
        try:
            np.dot(M[i]["hC"], M[i]["hE"])
        except Exception:
            M[i]["hC"] = np.eye(len(M[i]["hE"])) / pP

        # Check and set gC
        try:
            np.dot(M[i]["gC"], M[i]["gE"])
        except Exception:
            M[i]["gC"] = np.eye(len(M[i]["gE"])) / pP

        # Ensure hC and gC are not empty
        if "hC" not in M[i] or M[i]["hC"].size == 0:
            M[i]["hC"] = np.eye(len(M[i]["hE"])) / pP
        if "gC" not in M[i] or M[i]["gC"].size == 0:
            M[i]["gC"] = np.eye(len(M[i]["gE"])) / pP

        # Adjust Q, R, hE, gE, hC, and gC lengths
        # Check components and assume i.i.d if not specified
        if len(M[i]["Q"]) > len(M[i]["hE"]):
            M[i]["hE"] = np.zeros(len(M[i]["Q"])) + M[i]["hE"][0]
        if len(M[i]["Q"]) < len(M[i]["hE"]):
            M[i]["Q"] = [np.eye(M[i]["l"])]
            M[i]["hE"] = M[i]["hE"][0]
        if len(M[i]["hE"]) > len(M[i]["hC"]):
            M[i]["hC"] = np.eye(len(M[i]["Q"])) * M[i]["hC"][0]
        if len(M[i]["R"]) > len(M[i]["gE"]):
            M[i]["gE"] = np.zeros(len(M[i]["R"])) + M[i]["gE"][0]
        if len(M[i]["R"]) < len(M[i]["gE"]):
            M[i]["R"] = [np.eye(M[i]["n"])]
            M[i]["gE"] = M[i]["gE"][0]
        if len(M[i]["gE"]) > len(M[i]["gC"]):
            M[i]["gC"] = np.eye(len(M[i]["R"])) * M[i]["gC"][0]

        # Check consistency and sizes (Q)
        for j in range(len(M[i]["Q"])):
            if M[i]["Q"][j].shape[0] != M[i]["l"]:
                raise ValueError(f"wrong size; M[{i}].Q[{j}]")

        # Check consistency and sizes (R)
        for j in range(len(M[i]["R"])):
            if M[i]["R"][j].shape[0] != M[i]["n"]:
                raise ValueError(f"wrong size; M[{i}].R[{j}]")

        # Check and adjust V and W

        def adjust_matrix(matrix, expected_length):
            # Adjust a matrix to have the expected length, assuming square matrices
            if isinstance(matrix, list) and len(matrix) == expected_length:
                return np.diag(matrix)
            elif isinstance(matrix, (float, int)):
                return np.eye(expected_length) * matrix
            elif (
                not isinstance(matrix, np.ndarray) or matrix.shape[0] != expected_length
            ):
                try:
                    return np.eye(expected_length) * matrix[0]
                except IndexError:
                    return np.zeros((expected_length, expected_length))
            return matrix

        M[i]["V"] = adjust_matrix(M[i].get("V", []), M[i]["l"])
        M[i]["W"] = adjust_matrix(M[i].get("W", []), M[i]["n"])

    # Temporal smoothness - s.d. of kernel
    if "E" not in M[0] or "s" not in M[0]["E"]:
        M[0].setdefault("E", {})["s"] = 1 / 2 if nx else 0

    # Time step
    M[0]["E"].setdefault("dt", 1)

    # Embedding orders
    if "E" not in M[0] or "d" not in M[0]["E"]:
        M[0]["E"]["d"] = 2 if nx else 0
    if "E" not in M[0] or "n" not in M[0]["E"]:
        M[0]["E"]["n"] = 6 if nx else 0

    M[0]["E"]["d"] = min(M[0]["E"]["d"], M[0]["E"]["n"])

    # Number of iterations
    M[0]["E"].setdefault("nD", 1 if nx else 8)
    M[0]["E"].setdefault("nE", 8)
    M[0]["E"].setdefault("nM", 8)
    M[0]["E"].setdefault("nN", 8)

    # Checks on smoothness hyperparameter
    for key in ["sv", "sw"]:
        M = [{k: v for k, v in d.items() if k != key} for d in M]

    for i in range(g):
        for key in ["sv", "sw"]:
            if key not in M[i] or not isinstance(M[i][key], (int, float)):
                M[i][key] = M[0]["E"]["s"]

    # Check on linear approximation scheme
    M[0]["E"].setdefault("linear", 0)

    # Checks on estimability
    # Assuming 'norm' function is defined elsewhere, as norm calculation is not shown in the selected code
    Q = np.linalg.norm(M[-1]["V"], 1) == 0
    for i in range(g - 1):
        P = np.linalg.norm(M[i].get("pC", 0), 1) > np.exp(8)
        if P and Q:
            print("Please use informative priors on causes or parameters")

    return M


def get_DEM_set(DEM):
    """
    Perform checks on DEM structures and convert sparse matrices to dense matrices.

    Parameters:
    DEM (dict): Dictionary containing DEM structures including M, Y, U, X, and possibly G and C.

    Returns:
    dict: Updated DEM dictionary.
    """

    # check recognition model
    DEM = copy.deepcopy(DEM)
    DEM["M"] = get_DEM_M_set(DEM["M"])

    # check format of inputs and data
    try:
        DEM["Y"]
    except KeyError:
        DEM["Y"] = np.array(DEM["Y"]["y"]).T
    try:
        DEM["U"]
    except KeyError:
        DEM["U"] = np.array(DEM["U"]["u"]).T

    # check whether data are specified explicitly or with a generative model
    try:
        N = DEM["Y"].shape[1]
    except KeyError:
        try:
            DEM["G"] = get_DEM_M_set(DEM["G"])
            N = DEM["C"].shape[1]
        except KeyError as exc:
            raise ValueError("Please specify data or inputs") from exc

    DEM.setdefault("class", "unknown")

    # ensure model and data dimensions check
    try:
        if DEM["Y"].shape[0] != DEM["M"][0]["l"]:
            raise ValueError("DCM and data are incompatible")
    except KeyError as exc:
        if DEM["C"].shape[0] != DEM["M"][-1]["l"]:
            raise ValueError("DCM and causes are incompatible") from exc

    # Default priors and confounds
    n = DEM["M"][-1]["l"]
    if "U" not in DEM:
        DEM["U"] = np.zeros((n, N))
    if "X" not in DEM:
        DEM["X"] = np.zeros((0, N))

    # transpose causes and confounds, if specified in conventional fashion
    if DEM["U"].shape[1] < N:
        DEM["U"] = DEM["U"].T
    if DEM["X"].shape[1] < N:
        DEM["X"] = DEM["X"].T

    # check prior expectation of causes (at level n) and confounds
    if not np.any(DEM["U"]):
        DEM["U"] = np.zeros((n, N))
    if not np.any(DEM["X"]):
        DEM["X"] = np.zeros((0, N))

    # ensure inputs and cause dimensions check
    if DEM["U"].shape[0] != DEM["M"][-1]["l"]:
        raise ValueError("DCM inputs and priors are not compatible")

    # ensure causes and data dimensions check
    if DEM["U"].shape[1] < N:
        raise ValueError("Priors and data have different lengths")

    # ensure confounds and data dimensions check
    if DEM["X"].shape[1] < N:
        raise ValueError("Confounds and data have different lengths")

    # check length of time-series
    if N < DEM["M"][0]["E"]["n"]:
        raise ValueError("Please ensure time-series is longer than embedding order")

    return DEM


def get_LAP_ph(h, M):
    """
    Default precision function for LAP models (causal states).

    Parameters
    ----------
    h : ndarray
        Precision parameters.
    M : dict
        Model structure containing fields l, V, and Q.

    Returns
    -------
    p : ndarray
        Log-precision of the model.

    Notes
    -----
    This function computes the log-precision of the model based on fixed
    components from M.V and free components scaled by precision parameters h
    across the components defined in M.Q.
    """

    # fixed components
    p = np.zeros((M["l"],))
    try:
        V = np.diag(M["V"])
        if np.all(V):
            p = np.log(V)
    except KeyError:
        pass  # If M.V is not defined, skip this part

    # free components
    for i in range(len(M["Q"])):
        p += h[i] * np.diag(M["Q"][i])

    return p


def get_LAP_pg(h, M):
    """
    Default precision function for LAP models (hidden states).

    Parameters
    ----------
    h : ndarray
        Precision parameters.
    M : dict
        Model structure containing fields n, W, and R.

    Returns
    -------
    p : ndarray
        Log-precision of the model.

    Notes
    -----
    This function computes the log-precision of the model based on fixed
    components from M.W and free components scaled by precision parameters h
    across the components defined in M.R.
    """

    # fixed components
    p = np.zeros((M["n"],))
    try:
        W = np.diag(M["W"])
        if np.all(W):
            p = np.log(W)
    except KeyError:
        pass  # If M.W is not defined, skip this part

    # free components
    for i in range(len(M["R"])):
        p += h[i] * np.diag(M["R"][i])

    return p


def get_DEM_R(n, s, form="Gaussian"):
    """
    Precision of the temporal derivatives of a Gaussian process.

    Parameters
    ----------
    n : int
        Truncation order.
    s : float
        Temporal smoothness - standard deviation of kernel (bins).
    form : str, optional
        'Gaussian' or '1/f'. The default is 'Gaussian'.

    Returns
    -------
    R : ndarray
        (n x n) E*V*E: precision of n derivatives.
    V : ndarray
        (n x n) V: covariance of n derivatives.

    Notes
    -----
    This function computes the precision of the temporal derivatives of a
    Gaussian process, assuming a known form for the temporal correlations.
    """

    if n == 0:
        return np.array([]), np.array([])
    if s is None or s == 0:
        s = np.exp(-8)
    r = np.zeros((2 * n - 1,))
    # Temporal correlations (assuming known form) - V
    if form == "Gaussian":
        k = np.arange(0, n)
        x = np.sqrt(2) * s
        r[2 * k] = np.cumprod(1 - 2 * k) / (x ** (2 * k))
    elif form == "1/f":
        k = np.arange(0, n)
        x = 8 * s**2
        r[2 * k] = ((-1) ** k) * gamma(2 * k + 1) / (x ** (2 * k))
    else:
        raise ValueError("Unknown autocorrelation form")

    # Create covariance matrix in generalised coordinates
    V = []
    for i in range(1, n + 1):
        V.append(r[np.arange(n) + i - 1])
        r = -r
    V = np.array(V)

    # And precision - R
    R = inv(V)

    return R, V


def get_LAP_eval(M, qu, qh, nargout=1):
    N = len(M)
    v = [None] * N
    x = [None] * N
    v[0 : N - 1] = unvectorise_object(qu["v"][0], [M[i + 1]["v"] for i in range(N - 1)])
    x[0 : N - 1] = unvectorise_object(qu["x"][0], [M[i]["x"] for i in range(N - 1)])

    h = generate_2d_dict(N, 1)
    g = generate_2d_dict(N, 1)
    for i in range(N):
        try:
            h[i][0] = vectorise_object(M[i]["ph"](qh["h"][i], M[i])).reshape(-1, 1)
        except Exception:
            h[i][0] = np.zeros((M[i]["l"], 1))
        try:
            g[i][0] = vectorise_object(M[i]["pg"](qh["g"][i], M[i])).reshape(-1, 1)
        except Exception:
            g[i][0] = np.zeros((M[i]["n"], 1))

    p = {"h": concatenate_dod_lol(h), "g": concatenate_dod_lol(g)}

    if nargout == 1:
        return p

    dp = {
        "h": {"dx": None, "dv": None, "dh": None},
        "g": {"dx": None, "dv": None, "dg": None},
    }
    nx = len(vectorise_object(x))
    nv = len(vectorise_object(v))
    hn = len(vectorise_object(qh["h"]))
    gn = len(vectorise_object(qh["g"]))
    nh = p["h"].shape[0]
    ng = p["g"].shape[0]

    dp["h"]["dh"] = np.zeros((nh, hn))
    dp["g"]["dg"] = np.zeros((ng, gn))
    dp["h"]["dx"] = np.zeros((nh, nx))
    dp["h"]["dv"] = np.zeros((nh, nv))
    dp["g"]["dx"] = np.zeros((ng, nx))
    dp["g"]["dv"] = np.zeros((ng, nv))

    method = {
        "h": M[0].get("E", {}).get("method", {}).get("h", 1),
        "g": M[0].get("E", {}).get("method", {}).get("g", 1),
        "x": M[0].get("E", {}).get("method", {}).get("x", 1),
        "v": M[0].get("E", {}).get("method", {}).get("v", 1),
    }

    if method["h"] or method["g"]:
        dhdh = generate_2d_dict(N, N)
        dgdg = generate_2d_dict(N, N)
        for i in range(N):
            dhdh[i][i], _ = get_diff(M[i]["ph"], qh["h"][i], M[i], 1)
            dgdg[i][i], _ = get_diff(M[i]["pg"], qh["g"][i], M[i], 1)
        dp["h"]["dh"] = concatenate_dod_lol(dhdh)
        dp["g"]["dg"] = concatenate_dod_lol(dgdg)

    if method["v"]:
        dhdv = generate_2d_dict(N, N)
        dgdv = generate_2d_dict(N, N)
        for i in range(N):
            dhdv[i][i] = np.zeros(np.size(v[i]))
            dgdv[i][i] = np.zeros(np.size(v[i]))
        dp["h"]["dv"] = concatenate_dod_lol(dhdv)
        dp["g"]["dv"] = concatenate_dod_lol(dgdv)

    if method["x"]:
        dhdx = generate_2d_dict(N, N)
        dgdx = generate_2d_dict(N, N)
        for i in range(N):
            dhdx[i][i] = np.zeros(np.size(x[i]))
            dgdx[i][i] = np.zeros(np.size(x[i]))
        dp["h"]["dx"] = concatenate_dod_lol(dhdx)
        dp["g"]["dx"] = concatenate_dod_lol(dgdx)

    return p, dp


def get_DEM_embed(Y, n, t, dt=1, d=None):
    """
    Temporal embedding into derivatives.

    Parameters
    ----------
    Y : ndarray
        (v x N) matrix of v time-series of length N.
    n : int
        Order of temporal embedding.
    t : float
        Time {bins} at which to evaluate derivatives (starting at t = 1).
    dt : float, optional
        Sampling interval {secs} [default = 1].
    d : ndarray or None, optional
        Delay (bins) for each row of Y. If None, d is set to 0 for all rows.

    Returns
    -------
    y : list of ndarray
        {n,1}(v x 1) temporal derivatives y[:] <- E*Y(t).

    """
    if d is None:
        d = np.zeros(1)

    # get dimensions
    q, N = Y.shape
    y = [np.zeros((q, 1)) for _ in range(n)]

    # return if ~q
    if not q:
        return y

    # loop over channels
    for p, delay in enumerate(d):

        # boundary conditions
        s = (t - delay) / dt
        k = np.arange(1, n + 1) + np.fix(s - (n + 1) / 2)
        x = s - min(k) + 1
        k[k < 1] = 1
        k[k > N] = N

        # Inverse embedding operator (T): cf, Taylor expansion Y(t) <- T*y[:]
        T = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                T[i, j] = ((i + 1 - x) * dt) ** j / np.prod(np.arange(1, j + 1))

        # embedding operator: y[:] <- E*Y(t)
        E = np.linalg.inv(T)

        # embed
        if len(d) == q:
            for i in range(n):
                y[i][p, :] = np.dot(Y[p, k.astype(int) - 1], E[i, :].T)
        else:
            for i in range(n):
                y[i] = np.dot(Y[:, k.astype(int) - 1], E[i, :].T)
    return y


def get_DEM_eval_diff(x, v, qp, M, bilinear=True):
    """
    Evaluate derivatives for DEM schemes.

    Parameters
    ----------
    x : array_like
        Hidden states.
    v : dict of array_like
        Causal states.
    qp : dict
        Conditional density of parameters.
        qp['p'] : dict of array_like
            Parameter deviates for i-th level.
        qp['u'] : array_like
            Basis set.
        qp['x'] : array_like
            Expansion point (= prior expectation).
    M : dict
        Model structure.
    bilinear : bool, optional
        Optional flag to suppress second-order derivatives, by default False.

    Returns
    -------
    D : dict
        Derivatives.
        D['dgdv'] : array_like
    """
    # Check for evaluation of bilinear terms
    if bilinear is None:
        bilinear = True

    # Get dimensions
    nl = len(M)  # number of levels
    ne = sum(vectorise_object([M[i]["l"] for i in range(nl)])).astype(
        int
    )  # number of e (errors)
    nx = sum(
        vectorise_object([M[i]["n"] for i in range(nl)]).astype(int)
    )  # number of x (hidden states)
    np_ = sum(
        vectorise_object([M[i]["p"] for i in range(nl)]).astype(int)
    )  # number of p (parameters)
    ny = M[0]["l"]  # number of y (inputs)
    nc = M[nl - 1]["l"]  # number of c (prior causes)

    # Initialise cell arrays for hierarchical structure
    df = {key: generate_2d_dict(nl - 1, nl - 1) for key in ["dv", "dx", "dp"]}
    dg = {key: generate_2d_dict(nl, nl - 1) for key in ["dv", "dx", "dp"]}

    for i in range(nl - 1):
        dg["dv"][i + 1][i] = np.zeros((M[i]["m"], M[i]["m"]))
        dg["dx"][i + 1][i] = np.zeros((M[i]["m"], M[i]["n"]))
        dg["dp"][i + 1][i] = np.zeros((M[i]["m"], M[i]["p"]))
        dg["dv"][i][i] = np.zeros((M[i]["l"], M[i]["m"]))
        dg["dx"][i][i] = np.zeros((M[i]["l"], M[i]["n"]))
        dg["dp"][i][i] = np.zeros((M[i]["l"], M[i]["p"]))
        df["dv"][i][i] = np.zeros((M[i]["n"], M[i]["m"]))
        df["dx"][i][i] = np.zeros((M[i]["n"], M[i]["n"]))
        df["dp"][i][i] = np.zeros((M[i]["n"], M[i]["p"]))

    if bilinear:
        for i in range(nl - 1):
            dg["dvp"] = {i: [copy.deepcopy(dg["dv"]) for _ in range(M[i]["p"])]}
            dg["dxp"] = {i: [copy.deepcopy(dg["dx"]) for _ in range(M[i]["p"])]}
            df["dvp"] = {i: [copy.deepcopy(df["dv"]) for _ in range(M[i]["p"])]}
            df["dxp"] = {i: [copy.deepcopy(df["dx"]) for _ in range(M[i]["p"])]}

    # Inline function for evaluating projected parameters
    h = lambda f, x, v, q, u, p: f(
        x, v, unvectorise_object(vectorise_object(p) + (u @ q).reshape((-1)), p)
    )

    for i in range(nl - 1):
        # States level i
        xvp = [x[i], v[i], qp["p"][i], qp["u"][i], M[i]["pE"]]

        # 1st and 2nd partial derivatives (states)
        if bilinear and np_:
            try:
                dgdxp, dgdx = get_diff(h, M[i]["gx"], *xvp, 4, "q")
                dgdvp, dgdv = get_diff(h, M[i]["gv"], *xvp, 4, "q")
                dfdxp, dfdx = get_diff(h, M[i]["fx"], *xvp, 4, "q")
                dfdvp, dfdv = get_diff(h, M[i]["fv"], *xvp, 4, "q")
            except Exception:
                dgdxp, dgdx, _ = get_diff(h, M[i]["g"], *xvp, np.array([2, 4]), "q")
                dgdvp, dgdv, _ = get_diff(h, M[i]["g"], *xvp, np.array([3, 4]), "q")
                dfdxp, dfdx, *_ = get_diff(h, M[i]["f"], *xvp, np.array([2, 4]), "q")
                dfdvp, dfdv, *_ = get_diff(h, M[i]["f"], *xvp, np.array([3, 4]), "q")
        else:
            try:
                dgdx = h(M[i]["gx"], *xvp)
                dgdv = h(M[i]["gv"], *xvp)
                dfdx = h(M[i]["fx"], *xvp)
                dfdv = h(M[i]["fv"], *xvp)
            except Exception:
                dgdx, _ = get_diff(h, M[i]["g"], *xvp, 2)
                dgdv, _ = get_diff(h, M[i]["g"], *xvp, 3)
                dfdx, _ = get_diff(h, M[i]["f"], *xvp, 2)
                dfdv, _ = get_diff(h, M[i]["f"], *xvp, 3)

        # 1st-order partial derivatives (parameters)
        try:
            dfdp = h(M[i]["fp"], *xvp)
            dgdp = h(M[i]["gp"], *xvp)
        except Exception:
            dfdp, _ = get_diff(h, M[i]["f"], *xvp, 4)
            dgdp, _ = get_diff(h, M[i]["g"], *xvp, 4)

        # Constant terms (linking causes over levels)
        dg["dv"][i + 1][i] = -np.eye(M[i]["m"], M[i]["m"])

        # Place 1st derivatives in array
        dg["dx"][i][i] = dgdx
        dg["dv"][i][i] = dgdv
        df["dx"][i][i] = dfdx
        df["dv"][i][i] = dfdv
        df["dp"][i][i] = dfdp
        dg["dp"][i][i] = dgdp

        # Place 2nd derivatives in array

        if bilinear and np_:
            for j in range(len(dgdxp)):
                dg["dxp"][i][j][i][i] = dgdxp[j]
                dg["dvp"][i][j][i][i] = dgdvp[j]
                df["dxp"][i][j][i][i] = dfdxp[j]
                df["dvp"][i][j][i][i] = dfdvp[j]

    # Concatenate hierarchical forms
    D = {
        "dgdv": concatenate_dod_lol(dg["dv"]),
        "dgdx": concatenate_dod_lol(dg["dx"]),
        "dfdv": concatenate_dod_lol(df["dv"]),
        "dfdx": concatenate_dod_lol(df["dx"]),
        "dfdp": concatenate_dod_lol(df["dp"]),
        "dgdp": concatenate_dod_lol(dg["dp"]),
        "dfdy": np.zeros((nx, ny)),
        "dfdc": np.zeros((nx, nc)),
        "dedy": np.eye(ne, ny),
        "dedc": -np.eye(ne, nc, nc - ne),
    }

    # Bilinear terms if required
    if bilinear:
        D["dgdvp"] = []
        D["dgdxp"] = []
        D["dfdvp"] = []
        D["dfdxp"] = []
        for i in range(len(dg["dvp"])):
            for j in range(len(dg["dvp"][i])):
                D["dgdvp"].append(concatenate_dod_lol(dg["dvp"][i][j]))
                D["dgdxp"].append(concatenate_dod_lol(dg["dxp"][i][j]))
                D["dfdvp"].append(concatenate_dod_lol(df["dvp"][i][j]))
                D["dfdxp"].append(concatenate_dod_lol(df["dxp"][i][j]))

    return D


def get_DEM_eval(M, qu, qp):
    """
    Evaluate state equations and derivatives for DEM schemes.

    Parameters
    ----------
    M : dict
        Model structure.
    qu : dict of lists
        Conditional mode of states.
        qu['v'][i] - causal states
        qu['x'][i] - hidden states
        qu['y'][i] - response
        qu['u'][i] - input
    qp : dict of lists
        Conditional density of parameters.
        qp['p'][i] - parameter deviates for i-th level
        qp['u'][i] - basis set
        qp['x'][i] - expansion point (= prior expectation)

    Returns
    -------
    E : list
        Generalised errors (i.e., y - g(x,v,P); x[1] - f(x,v,P)).
    dE : dict
        Derivatives.
    f : list
        Evaluated state equations.
    g : list
        Evaluated output equations.
    """
    # get dimensions
    nl = len(M)  # number of levels
    ne = sum(vectorise_object([m["l"] for m in M])).astype(int)  # number of e (errors)
    nv = sum(vectorise_object([m["m"] for m in M])).astype(
        int
    )  # number of v (causal states)
    nx = sum(vectorise_object([m["n"] for m in M])).astype(
        int
    )  # number of x (hidden states)
    np_ = sum(vectorise_object([m["p"] for m in M])).astype(
        int
    )  # number of p (parameters)

    # evaluate functions at each hierarchical level
    v = unvectorise_object(qu["v"][0], [M[i + 1]["v"] for i in range(nl - 1)])
    x = unvectorise_object(qu["x"][0], [M[i]["x"] for i in range(nl - 1)])
    f = []
    g = []
    for i in range(nl - 1):
        # p = unvectorise_object(
        #     vectorise_object(M[i]["pE"]) + (qp["u"][i] @ qp["p"][i]).reshape((-1)),
        #     M[i]["pE"],
        # )
        if qp["u"][i] is not None and qp["p"][i] is not None:
            p_vectorised = vectorise_object(M[i]["pE"])
            qp_u_p_product = (qp["u"][i] @ qp["p"][i]).reshape((-1))
            p = unvectorise_object(p_vectorised + qp_u_p_product, M[i]["pE"])
        else:
            p = copy.deepcopy(M[i]["pE"])
        f.append(M[i]["f"](x[i], v[i], p))
        g.append(M[i]["g"](x[i], v[i], p))
    # Get Derivatives
    try:
        method = M[0]["E"]["linear"]
    except KeyError:
        method = 0
    global D
    if method == 0:
        # Full evaluation for derivatives at each D-step
        D = get_DEM_eval_diff(x, v, qp, M)

        # Gradients w.r.t. states
        dedy = D["dedy"]
        dedc = D["dedc"]
        dfdy = D["dfdy"]
        dfdc = D["dfdc"]
        dgdx = D["dgdx"]
        dgdv = D["dgdv"]
        dfdv = D["dfdv"]
        dfdx = D["dfdx"]
        dgdxp = D["dgdxp"]
        dfdxp = D["dfdxp"]
        dgdvp = D["dgdvp"]
        dfdvp = D["dfdvp"]

        # Gradients w.r.t. parameters
        dgdp = D["dgdp"]
        dfdp = D["dfdp"]
    elif method == 1:
        # Linear case
        if not D:
            D = get_DEM_eval_diff(x, v, qp, M)
            D["x"] = copy.deepcopy(x)
            D["v"] = copy.deepcopy(v)

            # Gradients w.r.t. states
            dedc = copy.deepcopy(D["dedc"])
            dedy = copy.deepcopy(D["dedy"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdv = copy.deepcopy(D["dfdv"])
            dfdx = copy.deepcopy(D["dfdx"])

            # Gradients w.r.t. parameters (state-dependent)
            dgdxp = copy.deepcopy(D["dgdxp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdvp = copy.deepcopy(D["dfdvp"])

            # Gradients w.r.t. parameterscopy.deepcopy(
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
        else:
            # Gradients w.r.t. states
            dedy = copy.deepcopy(D["dedy"])
            dedc = copy.deepcopy(D["dedc"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdv = copy.deepcopy(D["dfdv"])
            dfdx = copy.deepcopy(D["dfdx"])

            # Gradients w.r.t. parameters (state-dependent)
            dgdxp = copy.deepcopy(D["dgdxp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdvp = copy.deepcopy(D["dfdvp"])
            # Linear expansion for derivatives w.r.t. parameters
            dx = vectorise_object(qu["x"][0]) - vectorise_object(D["x"])
            dv = vectorise_object(qu["v"][0]) - vectorise_object(D["v"])
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
            for p in range(np_):
                dgdp[:, p] = D["dgdp"][:, p] + D["dgdxp"][p] @ dx + D["dgdvp"][p] @ dv
                if nx:
                    dfdp[:, p] = (
                        D["dfdp"][:, p] + D["dfdxp"][p] @ dx + D["dfdvp"][p] @ dv
                    )
    elif method == 2:
        # Bilinear case
        if not D:
            # Get high-order derivatives
            Dv, D = get_diff(get_DEM_eval_diff, x, v, qp, M, 2)

            for i in range(nv):
                Dv[i] = unvectorise_object(Dv[i], D)
            D["x"] = copy.deepcopy(x)
            D["v"] = copy.deepcopy(v)
            D["Dv"] = copy.deepcopy(Dv)

            # Gradients w.r.t. states
            dedy = copy.deepcopy(D["dedy"])
            dedc = copy.deepcopy(D["dedc"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdv = copy.deepcopy(D["dfdv"])
            dfdx = copy.deepcopy(D["dfdx"])
            dgdxp = copy.deepcopy(D["dgdxp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdvp = copy.deepcopy(D["dfdvp"])

            # Gradients w.r.t. parameters
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
        else:
            # Gradients w.r.t. states
            dedy = copy.deepcopy(D["dedy"])
            dedc = copy.deepcopy(D["dedc"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])
            # Second-order derivatives
            dv = vectorise_object(qu["v"][0]) - vectorise_object(D["v"])
            # Update gradients w.r.t. states and parameters
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdv = copy.deepcopy(D["dfdv"])
            dfdx = copy.deepcopy(D["dfdx"])
            for i in range(nv):
                dgdx += D["Dv"][i]["dgdx"] @ dv[i]
                dgdv += D["Dv"][i]["dgdv"] @ dv[i]
                dfdx += D["Dv"][i]["dfdx"] @ dv[i]
                dfdv += D["Dv"][i]["dfdv"] @ dv[i]

            dgdxp = copy.deepcopy(D["dgdxp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdvp = copy.deepcopy(D["dfdvp"])
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])

            for p in range(np_):
                for i in range(nv):
                    dgdxp[p] += D["Dv"][i]["dgdxp"][p] @ dv[i]
                    dgdvp[p] += D["Dv"][i]["dgdvp"][p] @ dv[i]
                    dfdxp[p] += D["Dv"][i]["dfdxp"][p] @ dv[i]
                    dfdvp[p] += D["Dv"][i]["dfdvp"][p] @ dv[i]
                # Update gradients w.r.t. parameters
                # Dgdxp = (D["dgdxp"][p] + dgdxp[p]) / 2
                Dgdvp = (D["dgdvp"][p] + dgdvp[p]) / 2
                # Dfdxp = (D["dfdxp"][p] + dfdxp[p]) / 2
                Dfdvp = (D["dfdvp"][p] + dfdvp[p]) / 2
                dgdp[:, p] += Dgdvp @ dv
                dfdp[:, p] += Dfdvp @ dv
    elif method == 3:
        # get derivatives and store expansion point (states)
        if not D:
            # get high-order derivatives
            Dx, D = get_diff("get_DEM_eval_diff", x, v, qp, M, 1, "q")
            Dv, D = get_diff("get_DEM_eval_diff", x, v, qp, M, 2, "q")

            for i in range(nx):
                Dx[i] = unvectorise_object(Dx[i], D)
            for i in range(nv):
                Dv[i] = unvectorise_object(Dv[i], D)
            D["x"] = copy.deepcopy(x)
            D["v"] = copy.deepcopy(v)
            D["Dx"] = copy.deepcopy(Dx)
            D["Dv"] = copy.deepcopy(Dv)

            # gradients w.r.t. states
            dedy = copy.deepcopy(D["dedy"])
            dedc = copy.deepcopy(D["dedc"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdv = copy.deepcopy(D["dfdv"])
            dfdx = copy.deepcopy(D["dfdx"])
            dgdxp = copy.deepcopy(D["dgdxp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdvp = copy.deepcopy(D["dfdvp"])

            # gradients w.r.t. parameters
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
        else:
            # gradients w.r.t. causes and data
            dedy = copy.deepcopy(D["dedy"])
            dedc = copy.deepcopy(D["dedc"])
            dfdy = copy.deepcopy(D["dfdy"])
            dfdc = copy.deepcopy(D["dfdc"])

            # states (relative to expansion point)
            dx = vectorise_object(qu["x"][0]) - vectorise_object(D["x"])
            dv = vectorise_object(qu["v"][0]) - vectorise_object(D["v"])

            # gradients w.r.t. states
            dgdx = copy.deepcopy(D["dgdx"])
            dgdv = copy.deepcopy(D["dgdv"])
            dfdx = copy.deepcopy(D["dfdx"])
            dfdv = copy.deepcopy(D["dfdv"])
            for i in range(nx):
                dgdx += D["Dx"][i]["dgdx"] @ dx[i]
            for i in range(nv):
                dgdx += D["Dv"][i]["dgdx"] @ dv[i]
            for i in range(nx):
                dgdv += D["Dx"][i]["dgdv"] @ dx[i]
            for i in range(nv):
                dgdv += D["Dv"][i]["dgdv"] @ dv[i]
            for i in range(nx):
                dfdx += D["Dx"][i]["dfdx"] @ dx[i]
            for i in range(nv):
                dfdx += D["Dv"][i]["dfdx"] @ dv[i]
            for i in range(nx):
                dfdv += D["Dx"][i]["dfdv"] @ dx[i]
            for i in range(nv):
                dfdv += D["Dv"][i]["dfdv"] @ dv[i]
            # second-order derivatives
            dgdxp = copy.deepcopy(D["dgdxp"])
            dgdvp = copy.deepcopy(D["dgdvp"])
            dfdxp = copy.deepcopy(D["dfdxp"])
            dfdvp = copy.deepcopy(D["dfdvp"])
            for p in range(np_):
                for i in range(nx):
                    dgdxp[p] += D["Dx"][i]["dgdxp"][p] @ dx[i]
                for i in range(nv):
                    dgdxp[p] += D["Dv"][i]["dgdxp"][p] @ dv[i]
                for i in range(nx):
                    dgdvp[p] += D["Dx"][i]["dgdvp"][p] @ dx[i]
                for i in range(nv):
                    dgdvp[p] += D["Dv"][i]["dgdvp"][p] @ dv[i]
                for i in range(nx):
                    dfdxp[p] += D["Dx"][i]["dfdxp"][p] @ dx[i]
                for i in range(nv):
                    dfdxp[p] += D["Dv"][i]["dfdxp"][p] @ dv[i]
                for i in range(nx):
                    dfdvp[p] += D["Dx"][i]["dfdvp"][p] @ dx[i]
                for i in range(nv):
                    dfdvp[p] += D["Dv"][i]["dfdvp"][p] @ dv[i]

            # gradients w.r.t. parameters
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
            for p in range(np_):
                Dgdxp = (D["dgdxp"][p] + dgdxp[p]) / 2
                Dgdvp = (D["dgdvp"][p] + dgdvp[p]) / 2
                Dfdxp = (D["dfdxp"][p] + dfdxp[p]) / 2
                Dfdvp = (D["dfdvp"][p] + dfdvp[p]) / 2
                dgdp[:, p] = dgdp[:, p] + Dgdxp @ dx + Dgdvp @ dv
                dfdp[:, p] = dfdp[:, p] + Dfdxp @ dx + Dfdvp @ dv
    elif method == 4:
        if not D:
            D = get_DEM_eval_diff(x, v, qp, M)
            D["x"] = x
            D["v"] = v

            # gradients w.r.t. states
            dedy = D["dedy"]
            dedc = D["dedc"]
            dfdy = D["dfdy"]
            dfdc = D["dfdc"]
            dgdx = D["dgdx"]
            dgdv = D["dgdv"]
            dfdv = D["dfdv"]
            dfdx = D["dfdx"]

            # gradients w.r.t. parameters (state-dependent)
            dgdxp = D["dgdxp"].copy()
            dfdxp = D["dfdxp"].copy()
            dgdvp = D["dgdvp"].copy()
            dfdvp = D["dfdvp"].copy()

            # gradients w.r.t. parameters
            dgdp = D["dgdp"]
            dfdp = D["dfdp"]
        else:
            # retain second-order gradients
            dgdxp = D["dgdxp"].copy()
            dfdxp = D["dfdxp"].copy()
            dgdvp = D["dgdvp"].copy()
            dfdvp = D["dfdvp"].copy()

            # re-evaluate first-order gradients
            D = get_DEM_eval_diff(x, v, qp, M, 0)
            dedy = D["dedy"]
            dedc = D["dedc"]
            dfdy = D["dfdy"]
            dfdc = D["dfdc"]
            dgdx = D["dgdx"]
            dgdv = D["dgdv"]
            dfdv = D["dfdv"]
            dfdx = D["dfdx"]

            # replace second-order gradients
            D["dgdxp"] = dgdxp.copy()
            D["dfdxp"] = dfdxp.copy()
            D["dgdvp"] = dgdvp.copy()
            D["dfdvp"] = dfdvp.copy()

            # gradients w.r.t. parameters
            dx = vectorise_object(qu["x"][0]) - vectorise_object(x)
            dv = vectorise_object(qu["v"][0]) - vectorise_object(v)
            dgdp = copy.deepcopy(D["dgdp"])
            dfdp = copy.deepcopy(D["dfdp"])
            for p in range(np_):
                dgdp[:, p] = D["dgdp"][:, p] + D["dgdxp"][p] @ dx + D["dgdvp"][p] @ dv
                if nx:
                    dfdp[:, p] = (
                        D["dfdp"][:, p] + D["dfdxp"][p] @ dx + D["dfdvp"][p] @ dv
                    )
    else:
        print("Unknown method")

    # order parameters (d = n = 1 for static models)
    d = M[0]["E"]["d"] + 1  # generalisation order of q(v)
    n = M[0]["E"]["n"] + 1  # embedding order (n >= d)

    # Generalised prediction errors and derivatives
    Ex = [np.zeros((nx, 1)) for _ in range(n)]
    Ev = [np.zeros((ne, 1)) for _ in range(n)]

    # prediction error (E) - causes
    for i in range(n):
        qu["y"][i] = vectorise_object(qu["y"][i]).reshape((-1, 1))
    Ev[0] = np.vstack([qu["y"][0], qu["v"][0]]) - np.vstack(
        [vectorise_object(g).reshape((-1, 1)), qu["u"][0].reshape((-1, 1))]
    )

    for i in range(1, n):
        Ev[i] = (
            dedy @ qu["y"][i]
            + dedc @ qu["u"][i].reshape((-1, 1))
            - dgdx @ qu["x"][i]
            - dgdv @ qu["v"][i]
        )

    # prediction error (E) - states
    try:
        Ex[0] = qu["x"][1] - vectorise_object(f).reshape((-1, 1))
    except Exception:
        pass
    for i in range(1, n - 1):
        Ex[i] = qu["x"][i + 1] - dfdx @ qu["x"][i] - dfdv @ qu["v"][i]

    # error
    E = vectorise_object([Ev, Ex])

    # # Kronecker forms of derivatives for generalised motion
    # if "nargout" not in locals() or nargout < 2:
    #     pass  # equivalent to MATLAB's return for nargout < 2

    # dE.dp (parameters)
    dgdp = [dgdp]
    dfdp = [dfdp]
    for i in range(1, n):
        dgdp.append(copy.deepcopy(dgdp[0]))
        dfdp.append(copy.deepcopy(dfdp[0]))
        for p in range(np_):
            dgdp[i][:, p] = (dgdxp[p] @ qu["x"][i] + dgdvp[p] @ qu["v"][i]).squeeze()
            dfdp[i][:, p] = (dfdxp[p] @ qu["x"][i] + dfdvp[p] @ qu["v"][i]).squeeze()
    dd = dgdp + dfdp
    # generalised temporal derivatives: dE.du (states)
    dedy = np.kron(np.eye(n), dedy)
    dedc = np.kron(np.eye(n, d), dedc)
    dfdy = np.kron(np.eye(n), dfdy)
    dfdc = np.kron(np.eye(n, d), dfdc)
    dgdx = np.kron(np.eye(n), dgdx)
    dgdv = np.kron(np.eye(n, d), dgdv)
    dfdv = np.kron(np.eye(n, d), dfdv)
    dfdx = np.kron(np.eye(n), dfdx) - np.kron(np.eye(n, n, 1), np.eye(nx))

    # 1st error derivatives (states)
    dE = {}
    dE["dy"] = concatenate_dod_lol([[dedy], [dfdy]])
    dE["dc"] = concatenate_dod_lol([[dedc], [dfdc]])
    dE["dp"] = 0 - concatenate_dod_lol([[element] for element in dd])
    dE["du"] = 0 - concatenate_dod_lol([[dgdx, dgdv], [dfdx, dfdv]])
    dE["dup"] = [None for _ in range(np_)]

    # bilinear derivatives
    for i in range(np_):
        dgdxp[i] = np.kron(np.eye(n), dgdxp[i])
        dfdxp[i] = np.kron(np.eye(n), dfdxp[i])
        dgdvp[i] = np.kron(np.eye(n, d), dgdvp[i])
        dfdvp[i] = np.kron(np.eye(n, d), dfdvp[i])
        dE["dup"][i] = 0 - concatenate_dod_lol(
            [[dgdxp[i], dgdvp[i]], [dfdxp[i], dfdvp[i]]]
        )
    if np_:
        dE["dpu"] = swap_dod_lol([dE["dup"]])
    else:
        dE["dpu"] = []

    return E, dE, f, g


def clear_get_DEM_eval():
    global D
    D = None


def get_LAP(DEM):
    """
    Laplacian model inversion (see also spm_LAPS).

    Parameters
    ----------
    DEM : dict
        A dictionary containing the DEM structures including M, Y, U, and possibly others.

    Returns
    -------
    dict
        The updated DEM dictionary after Laplacian model inversion.

    Notes
    -----
    spm_LAP implements a variational scheme under the Laplace approximation to the conditional joint density q on states u, parameters p, and hyperparameters (h,g) of an analytic nonlinear hierarchical dynamic model, with additive Gaussian innovations.

            q(u,p,h,g) = max <L(t)>q

    L is the ln p(y,u,p,h,g|M) under the model M. The conditional covariances obtain analytically from the curvature of L with respect to the unknowns.

    Generative model:
    -----------------
    - M(i).g  = v     =  g(x,v,P)   {inline function, string or m-file}
    - M(i).f  = dx/dt =  f(x,v,P)   {inline function, string or m-file}
    - M(i).ph = pi(v) = ph(x,v,h,M) {inline function, string or m-file}
    - M(i).pg = pi(x) = pg(x,v,g,M) {inline function, string or m-file}
                                    (assumed to be linear in v and x)
    - pi(v,x) = vectors of log-precisions; (h,g) = precision parameters
    - M(i).pE = prior expectation of p model-parameters
    - M(i).pC = prior covariances of p model-parameters
    - M(i).hE = prior expectation of h log-precision (cause noise)
    - M(i).hC = prior covariances of h log-precision (cause noise)
    - M(i).gE = prior expectation of g log-precision (state noise)
    - M(i).gC = prior covariances of g log-precision (state noise)
    - M(i).xP = precision (states)
    - M(i).Q  = precision components (input noise)
    - M(i).R  = precision components (state noise)
    - M(i).V  = fixed precision (input noise)
    - M(i).W  = fixed precision (state noise)
    - M(i).P  = optional initial value for parameters (defaults to M(i).pE)
    - M(i).m  = number of inputs v(i + 1);
    - M(i).n  = number of states x(i);
    - M(i).l  = number of output v(i).

    Conditional moments of model-states - q(u):
    ------------------------------------------
    - qU.x    = Conditional expectation of hidden states
    - qU.v    = Conditional expectation of causal states
    - qU.w    = Conditional prediction error (states)
    - qU.z    = Conditional prediction error (causes)
    - qU.C    = Conditional covariance: cov(v)
    - qU.S    = Conditional covariance: cov(x)

    Conditional moments of model-parameters - q(p):
    ----------------------------------------------
    - qP.P    = Conditional expectation
    - qP.C    = Conditional covariance

    Conditional moments of hyper-parameters (log-transformed) - q(h):
    ------------------------------------------------------------------
    - qH.h    = Conditional expectation (cause noise)
    - qH.g    = Conditional expectation (state noise)
    - qH.C    = Conditional covariance

    F         = log-evidence = log-marginal likelihood = negative free-energy

    Accelerated methods:
    --------------------
    To accelerate computations one can specify the nature of the model equations using:

    - M(1).E.linear = 0: full        - evaluates 1st and 2nd derivatives
    - M(1).E.linear = 1: linear      - equations are linear in x and v
    - M(1).E.linear = 2: bilinear    - equations are linear in x, v & x*v
    - M(1).E.linear = 3: nonlinear   - equations are linear in x, v, x*v, & x*x
    - M(1).E.linear = 4: full linear - evaluates 1st derivatives (for GF)

    Similarly, for evaluating precisions:

    - M(1).E.method.h = 0,1  switch for precision parameters (hidden causes)
    - M(1).E.method.g = 0,1  switch for precision parameters (hidden states)
    - M(1).E.method.v = 0,1  switch for precision (hidden causes)
    - M(1).E.method.x = 0,1  switch for precision (hidden states)
    """
    DEM = copy.deepcopy(DEM)
    # Check model, data, and priors
    DEM_update = get_DEM_set(DEM)
    M = DEM_update["M"]
    Y = DEM_update["Y"]
    U = DEM_update["U"]

    # Set regularisation
    try:
        dt = DEM["M"][0]["E"]["v"]
    except KeyError:
        dt = 0
        DEM["M"][0]["E"]["v"] = dt

    # number of iterations
    try:
        nD = copy.deepcopy(M[0]["E"]["nD"])
    except KeyError:
        nD = 1

    try:
        nN = copy.deepcopy(M[0]["E"]["nN"])
    except KeyError:
        nN = 8

    # ensure integration scheme evaluates gradients at each time-step
    M[0]["E"]["linear"] = 4

    # assume precisions are a function of, and only of, hyperparameters
    try:
        method = copy.deepcopy(M[0]["E"]["method"])
    except KeyError:
        method = {"h": 1, "g": 1, "x": 0, "v": 0}

    method.setdefault("h", 0)
    method.setdefault("g", 0)
    method.setdefault("x", 0)
    method.setdefault("v", 0)

    # assume precisions are a function of, and only of, hyperparameters
    try:
        form = copy.deepcopy(M[0]["E"]["form"])
    except KeyError:
        form = "Gaussian"

    # checks for Laplace models (precision functions; ph and pg)
    for i, model in enumerate(M):
        try:
            getattr(model, "ph")(model["x"], M[i + 1]["v"], model["hE"], model)
            method["v"] = 1
        except (AttributeError, KeyError):
            model["ph"] = get_LAP_ph
        try:
            getattr(model, "pg")(model["x"], M[i + 1]["v"], model["gE"], model)
            method["x"] = 1
        except (AttributeError, KeyError):
            model["pg"] = get_LAP_pg

    M[0]["E"]["method"] = method

    # order parameters (d = n = 1 for static models) and checks
    d = M[0]["E"]["d"] + 1  # embedding order of q(v)
    n = M[0]["E"]["n"] + 1  # embedding order of q(x)

    # number of states and parameters
    ns = Y.shape[1]  # number of samples
    nl = len(M)  # number of levels
    nv = int(
        sum(vectorise_object([model["m"] for model in M]))
    )  # number of v (casual states)
    nx = int(
        sum(vectorise_object([model["n"] for model in M]))
    )  # number of x (hidden states)
    ny = M[0]["l"]  # number of y (inputs)
    nc = M[-1]["l"]  # number of c (prior causes)
    nu = int(nv * d + nx * n)  # number of generalised states
    ne = int(nv * n + nx * n + ny * n)  # number of generalised errors

    s = copy.deepcopy(M[0]["E"]["s"])
    Rh, _ = get_DEM_R(n, s, form)
    Rg, _ = get_DEM_R(n, s, form)

    if not nx:
        Rg = np.zeros((0, 0))

    W = np.zeros((nx * n, nx * n))
    V = np.zeros(((ny + nv) * n, (ny + nv) * n))

    # fixed priors on states (u)
    Px = np.kron(
        get_DEM_R(n, 2)[0],
        concatenate_dod_lol(get_diag([model["xP"] for model in M[:]])),
    )
    Pv = np.kron(
        get_DEM_R(d, 2)[0],
        concatenate_dod_lol(get_diag([model["vP"] for model in M[1:]])),
    )
    Pu = concatenate_dod_lol(get_diag([Px, Pv]))

    # hyperpriors
    ph_h = vectorise_object(
        [model["hE"] for model in M] + [model["gE"] for model in M]
    )  # prior expectation of h,g
    ph_c = concatenate_dod_lol(
        get_diag([model["hC"] for model in M] + [model["gC"] for model in M])
    )  # prior covariances of h,g
    ph = {"h": ph_h, "c": ph_c}
    Ph = get_inv(ph_c)  # prior precision of h,g
    qh = {}
    qh["h"] = [
        copy.deepcopy(model["hE"].astype(np.float64)) for model in M
    ]  # conditional expectation h
    qh["g"] = [copy.deepcopy(model["gE"]) for model in M]  # conditional expectation g
    nh = len(vectorise_object(qh["h"]))  # number of hyperparameters h
    ng = len(vectorise_object(qh["g"]))  # number of hyperparameters g
    nb = nh + ng  # number of hyperparameters

    # priors on parameters (in reduced parameter space)
    # ==========================================================================
    pp = {"c": generate_2d_dict(nl, nl), "p": None}
    qp = {"p": [None for _ in range(nl)], "u": [None for _ in range(nl - 1)]}
    for i in range(nl - 1):
        # eigenvector reduction: p <- pE + qp.u*qp.p
        # ----------------------------------------------------------------------
        u, _, _ = truncate_svd(M[i]["pC"], tol=0)
        qp["u"][i] = u  # basis for parameters
        M[i]["p"] = u.shape[1]  # number of qp.p
        qp["p"][i] = np.zeros((M[i]["p"], 1))  # initial deviates
        pp["c"][i][i] = u.T @ M[i]["pC"] @ u  # prior covariance
    M[nl - 1]["p"] = np.array([])
    Up = concatenate_dod_lol(get_diag([u for u in qp["u"]]))

    # priors on parameters
    # --------------------------------------------------------------------------
    pp["p"] = vectorise_object([m["pE"] for m in M])
    pp["c"] = concatenate_dod_lol(pp["c"])
    Pp = get_inv(pp["c"])

    # initialise conditional density q(p)
    # --------------------------------------------------------------------------
    for i in range(nl - 1):
        try:
            qp["p"][i] += qp["u"][i].T @ (
                vectorise_object(M[i]["P"]) - vectorise_object(M[i]["pE"])
            )
        except Exception:
            pass

    np_ = Up.shape[1]

    # initialise cell arrays for D-Step; e{i + 1} = (d/dt)^i[e] = e[i]
    # ==========================================================================
    qu = {
        "x": [np.zeros((nx, 1)) for _ in range(n)],
        "v": [np.zeros((nv, 1)) for _ in range(n)],
        "y": [np.zeros((ny, 1)) for _ in range(n)],
        "u": [np.zeros((nc, 1)) for _ in range(n)],
    }

    # initialise cell arrays for hierarchical structure of x[0] and v[0]
    # --------------------------------------------------------------------------
    x = [copy.deepcopy(m["x"]) for m in M[:-1]]
    v = [copy.deepcopy(m["v"]) for m in M[1:]]
    qu["x"][0] = vectorise_object(x).reshape((-1, 1))
    qu["v"][0] = vectorise_object(v).reshape((-1, 1))

    # derivatives for Jacobian of D-step
    # --------------------------------------------------------------------------
    Dx = np.kron(np.eye(n, n, 1), np.eye(nx))
    Dv = np.kron(np.eye(d, d, 1), np.eye(nv))
    Dy = np.kron(np.eye(n, n, 1), np.eye(ny))
    Dc = np.kron(np.eye(d, d, 1), np.eye(nc))
    Du = concatenate_dod_lol(get_diag([Dx, Dv]))
    Ib = np.eye(np_ + nb)
    dbdt = np.zeros((np_ + nb, 1))

    # gradients of generalised weighted errors
    # --------------------------------------------------------------------------
    dedh = np.zeros((nh, ne))
    dedg = np.zeros((ng, ne))
    dedv = np.zeros((nv, ne))
    dedx = np.zeros((nx, ne))
    dedhh = np.zeros((nh, nh))
    dedgg = np.zeros((ng, ng))

    # curvatures of Gibb's energy w.r.t. hyperparameters
    # --------------------------------------------------------------------------
    dHdh = np.zeros((nh, 1))
    dHdg = np.zeros((ng, 1))
    dHdp = np.zeros((np_, 1))
    dHdu = np.zeros((nu, 1))

    # preclude unnecessary iterations and set switches
    # --------------------------------------------------------------------------
    if not np_ and not nh and not ng:
        nN = 1
    mnx = nx * bool(method.get("x"))
    mnv = nv * bool(method.get("v"))

    # preclude very precise states from entering free-energy/action
    # --------------------------------------------------------------------------
    p = get_LAP_eval(M, qu, qh, nargout=1)
    ih = p["h"] < 8
    ig = p["g"] < 8
    ie = np.kron(np.ones((n, 1)), ih)
    ix = np.kron(np.ones((n, 1)), ig)
    iv = np.kron(np.ones((d, 1)), ih[np.arange(nv) + ny])
    je = np.where(np.concatenate([ie, ix]))[0]
    ix[:nx] = 1
    ju = np.where(np.concatenate([ix, iv]))[0]

    # and other useful indices
    # --------------------------------------------------------------------------
    ix = np.arange(nx)
    ih = np.arange(nb)
    iv = np.arange(nv) + nx * n

    # number of iterations for convergence
    # --------------------------------------------------------------------------
    convergence = -4
    global D
    D = None
    F = np.zeros((nN,))
    F[0] = -np.inf
    CC = np.zeros((nN, 9))
    S = np.zeros((nN,))

    for iN in range(nN):
        # get time and clear persistent variables in evaluation routines
        start_time = time.time()

        # [re-]set states & their derivatives
        clear_get_DEM_eval()
        try:
            qu = copy.deepcopy(Q[0]["u"])
        except Exception:
            pass
        Q = {
            i: {"e": None, "E": None, "u": None, "p": None, "h": None}
            for i in range(ns)
        }
        Fc = np.zeros((ns, 5))
        AC = np.zeros(ns)
        for is_ in range(ns):
            for iD in range(nD):
                ts = (is_ + 1) + iD / nD
                # print(ts)

                try:
                    qu["y"][:n] = get_DEM_embed(Y, n, ts, 1, M[0]["delays"])
                    qu["u"][:d] = get_DEM_embed(U, d, ts)
                except Exception:
                    qu["y"][:n] = get_DEM_embed(Y, n, ts)
                    qu["u"][:d] = get_DEM_embed(U, d, ts)

                E, dE, *_ = get_DEM_eval(M, qu, qp)
                p, dp = get_LAP_eval(M, qu, qh, nargout=2)

                iSh = np.diag(np.exp(p["h"].flatten()))
                iSg = np.diag(np.exp(p["g"].flatten()))
                iS = np.block(
                    [
                        [
                            np.kron(Rh, iSh),
                            np.zeros((len(Rh) * len(iSh), len(iSg) * len(Rg))),
                        ],
                        [
                            np.zeros((len(iSg) * len(Rg), len(Rh) * len(iSh))),
                            np.kron(Rg, iSg),
                        ],
                    ]
                )

                dpdx = n * np.sum(
                    concatenate_dod_lol([[dp["h"]["dx"]], [dp["g"]["dx"]]]),
                    axis=0,
                    keepdims=True,
                )
                dpdv = n * np.sum(
                    concatenate_dod_lol([[dp["h"]["dv"]], [dp["g"]["dv"]]]),
                    axis=0,
                    keepdims=True,
                )
                dpdh = n * np.sum(dp["h"]["dh"], axis=0)
                dpdg = n * np.sum(dp["g"]["dg"], axis=0)
                matrix_temp = np.zeros((1, n))
                matrix_temp[0, 0] = 1
                dpdx = np.kron(matrix_temp, dpdx)
                matrix_temp = np.zeros((1, d))
                matrix_temp[0, 0] = 1
                dpdv = np.kron(matrix_temp, dpdv)
                dDdu = np.concatenate([dpdx, dpdv], axis=1).T
                dDdh = np.hstack([dpdh, dpdg]).T
                diSdh = {}
                for i in range(nh):
                    diS = np.diag(dp["h"]["dh"][:, i] * np.exp(p["h"]).flatten())
                    diSdh[i] = np.block(
                        [
                            [
                                np.kron(Rh, diS),
                                np.zeros((len(diS) * len(Rh), W.shape[1])),
                            ],
                            [np.zeros((W.shape[0], len(diS) * len(Rh))), W],
                        ]
                    )
                    dedh[i, :] = E.T @ diSdh[i]
                diSdg = {}
                for i in range(ng):
                    diS = np.diag(dp["g"]["dg"][:, i] * np.exp(p["g"]))
                    diSdg[i] = np.block(
                        [
                            [V, np.zeros((len(diS), np.kron(Rg, diS).shape[1]))],
                            [
                                np.zeros((np.kron(Rg, diS).shape[0], V.shape[0])),
                                np.kron(Rg, diS),
                            ],
                        ]
                    )
                    dedg[i, :] = E.T @ diSdg[i]

                for i in range(mnx):
                    diV = np.diag(dp["h"]["dx"][:, i] * np.exp(p["h"].flatten()))
                    diW = np.diag(dp["g"]["dx"][:, i] * np.exp(p["g"].flatten()))
                    diSdx = np.block(
                        [
                            [
                                np.kron(Rh, diV),
                                np.zeros((len(Rh) * len(diV), len(diW) * len(Rg))),
                            ],
                            [
                                np.zeros((len(diW) * len(Rg), len(Rh) * len(diV))),
                                np.kron(Rg, diW),
                            ],
                        ]
                    )
                    dedx[i, :] = E.T @ diSdx

                for i in range(mnv):
                    diV = np.diag(dp["h"]["dv"][:, i] * np.exp(p["h"].flatten()))
                    diW = np.diag(dp["g"]["dv"][:, i] * np.exp(p["g"].flatten()))
                    diSdv = np.block(
                        [
                            [
                                np.kron(Rh, diV),
                                np.zeros((len(Rh) * len(diV), len(Rg) * len(diW))),
                            ],
                            [
                                np.zeros((len(Rg) * len(diW), len(Rh) * len(diV))),
                                np.kron(Rg, diW),
                            ],
                        ]
                    )
                    dedv[i, :] = E.T @ diSdv

                dSdx = np.kron(np.ones((n, 1)), dedx)
                dSdv = np.kron(np.ones((d, 1)), dedv)
                dSdu = np.vstack([dSdx, dSdv])
                dEdh = np.vstack([dedh, dedg])
                dEdp = dE["dp"].T @ iS
                dEdu = dE["du"].T @ iS

                # Curvatures w.r.t. hyperparameters
                for i in range(nh):
                    for j in range(i, nh):
                        diS = np.diag(
                            dp["h"]["dh"][:, i]
                            * dp["h"]["dh"][:, j]
                            * np.exp(p["h"].flatten())
                        )
                        diS = np.block(
                            [
                                [
                                    np.kron(Rh, diS),
                                    np.zeros((len(Rh) * len(diS), W.shape[1])),
                                ],
                                [np.zeros((W.shape[0], len(Rh) * len(diS))), W],
                            ]
                        )
                        dedhh[i, j] = E.T @ diS @ E
                        dedhh[j, i] = dedhh[i, j]

                for i in range(ng):
                    for j in range(i, ng):
                        diS = np.diag(
                            dp["g"]["dg"][:, i]
                            * dp["g"]["dg"][:, j]
                            * np.exp(p["g"].flatten())
                        )
                        diS = np.block(
                            [
                                [V, np.zeros((V.shape[0], len(Rg) * len(diS)))],
                                [
                                    np.zeros((len(Rg) * len(diS), V.shape[1])),
                                    np.kron(Rg, diS),
                                ],
                            ]
                        )
                        dedgg[i, j] = E.T @ diS @ E
                        dedgg[j, i] = dedgg[i, j]

                # Combined curvature
                dSdhh = concatenate_dod_lol(
                    [[dedhh, np.array([])], [np.array([]), dedgg]]
                )

                # Errors (from prior expectations)
                Eu = vectorise_object(qu["x"][0:n], qu["v"][0:d])
                Ep = vectorise_object(qp["p"])
                Eh = vectorise_object(qh["h"], qh["g"]) - ph["h"]

                # First-order derivatives of Gibb's Energy
                dLdu = dEdu @ E + dSdu @ E / 2 - (dDdu / 2).reshape((1, -1)) + Pu @ Eu
                dLdh = dEdh @ E / 2 - dDdh / 2 + Ph @ Eh
                dLdp = dEdp @ E + Pp @ Ep

                # And second-order derivatives of Gibb's Energy
                dLduu = dEdu @ dE["du"] + Pu
                dLdpp = dEdp @ dE["dp"] + Pp
                dLdhh = dSdhh / 2 + Ph
                dLduy = dEdu @ dE["dy"]
                dLduc = dEdu @ dE["dc"]
                dLdup = dEdu @ dE["dp"]
                dLdhp = dEdh @ dE["dp"]
                dLdpu = dLdup.T
                dLdph = dLdhp.T

                # Precision and covariances for entropy
                dLdaa = concatenate_dod_lol([[dLduu, dLdup], [dLdpu, dLdpp]])
                dLdbb = concatenate_dod_lol([[dLdpp, dLdph], [dLdhp, dLdhh]])

                Cup = get_inv(dLdaa)
                Chh = get_inv(dLdhh)

                # First-order derivatives of Entropy term

                # Log-precision
                for i in range(nh):
                    Luub = dE["du"].T @ diSdh[i] @ dE["du"]
                    Lpub = dE["dp"].T @ diSdh[i] @ dE["du"]
                    Lppb = dE["dp"].T @ diSdh[i] @ dE["dp"]
                    diCdh = concatenate_dod_lol([[Luub, Lpub.T], [Lpub, Lppb]])
                    dHdh[i] = get_trace(diCdh, Cup) / 2

                for i in range(ng):
                    Luub = dE["du"].T @ diSdg[i] @ dE["du"]
                    Lpub = dE["dp"].T @ diSdg[i] @ dE["du"]
                    Lppb = dE["dp"].T @ diSdg[i] @ dE["dp"]
                    diCdg = concatenate_dod_lol([[Luub, Lpub.T], [Lpub, Lppb]])
                    dHdg[i] = get_trace(diCdg, Cup) / 2

                # Parameters
                for i in range(np_):
                    Luup = dE["dup"][i].T @ dEdu.T
                    Lpup = dEdp @ dE["dup"][i]
                    Luup = Luup + Luup.T
                    diCdp = concatenate_dod_lol([[Luup, Lpup.T], [Lpup, np.array([])]])
                    dHdp[i] = get_trace(diCdp, Cup) / 2

                # Hidden states and causes (disabled for stability)

                # And concatenate
                dHdb = np.concatenate([dHdh, dHdg])
                dHdb = np.concatenate([dHdp, dHdb])
                dLdb = np.concatenate([dLdp, dLdh])

                if iD == 0 or ns == 1:
                    # Save means
                    Q[is_]["e"] = copy.deepcopy(E)
                    Q[is_]["E"] = copy.deepcopy(np.diag(np.diag(iS)) @ E)
                    Q[is_]["u"] = copy.deepcopy(qu)
                    Q[is_]["p"] = copy.deepcopy(qp)
                    Q[is_]["h"] = copy.deepcopy(qh)

                    # And conditional covariances
                    Q[is_]["u"]["s"] = copy.deepcopy(Cup[:nx, :nx])
                    Q[is_]["u"]["c"] = copy.deepcopy(
                        Cup[nx * n : nx * n + nv, nx * n : nx * n + nv]
                    )
                    Q[is_]["p"]["c"] = copy.deepcopy(Cup[nu : nu + np_, nu : nu + np_])
                    Q[is_]["h"]["c"] = copy.deepcopy(Chh[:nb, :nb])

                    # Free-energy (components)
                    Fc[is_, 0] = -E[je].T @ iS[np.ix_(je, je)] @ E[je] / 2
                    Fc[is_, 1] = -Eu[ju].T @ Pu[np.ix_(ju, ju)] @ Eu[ju] / 2
                    Fc[is_, 2] = -n * ny * np.log(2 * np.pi) / 2
                    Fc[is_, 3] = get_logdet(iS[np.ix_(je, je)]) / 2
                    Fc[is_, 4] = (
                        get_logdet(Pu[np.ix_(ju, ju)] @ Cup[np.ix_(ju, ju)]) / 2
                    )

                    # Free-action (states and parameters)
                    AC[is_] = (
                        sum(Fc[is_, :])
                        - Ep.T @ Pp @ Ep / 2
                        - Eh.T @ Ph @ Eh / 2
                        + get_logdet(Pp) / 2
                        + get_logdet(Ph) / 2
                        - get_logdet(dLdbb) / 2
                    )

                # Prior precision of fluctuations on [hyper] parameters
                Kb = ns * Ib

                # Accumulate curvatures of [hyper] parameters
                try:
                    dLdBB = dLdBB * (1 - 1 / ns) + dLdbb / ns
                except NameError:  # If dLdBB is not defined
                    dLdBB = dLdbb + Ib * 32

                # Whiten gradient (and curvatures) with regularised precision
                Cb = get_inv(dLdBB + np.diag(np.diag(dLdBB)) * np.exp(dt))
                dLdb = Cb @ dLdb
                dHdb = Cb @ dHdb
                q = {}
                # Assemble conditional means
                q["y"] = copy.deepcopy(qu["y"][:n])
                q["x"] = copy.deepcopy(qu["x"][:n])
                q["v"] = copy.deepcopy(qu["v"][:d])
                q["c"] = copy.deepcopy(qu["u"][:d])
                q["p"] = copy.deepcopy(qp["p"])
                q["h"] = copy.deepcopy(qh["h"])
                q["g"] = copy.deepcopy(qh["g"])
                q["d"] = copy.deepcopy(dbdt)

                # Flow
                f = {
                    "y": Dy @ vectorise_object(q["y"]),
                    "u": Du @ vectorise_object(q["x"], q["v"])
                    - dLdu.flatten()
                    - dHdu.flatten(),
                    "c": Dc @ vectorise_object(q["c"]),
                    "b": vectorise_object(q["d"]),
                    "d": -Kb @ vectorise_object(q["d"]) - dLdb - dHdb.flatten(),
                }

                # Jacobian
                dfdq = [
                    [Dy, None, None, None, None],
                    [-dLduy, Du - dLduu, -dLduc, None, None],
                    [None, None, Dc, None, None],
                    [None, None, None, None, Ib],
                    [None, None, None, -Ib, -Kb],
                ]

                dfdq = concatenate_dod_lol(dfdq)
                try:
                    if DEM["options"]["eigenvalues"]:
                        DEM["E"][:, is_] = np.linalg.eig(Du - dLduu)[0]
                except KeyError:
                    pass

                # Update conditional modes of states
                dq = get_dx(dfdq, vectorise_object(f), 1 / nD)
                q = unvectorise_object(vectorise_object(q) + dq, q)

                # Unpack conditional means
                qu["x"][0:n] = copy.deepcopy(q["x"])
                qu["v"][0:d] = copy.deepcopy(q["v"])
                qp["p"] = copy.deepcopy(q["p"])
                qh["h"] = copy.deepcopy(q["h"])
                qh["g"] = copy.deepcopy(q["g"])
                dbdt = copy.deepcopy(q["d"])

        # Bayesian parameter averaging
        # Conditional moments of time-averaged parameters
        Ep = 0
        Qp = 0
        for i in range(ns):
            P = get_inv(Q[i]["p"]["c"])
            Ep += P @ vectorise_object(Q[i]["p"]["p"])
            Qp += P
        Ep = get_inv(Qp) @ Ep
        Cp = get_inv(Qp + (1 - ns) * Pp)

        # Conditional moments of hyper-parameters
        Eh = 0
        Qh = 0
        for i in range(ns):
            P = get_inv(Q[i]["h"]["c"])
            Eh += P @ vectorise_object([Q[i]["h"]["h"], Q[i]["h"]["g"]])
            Qh += P
        Eh = get_inv(Qh) @ Eh - ph["h"]
        Ch = get_inv(Qh + (1 - ns) * Ph)

        # Free-action of states plus free-energy of parameters
        FT = np.sum(Fc, axis=1)  # Instantaneous Free energy (of states)
        FC = np.zeros(9)
        FC[:5] = np.sum(Fc, axis=0)
        FC[5] = -Ep.T @ Pp @ Ep / 2
        FC[6] = -Eh.T @ Ph @ Eh / 2
        FC[7] = get_logdet(Pp @ Cp) / 2
        FC[8] = get_logdet(Ph @ Ch) / 2

        CC[iN, :] = FC
        S[iN] = np.sum(AC)
        Fe = np.sum(FC)

        # if F is decreasing, revert [hyper] parameters and slow down
        if Fe < F[iN] and iN > 3:
            # save free-energy
            F[iN + 1] = copy.deepcopy(F[iN])

            # load current MAP estimates
            qp = copy.deepcopy(PQ["qp"])
            qh = copy.deepcopy(PQ["qh"])

            # decrease update time
            dt = max(dt + 2, 2)

            # convergence
            if dt > 6:
                convergence = 1
        else:
            # convergence
            if Fe - F[iN] < 1:
                convergence += 1

            # save free-energy
            F[iN] = copy.deepcopy(Fe)
            F[iN + 1] = copy.deepcopy(Fe)

            # save current MAP estimates
            PQ = {"qp": None, "qh": None}
            PQ["qp"] = copy.deepcopy(qp)
            PQ["qh"] = copy.deepcopy(qh)

            # increase update time
            dt = max(dt - 1, -8)

        if convergence > 0:
            break

        qU = {"v": {}, "x": {}, "w": {}, "z": {}, "Z": {}, "W": {}, "S": {}, "C": {}}
        qP = {"p": {}, "c": {}}
        qH = {"p": {}, "c": {}}
        qU["v"][0] = np.zeros((get_object_length(v[0]), ns))
        qU["v"][1] = np.zeros((get_object_length(v[0]), ns))
        qU["x"][0] = np.zeros((get_object_length(x[0]), ns))
        qU["w"][0] = np.zeros((get_object_length(M[0]["x"]), ns))
        qU["z"][0] = np.zeros((get_object_length(M[0]["v"]), ns))
        qU["Z"][0] = np.zeros((get_object_length(M[0]["v"]), ns))
        qU["z"][1] = np.zeros((get_object_length(M[1]["v"]), ns))
        qU["Z"][1] = np.zeros((get_object_length(M[1]["v"]), ns))
        qU["W"][0] = np.zeros((get_object_length(M[0]["x"]), ns))
        for t, q_temp in Q.items():
            # states and predictions
            v = unvectorise_object(q_temp["u"]["v"][0], v)
            x = unvectorise_object(q_temp["u"]["x"][0], x)
            z = unvectorise_object(q_temp["e"][0 : (ny + nv)], [m["v"] for m in M])
            Z = unvectorise_object(q_temp["E"][0 : (ny + nv)], [m["v"] for m in M])
            w = unvectorise_object(
                q_temp["e"][((ny + nv) * n) : (nx + (ny + nv) * n)], [m["x"] for m in M]
            )
            X = unvectorise_object(
                q_temp["E"][((ny + nv) * n) : (nx + (ny + nv) * n)], [m["x"] for m in M]
            )
            for i in range(nl - 1):
                if M[i]["m"]:
                    qU["v"][i + 1][:, t] = vectorise_object(v[i])
                if M[i]["n"]:
                    qU["x"][i][:, t] = vectorise_object(x[i])
                    qU["w"][i][:, t] = vectorise_object(w[i])
                if M[i]["l"]:
                    qU["z"][i][:, t] = vectorise_object(z[i])
                    qU["Z"][i][:, t] = vectorise_object(Z[i])
                if M[i]["n"]:
                    qU["W"][i][:, t] = vectorise_object(X[i])
            if M[nl - 1]["l"]:
                qU["z"][nl - 1][:, t] = vectorise_object(z[nl - 1])
                qU["Z"][nl - 1][:, t] = vectorise_object(Z[nl - 1])

            qU["v"][0][:, t] = vectorise_object(q_temp["u"]["y"][0]) - vectorise_object(
                z[0]
            )

            # and conditional covariances
            qU["S"][t] = copy.deepcopy(q_temp["u"]["s"])
            qU["C"][t] = copy.deepcopy(q_temp["u"]["c"])

            # parameters
            qP["p"][t] = vectorise_object(q_temp["p"]["p"])
            qP["c"][t] = copy.deepcopy(q_temp["p"]["c"])

            # hyperparameters
            qH["p"][t] = vectorise_object(q_temp["h"]["h"], q_temp["h"]["g"])
            qH["c"][t] = copy.deepcopy(q_temp["h"]["c"])

        if iN >= 1:
            dF = F[iN] - F[iN - 1]
        else:
            dF = 0

        str1 = f"LAP: {iN} ({iD})"
        if iN == 0:
            str2 = f"  F0:{F[iN]:.4e}"
        else:
            str2 = f"F-F0:{F[iN] - F[0]:.4e}"
        str3 = f"dF:{dF:.2e}"
        str4 = f"({time.time() - start_time:.2e} sec)"

        print(f"{str1:<16}{str2:<20}{str3:<14}{str4:<16}")

    print(f'{"LAP: Converged":<19}F:{F[-1]:.4e}')
    # Conditional moments of time-averaged parameters
    Qp = 0
    Ep = 0
    for i in range(ns):
        # weight in proportion to precisions
        P = get_inv(qP["c"][i])
        Ep += P @ qP["p"][i]
        Qp += P

    Ep = get_inv(Qp) @ Ep
    Cp = get_inv(Qp + (1 - ns) * Pp)
    qP["P"] = unvectorise_object(Up @ Ep + pp["p"], [m["pE"] for m in M])
    qP["C"] = Up @ Cp @ Up.T
    qP["V"] = unvectorise_object(np.diag(qP["C"]), [m["pE"] for m in M])
    qP["U"] = copy.deepcopy(Up)

    # conditional moments of hyper-parameters
    Qh = 0
    Eh = 0
    for i in range(ns):
        # weight in proportion to precisions
        P = get_inv(qH["c"][i])
        Eh += P @ qH["p"][i]
        Qh += P

    Eh = get_inv(Qh) @ Eh
    Ch = get_inv(Qh + (1 - ns) * Ph)
    P = unvectorise_object(Eh, [qh["h"], qh["g"]])
    qH["h"] = copy.deepcopy(P[0])
    qH["g"] = copy.deepcopy(P[1])
    qH["C"] = copy.deepcopy(Ch)
    P = unvectorise_object(np.diag(qH["C"]), P)
    qH["V"] = copy.deepcopy(P[0])
    qH["W"] = copy.deepcopy(P[1])

    # Assign output variables
    DEM["M"] = copy.deepcopy(M)  # model
    DEM["U"] = copy.deepcopy(U)  # causes

    DEM["qU"] = copy.deepcopy(qU)  # conditional moments of model-states
    DEM["qP"] = copy.deepcopy(qP)  # conditional moments of model-parameters
    DEM["qH"] = copy.deepcopy(qH)  # conditional moments of hyper-parameters

    DEM["F"] = copy.deepcopy(F[:iN])  # [-ve] Free-energy
    DEM["S"] = copy.deepcopy(S[:iN])  # [-ve] Free-action
    DEM["FC"] = copy.deepcopy(FC)  # Free-energy components
    DEM["CC"] = copy.deepcopy(CC)  # over iterations
    DEM["FT"] = copy.deepcopy(FT)  # over time
    return DEM


def estimate_dcm(dcm):
    if isinstance(dcm, dict):
        dcm_dictionary = copy.deepcopy(dcm)
    elif isinstance(dcm, str):
        data_mat = sio.loadmat(dcm, simplify_cells=True)
        data_mat = convert_sparse_to_dense(data_mat)
        dcm_dictionary = copy.deepcopy(data_mat["DCM"])

    # check options
    dcm_dictionary.setdefault("options", {}).setdefault("two_state", 0)
    dcm_dictionary.setdefault("options", {}).setdefault("stochastic", 0)
    dcm_dictionary.setdefault("options", {}).setdefault("nonlinear", 0)
    dcm_dictionary.setdefault("options", {}).setdefault("centre", 0)
    dcm_dictionary.setdefault("options", {}).setdefault(
        "hidden", np.array([], dtype=np.uint8)
    )
    dcm_dictionary.setdefault("options", {}).setdefault("hE", 6)
    dcm_dictionary.setdefault("options", {}).setdefault("hC", 1 / 128)

    if "a" in dcm_dictionary:
        dcm_dictionary.setdefault("n", len(dcm_dictionary["a"]))
    if "Y" in dcm_dictionary and "y" in dcm_dictionary["Y"]:
        dcm_dictionary.setdefault("v", len(dcm_dictionary["Y"]["y"]))
    # specify DCM models

    M = {}
    # do not show the iteration graph in python
    M["nograph"] = not dcm_dictionary["options"].get("nograph", get_default("cmdline"))
    M["noprint"] = not get_default("dcm_verbose")

    #

    # check max iterations

    if "maxit" not in dcm_dictionary["options"]:
        if dcm_dictionary["options"].get("stochastic", False):
            dcm_dictionary["options"]["maxit"] = 32
        else:
            dcm_dictionary["options"]["maxit"] = 128

    M["Nmax"] = dcm_dictionary["M"].get("Nmax", dcm_dictionary["options"]["maxit"])
    # check max nodes

    if "maxnodes" not in dcm_dictionary["options"]:
        dcm_dictionary["options"]["maxnodes"] = 8
    # analysis and options

    dcm_dictionary["options"]["induced"] = 0

    # unpack

    U = dcm_dictionary["U"]  # exogenous inputs
    Y = dcm_dictionary["Y"]  # responses
    n = dcm_dictionary["n"]  # number of regions
    v = dcm_dictionary["v"]  # number of scans

    # detrend outputs (and inputs)

    Y["y"] = detrend(Y["y"], type="constant", axis=0)
    if dcm_dictionary["options"]["centre"]:
        U["u"] = detrend(U["u"])

    # check scaling of Y (enforcing a maximum change of 4)

    scale = np.max(Y["y"]) - np.min(Y["y"])
    scale = 4 / max(scale, 4)
    Y["y"] = Y["y"] * scale
    Y["scale"] = scale
    # check confounds (add constant if necessary)

    if "X0" not in Y:
        Y["X0"] = np.ones((v, 1))
    if Y["X0"].shape[1] == 1:
        Y["X0"] = np.ones((v, 1))

    # fMRI slice time sampling

    M["delays"] = dcm_dictionary.get("delays", np.ones((n, 1)))
    M["TE"] = dcm_dictionary.get("TE")

    # create priors

    # check dcm_dictionary.d (for nonlinear DCMs)
    if "d" in dcm_dictionary and dcm_dictionary["d"].ndim > 2:
        dcm_dictionary["options"]["nonlinear"] = True
    else:
        dcm_dictionary["d"] = np.array([])
        dcm_dictionary["options"]["nonlinear"] = False

    # specify parameters for spm_int_D (ensuring updates every second or so)
    if dcm_dictionary["options"]["nonlinear"]:
        M["IS"] = "spm_int_D"
        M["nsteps"] = round(np.max(Y["dt"], 0))
        M["states"] = np.arange(0, n)
    else:
        M["IS"] = integrate_bilinear

    # check for endogenous DCMs, with no exogenous driving effects
    if (
        dcm_dictionary.get("c") is None
        or not dcm_dictionary["c"].size
        or U.get("u") is None
        or not U["u"].size
    ):
        dcm_dictionary["c"] = np.zeros((n, 1))
        dcm_dictionary["b"] = np.zeros((n, n, 1))
        U["u"] = np.zeros((v, 1))
        U["name"] = ["null"]

    if not np.any(vectorise_object(U["u"])) or not np.any(
        vectorise_object(dcm_dictionary["c"])
    ):
        dcm_dictionary["options"]["stochastic"] = 1

    # priors (and initial states)
    pE, pC, x = specify_dcm_fmri_priors(
        dcm_dictionary["a"],
        dcm_dictionary["b"],
        dcm_dictionary["c"],
        dcm_dictionary["d"],
        dcm_dictionary["options"],
    )
    prior_str = "Using specified priors "
    prior_str += "(any changes to DCM.a,b,c,d will be ignored)\n"

    try:
        M["P"] = dcm_dictionary["options"]["P"]  # initial parameters
    except KeyError:
        pass
    try:
        pE = dcm_dictionary["options"]["pE"]
        print(prior_str)  # prior expectation
    except KeyError:
        pass
    try:
        pC = dcm_dictionary["options"]["pC"]
        print(prior_str)  # prior covariance
    except KeyError:
        pass

    try:
        M["P"] = dcm_dictionary["M"]["P"]  # initial parameters
    except KeyError:
        pass
    try:
        pE = dcm_dictionary["M"]["pE"]
        print(prior_str)  # prior expectation
    except KeyError:
        pass
    try:
        pC = dcm_dictionary["M"]["pC"]
        print(prior_str)  # prior covariance
    except KeyError:
        pass

    # eigenvector constraints on pC for large models

    if n > dcm_dictionary["options"]["maxnodes"]:

        # remove confounds and find principal (nmax) modes
        # ----------------------------------------------------------------------
        y = Y["y"] - Y["X0"] @ (np.linalg.pinv(Y["X0"]) @ (Y["y"]))
        V, *_ = truncate_svd(y.T)
        V = V[:, : dcm_dictionary["options"]["maxnodes"]]

        # remove minor modes from priors on A
        # ----------------------------------------------------------------------
        j = np.arange(n * n)
        V = np.kron(V @ V.T, V @ V.T)
        pC[j, j] = V @ pC[j, j] @ V.T

    # hyperpriors over precision - expectation and covariance

    hE = np.zeros((n, 1)) + dcm_dictionary["options"]["hE"]
    hC = np.eye(n, n) * dcm_dictionary["options"]["hC"]
    if len(dcm_dictionary["options"]["hidden"]) > 0:
        i = dcm_dictionary["options"]["hidden"]
        hE[i] = -4
        hC[i, i] = np.exp(-16)

    # complete model specification

    M["f"] = define_fx_fmri  # equations of motion
    M["g"] = define_gx_fmri  # observation equation
    M["x"] = x  # initial condition (states)
    M["pE"] = pE  # prior expectation (parameters)
    M["pC"] = pC  # prior covariance  (parameters)
    M["hE"] = hE  # prior expectation (precisions)
    M["hC"] = hC  # prior covariance  (precisions)
    M["m"] = U["u"].shape[1]
    M["n"] = np.size(x)
    M["l"] = x.shape[0]
    M["N"] = 64
    M["dt"] = 32 / M["N"]
    M["ns"] = v

    # nonlinear system identification (nlsi)

    if not dcm_dictionary["options"]["stochastic"]:
        # nonlinear system identification (Variational EM) - deterministic DCM
        Ep, Cp, Eh, F, *_ = get_nlsi_GN(M, U, Y)

        # predicted responses (y) and residuals (R)
        # ----------------------------------------------------------------------
        y = M["IS"](Ep, M, U)
        R = Y["y"] - y
        R = R - Y["X0"] @ get_inv(Y["X0"].T @ Y["X0"]) @ (Y["X0"].T @ R)
        Ce = np.exp(-Eh)

    else:
        # proceed to stochastic (initialising with deterministic estimates)
        # ======================================================================

        # Decimate U.u from micro-time
        # ----------------------------------------------------------------------
        u = copy.deepcopy(U["u"])
        y = copy.deepcopy(Y["y"])
        Dy = get_dctmtx(y.shape[0], y.shape[0])
        Du = get_dctmtx(u.shape[0], y.shape[0])
        Dy = Dy * np.sqrt(y.shape[0] / u.shape[0])
        u = Dy @ (Du.T @ u)

        # DEM Structure: place model, data, input and confounds in DEM
        # ----------------------------------------------------------------------
        dem_dictionary = {
            "M": [copy.deepcopy(M), {key: np.array([]) for key in M}],
            "Y": copy.deepcopy(y.T),
            "U": copy.deepcopy(u.T),
            "X": copy.deepcopy(Y["X0"].T),
        }

        # set inversion parameters
        # ----------------------------------------------------------------------
        dem_dictionary["M"][0]["E"] = {
            "form": "Gaussian",
            "s": 1 / 2,
            "d": 2,
            "n": 4,
            "nN": dcm_dictionary["options"]["maxit"],
        }

        # adjust M.f (dem_dictionary works in time bins not seconds) and initialize M.P
        # ----------------------------------------------------------------------
        dem_dictionary["M"][0]["delays"] = M["delays"] / Y["dt"]
        dem_dictionary["M"][0]["f"] = lambda *args: Y["dt"] * define_fx_fmri(*args)

        # Specify hyper-priors on (log-precision of) observation noise
        # ----------------------------------------------------------------------
        dem_dictionary["M"][0]["Q"] = get_Ce(
            "ar", [1 for _ in range(n)]
        )  # precision components
        dem_dictionary["M"][0]["hE"] = hE  # prior expectation
        dem_dictionary["M"][0]["hC"] = hC  # prior covariance

        # allow (only) neuronal [x, s, f, q, v] hidden states to fluctuate
        # ----------------------------------------------------------------------
        W = np.ones((n, 1)) * [12, 16, 16, 16, 16]
        dem_dictionary["M"][0]["xP"] = np.exp(6)  # precision (hidden-state)
        dem_dictionary["M"][0]["W"] = np.diag(np.exp(W))  # precision (hidden-motion)
        dem_dictionary["M"][1] = {key: np.array([]) for key in dem_dictionary["M"][0]}
        dem_dictionary["M"][0]["V"] = np.array([])
        dem_dictionary["M"][1]["V"] = np.exp(16)  # precision (hidden-cause)

        # Generalised filtering (under the Laplace assumption)
        # =====================================================================
        dem_dictionary = get_LAP(dem_dictionary)

        # Save DEM estimates
        # ----------------------------------------------------------------------
        dcm_dictionary["qU"] = dem_dictionary["qU"]
        dcm_dictionary["qP"] = dem_dictionary["qP"]
        dcm_dictionary["qH"] = dem_dictionary["qH"]

        # unpack results
        # ----------------------------------------------------------------------
        F = dem_dictionary["F"][-1]
        Ep = dem_dictionary["qP"]["P"][0]
        Cp = dem_dictionary["qP"]["C"]

        # predicted responses (y) and residuals (R)
        # ----------------------------------------------------------------------
        y = dem_dictionary["qU"]["v"][0].T
        R = dem_dictionary["qU"]["z"][0].T
        R = R - Y["X0"] @ get_inv(Y["X0"].T @ Y["X0"]) @ (Y["X0"].T @ R)
        Ce = np.exp(-dem_dictionary["qH"]["h"][0])
    # Bilinear representation and first-order hemodynamic kernel
    M0, M1, L1, L2, *_ = get_bireduce(M, Ep)
    _, H1, *_ = get_kernels(M0, M1, L1, L2, M["N"], M["dt"], nargout=2)

    # and neuronal kernels
    L = np.zeros((n, len(M0)))
    L[np.arange(n), np.arange(n) + 1] = 1
    _, K1, *_ = get_kernels(M0, M1, L, M["N"], M["dt"])

    # Bayesian inference and variance {threshold: prior mean plus T = 0}
    T = vectorise_object(pE)
    sw = np.seterr(
        all="ignore"
    )  # Equivalent to MATLAB's warning('off','SPM:negativeVariance')
    Pp = unvectorise_object(
        1 - norm.cdf(T, np.abs(vectorise_object(Ep)), np.sqrt(np.diag(Cp))), Ep
    )
    Vp = unvectorise_object(np.diag(Cp).copy(), Ep)
    np.seterr(**sw)  # Restore previous warning settings

    # Remove 'nograph' field from M, if it exists
    M.pop("nograph", None)
    dcm_dictionary["M"] = M
    dcm_dictionary["Y"] = Y
    dcm_dictionary["U"] = U
    dcm_dictionary["Ce"] = Ce
    dcm_dictionary["Ep"] = Ep
    dcm_dictionary["Cp"] = Cp
    dcm_dictionary["Pp"] = Pp
    dcm_dictionary["Vp"] = Vp
    dcm_dictionary["H1"] = H1
    dcm_dictionary["K1"] = K1
    dcm_dictionary["R"] = R
    dcm_dictionary["y"] = y
    dcm_dictionary["T"] = 0
    #
    # get log-connections
    evidence = get_dcm_evidence(dcm_dictionary)
    dcm_dictionary["F"] = F
    dcm_dictionary["AIC"] = evidence["aic_overall"]
    dcm_dictionary["BIC"] = evidence["bic_overall"]
    return dcm_dictionary


def get_soreduce(M, P):
    """
    Reduction of a fully nonlinear MIMO system to second order form
    """

    # set up the f functions, in python directly assign the function name.
    M = copy.deepcopy(M)
    try:
        funx = M["f"]
    except KeyError:
        M["f"] = lambda x, u, P, M: np.zeros((0, 1))
        M["n"] = 0
        M["x"] = np.zeros((0, 0))
        funx = M["f"]

    try:
        fung = M["g"]
    except KeyError:
        # Define a default lambda function equivalent to MATLAB's inline function
        M["g"] = lambda x, u, P, M: vectorise_object(x)
        M["l"] = M["n"]
        fung = M["g"]

    # expansion point
    x = copy.deepcopy(M["x"])
    u = vectorise_object(M.get("u", np.zeros((M["m"], 1)))).reshape((3, 1))

    # Partial derivatives for 1st order Bilinear operators
    dfdxx, dfdx, f0 = get_diff(funx, x, u, P, M, np.array([1, 1]))
    dfdxu, dfdx, *_ = get_diff(funx, x, u, P, M, np.array([1, 2]))
    dfdu, _ = get_diff(funx, x, u, P, M, 2)

    m = len(dfdxu)  # m inputs
    n = len(f0)  # n states
    xx = vectorise_object(x)
    # Bilinear operators
    M0 = concatenate_dod_lol(
        [
            [np.array([[0]]), []],
            [(f0.reshape((n, 1)) - dfdx @ xx.reshape((n, 1))), dfdx],
        ]
    )
    M1 = [
        concatenate_dod_lol(
            [
                [np.array([[0]]), []],
                [(dfdu[:, [i]] - dfdxu[i] @ xx.reshape((n, 1))), dfdxu[i]],
            ]
        )
        for i in range(m)
    ]
    M2 = [
        concatenate_dod_lol(
            [
                [np.array([[0]]), []],
                [(dfdx[:, [i]] - dfdxx[i] @ xx.reshape((n, 1))), dfdxx[i]],
            ]
        )
        for i in range(n)
    ]
    # if "g" not in M:
    #     M["g"] = lambda x, u, P, M: vectorise_object(x)
    #     M["l"] = n
    # fung = M["g"]

    dgdx, g0 = get_diff(fung, x, u, P, 1)
    g0 = vectorise_object(g0)
    l = len(g0)

    L1 = concatenate_dod_lol([[(g0.reshape((l, 1)) - dgdx @ xx.reshape((n, 1))), dgdx]])

    if "l" not in M:
        return M0, M1, L1

    dgdxx, *_ = get_diff(fung, x, u, P, np.array([1, 1]), "nocat")
    L2 = {}
    for i in range(l):
        D = np.zeros((n, n))
        for j in range(n):
            D[j, :] = dgdxx[j][i, :]
        L2[i] = concatenate_dod_lol(get_diag([np.array([[0]]), D]))
    return M0, M1, M2, L1, L2


def get_int_D(P, M, U):
    """
    Integrate a MIMO bilinear system dx/dt = f(x,u) = A*x + B*x*u + Cu + D;
    Parameters:
    P   - model parameters
    M   - model structure
    U   - input structure or matrix

    Returns:
    y   - response y = g(x,u,P)
    """
    if not isinstance(U, dict):
        U = {"u": U}
    dt = U.setdefault("dt", 1)

    u = U["u"].shape[0]
    v = M.get("ns", u)

    x = np.hstack([1, vectorise_object(M["x"])])

    if "f" not in M:
        M["f"] = lambda x, u, P, M: np.zeros((0, 1))
        M["n"] = 0
        M["x"] = np.zeros((0, 0))

    try:
        g = M["g"]
    except KeyError:
        g = lambda x, u, P, M: x
        M["g"] = g

    M0, M1, M2 = get_soreduce(M, P)
    n = len(M2)
    m = len(M1)

    try:
        D = np.maximum(np.round(np.array(M["delays"]) / U["dt"]).astype(int), 0)
    except KeyError:
        D = np.ones(M["l"], dtype=int) * np.round(u / v)

    try:
        M2 = [M2[i] for i in M["states"]]
        n = len(M2)
    except KeyError:
        pass

    N = M.get("nsteps", 1)

    i = (
        list([0])
        + (np.nonzero(np.any(np.diff(U["u"], axis=0), axis=1))[0] + 1).tolist()
    )
    su = np.zeros(u, dtype=bool)
    su[i] = True

    s = np.ceil(np.arange(v) * u / v).astype(int)
    sy = np.zeros((M["l"], u))
    for j in range(M["l"]):
        sy[j, s + D[j] - 1] = np.arange(1, v + 1)

    i = np.ceil(np.arange(v * N) * u / v / N).astype(int) + D[0]
    sx = np.zeros(u)
    sx[i] = 1

    t = np.nonzero(su | np.any(sy, axis=0 | sx))[0]
    sy = sy[:, t]
    dt = np.append(np.diff(t), 0) * U["dt"]

    y = np.zeros((M["l"], v))
    J = copy.deepcopy(M0)
    uu = U["u"]
    for i, ti in enumerate(t):

        u = uu[ti, :]
        J = copy.deepcopy(M0)
        for j in range(m):
            J += u[j] * M1[j]
        for j in range(n):
            J += (x[j + 1] - M["x"][j]) * M2[j]

        if np.any(sy[:, i]):
            q = unvectorise_object(x[1:], M["x"])
            q = vectorise_object(M["g"](q, u, P))
            j = np.nonzero(sy[:, i])[0]
            s = int(sy[j[0], i]) - 1
            y[j, s] = q[j]

        x = expm(J * dt[i]).dot(x)

        if np.linalg.norm(x, 1) > 1e6:
            break

    return y.T
