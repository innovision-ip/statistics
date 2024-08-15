"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2024-07-28 21:18:57
Last Edited : 2024-08-15 12:25:23
Last Author : Jie Li
File Path   : /undefined/Users/Jie/Documents/dcm_IP/dcm/utils_erp.py
Description :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

import numpy as np
from scipy.linalg import expm, solve

from utils import *


def set_erp_priors(A=None, B=None, C=None):
    """
    Prior moments for a neural-mass model of ERPs.

    Parameters
    ----------
    A : list of arrays, optional
        Binary constraints on extrinsic connections (default is None).
    B : list of arrays, optional
        Binary constraints on extrinsic connections (default is None).
    C : array, optional
        Binary constraints on extrinsic connections (default is None).

    Returns
    -------
    pE : dict
        Prior expectation - f(x,u,P,M)

        Synaptic parameters
        -------------------
        T : array
            Synaptic time constants.
        G : array
            Synaptic densities (intrinsic gain).
        S : array
            Activation function parameters.
        G : array
            Intrinsic connection strengths.

        Connectivity parameters
        -----------------------
        A : list of arrays
            Extrinsic connectivity.
        B : list of arrays
            Trial-dependent connectivity.
        C : array
            Stimulus input connectivity.
        D : array
            Delays.

        Stimulus and noise parameters
        -----------------------------
        R : array
            Onset and dispersion.

    pC : dict
        Prior (co)variances.

    Notes
    -----
    Because priors are specified under log normal assumptions, most
    parameters are simply scaling coefficients with a prior expectation
    and variance of one. After log transform this renders pE = 0 and
    pC = 1. The prior expectations of what they scale are specified in
    define_erp_fx.

    References
    ----------
    David O, Friston KJ (2003) A neural mass model for MEG/EEG: coupling and
    neuronal dynamics. NeuroImage 20: 1743-1755.

    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """
    # Default: a single source model
    if A is None:
        A = [np.array([0, 0, 0])]
    if B is None:
        B = []
    if C is None:
        C = np.array([1])

    # -log of absent (null) connections
    N = 4

    # Number of sources and inputs
    n = C.shape[0]
    u = C.shape[1] if C.ndim > 1 else 1

    # Parameters for neural-mass forward model
    E = {}  # expectation
    V = {}  # standard deviation

    # Set intrinsic [excitatory] time constants and gain
    E["T"] = np.zeros((n, 2))
    V["T"] = np.ones((n, 2)) / 16

    E["G"] = np.zeros((n, 2))
    V["G"] = np.ones((n, 2)) / 16

    # Set parameter of activation function
    E["S"] = np.array([0, 0])
    V["S"] = np.array([1, 1]) / 16

    # Set extrinsic connectivity
    Q = np.zeros((n, n))
    E["A"] = []
    V["A"] = []
    for a in A:
        a = np.array(a, dtype=bool)
        E["A"].append(a * N - N)
        V["A"].append(a / 16)
        Q = np.logical_or(Q, a)

    E["B"] = np.zeros((n, n))
    V["B"] = np.zeros((n, n))
    for i, b in enumerate(B):
        b = np.array(b, dtype=bool)
        E["B"][i, :] = np.zeros_like(b)
        V["B"][i, :] = b / 8
        Q = np.logical_or(Q, b)

    C = np.array(C, dtype=bool)
    E["C"] = C * N - N
    V["C"] = C / 32

    # Set intrinsic connectivity
    E["H"] = np.zeros(4)
    V["H"] = np.ones(4) / 16

    # Set (extrinsic) delay
    E["D"] = np.zeros((n, n))
    V["D"] = Q / 16

    # fixed intrinsic delays
    np.fill_diagonal(V["D"], 0)

    # Set stimulus parameters: onset, dispersion and sustained proportion
    E["R"] = np.zeros((u, 2))
    V["R"] = np.ones((u, 2)) / 16

    return E, V


def set_dcm_neural_priors(A, B, C, model):
    model = model.lower()

    if model in {"erp", "sep"}:
        # Prior moments on parameters
        pE, pC = set_erp_priors(A, B, C)
        return pE, pC
    else:
        raise ValueError(f"Unknown model type: {model}")


def set_zeros(x):
    x = x.copy()
    x_len = len(vectorise_object(x))
    x = unvectorise_object(np.zeros((x_len, 1)), x)
    return x


def set_L_priors(dipfit, pE=None, pC=None):
    """
    Prior moments for the lead-field parameters of ERP models.

    Parameters
    ----------
    dipfit : dict
        Forward model structure:

        - dipfit['type'] : str
            'ECD', 'LFP' or 'IMG'
        - dipfit['symmetry'] : float
            Distance (mm) for symmetry constraints (ECD)
        - dipfit['location'] : bool
            Allow changes in source location (ECD)
        - dipfit['Lpos'] : list of list of float
            x, y, z source positions (mm) (ECD)
        - dipfit['Nm'] : int
            Number of modes (IMG)
        - dipfit['Ns'] : int
            Number of sources
        - dipfit['Nc'] : int
            Number of channels

    pE : dict, optional
        Prior expectation.

    pC : dict, optional
        Prior covariance.

    Returns
    -------
    pE : dict
        Updated prior expectation with spatial parameters:

        - pE['Lpos'] : list of list of float
            Position (ECD)
        - pE['L'] : list of list of float
            Orientation (ECD), coefficients of local modes (Imaging), gain of electrodes (LFP)
        - pE['J'] : list
            Contributing states (length(J) = number of states per source)

    pC : dict
        Updated prior covariance.

    References
    ----------
    David O, Friston KJ (2003) A neural mass model for MEG/EEG: coupling and
    neuronal dynamics. NeuroImage 20: 1743-1755

    Notes
    -----
    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """

    # defaults
    # --------------------------------------------------------------------------
    model = dipfit.get("model", "LFP")
    type1 = dipfit.get("type", "LFP")
    location = dipfit.get("location", 0)
    if pC is None:
        pC = {}
    if pE is None:
        pE = {}

    # number of sources
    # --------------------------------------------------------------------------
    try:
        n = dipfit["Ns"]
        m = dipfit["Nc"]
    except KeyError:
        n = dipfit
        m = n

    # location priors (4 mm)
    # --------------------------------------------------------------------------
    V = 2**2 if location else 0

    if type1 == "ECD":
        # Mean and variance
        # ----------------------------------------------------------------------
        pE["Lpos"] = dipfit["Lpos"]
        pC["Lpos"] = [[V] * n for _ in range(3)]
        pE["L"] = [[0] * n for _ in range(3)]
        pC["L"] = [[64] * n for _ in range(3)]

        Sc = [i for i, x in enumerate(dipfit["silent_source"]) if x.size > 0]

        # Silence sources for CSD
        if Sc:
            pC["L"][:, Sc] = pC["L"][:, Sc] * 0

    elif type1 == "IMG":
        # ----------------------------------------------------------------------
        m = dipfit["Nm"]
        pE["Lpos"] = np.zeros((3, 0))
        pC["Lpos"] = np.zeros((3, 0))
        pE["L"] = np.zeros((m, n))
        pC["L"] = np.ones((m, n)) * 64

        # Find indices of non-empty elements in dipfit['silent_source']
        Sc = [i for i, x in enumerate(dipfit["silent_source"]) if x.size > 0]

        # Silence sources for CSD
        if Sc:
            pC["L"][:, Sc] = pC["L"][:, Sc] * 0

    elif type1 == "LFP":
        # ----------------------------------------------------------------------
        pE["Lpos"] = []
        pC["Lpos"] = []
        pE["L"] = [1] * m
        pC["L"] = [64] * m

    else:
        print("Unknown spatial model")

    # Contributing states (encoded in J)
    # ==========================================================================
    # Check if model is a string, if so, convert it to a dictionary
    if isinstance(model, str):
        mod = {"source": model}
        model = [mod]

    pE["J"] = []
    pC["J"] = []

    for i, mod in enumerate(model):
        source = mod["source"].upper()

        if source in {"ERP", "SEP"}:
            # 9 states
            pE["J"].append(np.zeros(9))
            pE["J"][-1][8] = 1
            pC["J"].append(np.zeros(9))
            pC["J"][-1][0] = 1 / 32
            pC["J"][-1][6] = 1 / 32
        elif source in {"CMC", "TFM"}:
            # 8 states
            pE["J"].append(np.zeros(8))
            pE["J"][-1][2] = 1
            pC["J"].append(np.zeros(8))
            pC["J"][-1][0] = 1 / 32
            pC["J"][-1][6] = 1 / 32

        # Cardinal sources
        if "J" in mod:
            if len(mod["J"]) > 0:
                pE["J"][i] = set_zeros(pE["J"][i])
                pE["J"][i][mod["J"]] = 1

        # Subsidiary (free) sources
        if "K" in mod:
            if len(mod["K"]) > 0:
                pC["J"][i] = set_zeros(pE["J"][i])
                pC["J"][i][mod["K"]] = 1 / 32
    # Replace J with a vector if there is only one sort of model
    if len(pE["J"]) == 1:
        pE["J"] = pE["J"][0]
        pC["J"] = pC["J"][0]
    return pE, pC


def get_erp_L(P, dipfit):
    """
    Construct the lead field (L) as a function of position and moments.

    Parameters
    ----------
    P : dict
        Model parameters.
    dipfit : dict
        Spatial model specification.

    Returns
    -------
    L : ndarray
        Lead field.

    Notes
    -----
    The lead field (L) is constructed using the specific parameters in P and,
    where necessary, information in the dipole structure dipfit. For ECD
    models, P['Lpos'] and P['L'] encode the position and moments of the ECD.
    The field dipfit['type']:

        'ECD', 'LFP' or 'IMG'

    determines whether the model is ECD or not. For imaging reconstructions,
    the parameters P['L'] are a (m x n) matrix of coefficients that scale the
    contribution of n sources to m = dipfit['Nm'] modes encoded in dipfit['G'].

    For LFP models (the default), P['L'] simply encodes the electrode gain for
    each source contributing an LFP.

    See Also
    --------
    Kiebel et al. (2006) NeuroImage

    References
    ----------
    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """
    # Create a persistent variable that remembers the last locations
    # Initialize persistent variables if they do not exist
    if not hasattr(get_erp_L, "LastLpos"):
        get_erp_L.LastLpos = np.full(P["L"].shape, np.nan)
        get_erp_L.LastL = np.full((dipfit["G"][0].shape[0], P["L"].shape[1]), np.nan)

    LastLpos = get_erp_L.LastLpos
    LastL = get_erp_L.LastL
    # global LastLpos
    # global LastL
    P = copy.deepcopy(P)
    dipfit = copy.deepcopy(dipfit)

    # Type of spatial model and modality
    type_ = dipfit.get("type", "LFP")

    if type_ == "IMG":
        # Number of sources - n
        n = P["L"].shape[1]

        # Re-compute lead field only if any coefficients have changed
        try:
            Id = np.where(np.any(LastLpos != P["L"], axis=0))[0]
        except Exception:
            Id = np.arange(n)
        for i in Id:
            LastL[:, i] = dipfit["G"][i] @ P["L"][:, i]

        # Record new spatial parameters
        L = LastL.copy()

    elif type_ == "LFP":
        m = len(P["L"])
        n = dipfit.get("Ns", m)
        L = np.zeros((m, n))
        np.fill_diagonal(L, P["L"])

        # Assume common sources contribute to the last channel
        if "common_source" in dipfit:
            L[m - 1, m - 1 : n] = L[m - 1, m - 1]

    else:
        raise ValueError("Unknown spatial model")

    get_erp_L.LastLpos = P["L"]
    get_erp_L.LastL = LastL
    return L


def get_dcm_eeg_channelmodes(dipfit, Nm=8, xY=None):
    """
    Return the channel eigenmodes.

    Parameters
    ----------
    dipfit : dict
        Spatial model specification.
    Nm : int
        Number of modes required (upper bound).
    xY : dict, optional
        Data structure.

    Returns
    -------
    U : ndarray
        Channel eigenmodes.

    Notes
    -----
    Uses SVD (an eigensolution) to identify the patterns with the greatest
    prior covariance, assuming independent source activity in the specified
    spatial (forward) model.

    If `xY` is specified, a CVA (a generalised eigensolution) will be used to
    find the spatial modes that are best by the spatial model.

    `U` is scaled to ensure `trace(U.T @ L @ L.T @ U) = Nm`.

    References
    ----------
    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """

    # Spatial modes
    pE, pC = set_L_priors(dipfit)

    # Evaluate eigenmodes of gain of covariance in sensor space
    dGdg, _ = get_diff(get_erp_L, pE, dipfit, 1, "nocat")
    L = concatenate_dod_lol([[dGdg[i] for i in np.where(vectorise_object(pC))[0]]])

    # Eigen-mode reduction
    U, S, _ = truncate_svd(L @ L.T, np.exp(-8))

    if xY is not None:
        print("Need to rewrite this part!")
        n = min(U.shape[1], 32)
        L = U[:, :n] @ np.diag(np.sqrt(S[:n]))

        # Response variable
        Y = [np.array(y).T for y in xY["y"]]
        Y = concatenate_dod_lol(Y)
        S = U.T @ (L @ L.T) @ U
        S = np.diag(S)

    # Eigen-mode reduction
    try:
        U = U[:, :Nm]
        S = S[:Nm]
    except Exception:
        pass

    # Re-scale spatial projector
    U = U / np.sqrt(np.mean(S))
    return U


def define_fy_erp(y, M):
    """
    Feature selection for ERP models.

    Parameters
    ----------
    y : ndarray or list of ndarray
        Input data.
    M : dict
        Model parameters. Should contain the key 'U' for spatial projection.

    Returns
    -------
    f : ndarray or list of ndarray
        Projected features.

    Notes
    -----
    If `y` is a numeric array, it is projected using `M['U']`.
    If `y` is a list of numeric arrays, each element is projected recursively.
    """
    # Ensure 'U' is in M, default to 1 if not present
    U = M.get("U", 1)

    # Spatial projection
    if isinstance(y, (list, tuple)):
        f = [define_fy_erp(yi, M) for yi in y]
    else:
        f = y @ U

    return f


def get_dcm_x_neural(P, model):
    """
    Return the state and equation of neural mass models.

    Parameters
    ----------
    P : dict
        Parameter structure.
    model : str or dict
        Model type ('ERP', 'SEP', 'CMC', 'LFP', 'CMM', 'NNM', 'MFM', 'CMM NMDA')
        or a dictionary representing a model structure.

    Returns
    -------
    x : ndarray or list of ndarray
        Initial states.
    f : str
        State equation dxdt = f(x,u,P,M) - synaptic activity.
    h : str or None
        State equation dPdt = f(x,u,P,M) - synaptic plasticity.
    """
    # Parametric state equation
    h = None
    P = copy.deepcopy(P)
    model = copy.deepcopy(model)
    # Assemble initial states for generic models
    if isinstance(model, dict):
        P["A"][0] = 1
        x = [get_dcm_x_neural(P, m["source"]) for m in model]
        f = define_fx_erp
        return x, f, h

    # Initial state and equation
    model = model.lower()
    if model == "erp":
        # Initial states and equations of motion
        n = len(P["A"][0])
        m = 9
        x = np.zeros((n, m))
        f = define_fx_erp
    elif model == "sep":
        # Initial states
        n = len(P["A"][0])
        m = 9
        x = np.zeros((n, m))
        f = define_fx_sep
    else:
        raise ValueError("Unknown model")

    return x, f, h


def define_fx_erp(x, u, P, M, Jacobian=False):
    """
    State equations for a neural mass model of ERPs.

    Parameters
    ----------
    x : array_like
        State vector
        x[:,0] : voltage (spiny stellate cells)
        x[:,1] : voltage (pyramidal cells) +ve
        x[:,2] : voltage (pyramidal cells) -ve
        x[:,3] : current (spiny stellate cells) depolarizing
        x[:,4] : current (pyramidal cells) depolarizing
        x[:,5] : current (pyramidal cells) hyperpolarizing
        x[:,6] : voltage (inhibitory interneurons)
        x[:,7] : current (inhibitory interneurons) depolarizing
        x[:,8] : voltage (pyramidal cells)
    u : array_like
        Input vector
    P : dict
        Parameters
    M : dict
        Model structure

    Returns
    -------
    f : array_like
        dx(t)/dt = f(x(t))
    J : array_like, optional
        df(t)/dx(t)
    D : array_like, optional
        Delay operator dx(t)/dt = f(x(t - d)) = D(d)*f(x(t))

    Notes
    -----
    Prior fixed parameter scaling [Defaults]

    M.pF.E = [32, 16, 4]           # extrinsic rates (forward, backward, lateral)
    M.pF.H = [1, 4/5, 1/4, 1/4]*128 # intrinsic rates (g1, g2, g3, g4)
    M.pF.D = [2, 16]               # propagation delays (intrinsic, extrinsic)
    M.pF.G = [4, 32]               # receptor densities (excitatory, inhibitory)
    M.pF.T = [8, 16]               # synaptic constants (excitatory, inhibitory)
    M.pF.S = [1, 1/2]              # parameters of activation function

    References
    ----------
    David O, Friston KJ (2003) A neural mass model for MEG/EEG: coupling and
    neuronal dynamics. NeuroImage 20: 1743-1755

    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """
    # Get dimensions and configure state variables
    x = unvectorise_object(x, M["x"])  # neuronal states
    n = x.shape[0]  # number of sources

    # Default fixed parameters
    E = np.array([1, 1 / 2, 1 / 8]) * 32.0
    G = np.array([1, 4 / 5, 1 / 4, 1 / 4]) * 128.0
    D = np.array([2, 16], dtype=np.float64)
    H = np.array([4, 32], dtype=np.float64)
    T = np.array([8, 16], dtype=np.float64)
    R = np.array([2, 1], dtype=np.float64) / 3

    # Specified fixed parameters
    if "pF" in M:
        if "E" in M["pF"]:
            E = M["pF"]["E"]
        if "G" in M["pF"]:
            G = M["pF"]["G"]
        if "T" in M["pF"]:
            T = M["pF"]["T"]
        if "R" in M["pF"]:
            R = M["pF"]["R"]
        if "H" in M["pF"]:
            H = M["pF"]["H"]
        if "D" in M["pF"]:
            D = M["pF"]["D"]

    # Test for free parameters on intrinsic connections
    try:
        G = G * np.exp(P["H"])
    except KeyError:
        pass
    G = np.ones((n, 1)) @ G.reshape((1, -1))

    # Exponential transform to ensure positivity constraints
    if n > 1:
        A = [
            np.exp(P["A"][0]) * E[0],
            np.exp(P["A"][1]) * E[1],
            np.exp(P["A"][2]) * E[2],
        ]
    else:
        A = [0, 0, 0]

    C = np.exp(P["C"])

    # Intrinsic connectivity and parameters
    Te = T[0] / 1000.0 * np.exp(P["T"][:, 0])
    Ti = T[1] / 1000.0 * np.exp(P["T"][:, 1])
    He = H[0] * np.exp(P["G"][:, 0])
    Hi = H[1] * np.exp(P["G"][:, 1])

    # Pre-synaptic inputs: s(V)
    R = R * np.exp(P["S"])
    S = 1.0 / (1 + np.exp(-R[0] * (x - R[1]))) - 1.0 / (1 + np.exp(R[0] * R[1]))

    # Input
    if "u" in M:
        # Endogenous input
        U = u.flatten() * 64
    else:
        # Exogenous input
        U = C * u.flatten() * 2.0

    # State: f(x)
    f = np.zeros((n, 9))
    f[:, 6] = x[:, 7]
    f[:, 7] = (
        He * ((A[1] + A[2]) @ S[:, 8] + G[:, 2] * S[:, 8]) - 2 * x[:, 7] - x[:, 6] / Te
    ) / Te
    # Granular layer (spiny stellate cells): Voltage & depolarizing current
    f[:, 0] = x[:, 3]
    f[:, 3] = (
        He * ((A[0] + A[2]) @ S[:, 8] + G[:, 0] * S[:, 8] + U)
        - 2 * x[:, 3]
        - x[:, 0] / Te
    ) / Te
    # Infra-granular layer (pyramidal cells): depolarizing current
    f[:, 1] = x[:, 4]
    f[:, 4] = (
        He * ((A[1] + A[2]) @ S[:, 8] + G[:, 1] * S[:, 0]) - 2 * x[:, 4] - x[:, 1] / Te
    ) / Te
    # Infra-granular layer (pyramidal cells): hyperpolarizing current
    f[:, 2] = x[:, 5]
    f[:, 5] = (Hi * G[:, 3] * S[:, 6] - 2 * x[:, 5] - x[:, 2] / Ti) / Ti
    # Infra-granular layer (pyramidal cells): Voltage
    f[:, 8] = x[:, 4] - x[:, 5]
    f = vectorise_object(f)
    if not Jacobian:
        return f
    else:
        # Jacobian
        J, _ = get_diff(M["f"], x, u, P, M, 1)

        # Delays
        De = D[1] * np.exp(P["D"]) / 1000.0
        Di = D[0] / 1000.0
        De = (1 - np.eye(n)) * De
        Di = (1 - np.eye(9)) * Di
        De = np.kron(np.ones((9, 9)), De)
        Di = np.kron(Di, np.eye(n))
        D = Di + De

        I_ = np.eye(len(J))
        Q = solve(I_ + D * J, I_)

        return f, J, Q


def get_int_L(P, M, U, N=1, Jacobian=True):
    """
    Integrate a MIMO nonlinear system using a fixed Jacobian: J(x(0))

    Parameters
    ----------
    P : array-like
        Model parameters
    M : dict
        Model structure
    U : array-like
        Input structure or matrix
    N : int, optional
        Number of local linear iterations per time step, by default 1
    Jacobian : logical, optional
        for using Jacobian matrix, by default True

    Returns
    -------
    y : array-like
        (v x l) response y = g(x,u,P)

    Notes
    -----
    Integrates the MIMO system described by

        dx/dt = f(x,u,P,M)
        y     = g(x,u,P,M)

    using the update scheme:

        x(t + dt) = x(t) + U*dx(t)/dt

                U = (expm(dt*J) - I)*inv(J)
                J = df/dx

    at input times. This integration scheme evaluates the update matrix (U)
    at the expansion point.

    SPM solvers or integrators:

    - spm_int_ode: uses ode45 (or ode113) which are one and multi-step solvers
    respectively. They can be used for any ODEs, where the Jacobian is
    unknown or difficult to compute; however, they may be slow.
    - spm_int_J: uses an explicit Jacobian-based update scheme that preserves
    nonlinearities in the ODE: dx = (expm(dt*J) - I)*inv(J)*f. If the
    equations of motion return J = df/dx, it will be used; otherwise it is
    evaluated numerically, using spm_diff at each time point. This scheme is
    infallible but potentially slow, if the Jacobian is not available (calls
    spm_dx).
    - spm_int_E: As for spm_int_J but uses the eigensystem of J(x(0)) to eschew
    matrix exponentials and inversion during the integration. It is probably
    the best compromise, if the Jacobian is not available explicitly.
    - spm_int_B: As for spm_int_J but uses a first-order approximation to J
    based on J(x(t)) = J(x(0)) + dJdx*x(t).
    - spm_int_L: As for spm_int_B but uses J(x(0)).
    - spm_int_U: like spm_int_J but only evaluates J when the input changes.
    This can be useful if input changes are sparse (e.g., boxcar functions).
    It is used primarily for integrating EEG models
    - spm_int: Fast integrator that uses a bilinear approximation to the
    Jacobian evaluated using spm_bireduce. This routine will also allow for
    sparse sampling of the solution and delays in observing outputs. It is
    used primarily for integrating fMRI models

    References
    ----------
    Karl Friston
    Copyright (C) 2008-2022 Wellcome Centre for Human Neuroimaging
    """
    # Convert U to U.u if necessary
    U = copy.deepcopy(U)
    if not isinstance(U, dict):
        U = {"u": U}
    dt = U.get("dt", 1)

    # Initial states and inputs
    x = M.get("x", np.zeros((0, 1)))
    M["x"] = x

    u = U.get("u", np.zeros((1, M["m"])))[0]

    # Add [0] states if not specified
    f = M.get("f", lambda x, u, P, M: np.zeros((0, 1)))

    # Output nonlinearity, if specified
    g = M.get("g", lambda x, u, P, M: x)
    if g is None:
        g = lambda x, u, P, M: x

    M["g"] = g

    # dx(t)/dt and Jacobian df/dx (and check for delay operator)
    n = get_object_length(x)
    if Jacobian is False:
        _, dfdx = f(x, u, P, M)
        dfdx = dfdx - np.eye(n) * np.exp(-16)
        Q = solve(dfdx.T, (expm(dt * dfdx / N) - np.eye(n)).T).T
    else:
        _, dfdx, D = f(x, u, P, M, Jacobian=Jacobian)
        dfdx = dfdx - np.eye(n) * np.exp(-16)
        Q = solve(dfdx.T, (expm(dt * D @ dfdx / N) - np.eye(n)).T).T

    # Integrate
    v = vectorise_object(x)
    num_u = U["u"].shape[0]
    y = np.zeros((n, num_u))
    for i in range(num_u):
        # Input
        u = U["u"][i, :]

        try:
            for j in range(N):
                v = v + Q @ f(v, u, P, M)

            # Output - implement g(x)
            y[:, i] = g(v, u, P, M)

        except Exception:
            for j in range(N):
                x = vectorise_object(x) + Q @ vectorise_object(f(x, u, P, M))
                x = unvectorise_object(x, M["x"])

            # Output - implement g(x)
            y[:, i] = vectorise_object(g(x, u, P, M))

    # Transpose
    y = np.real(y.T)
    return y


def get_gen_Q(P, X):
    """
    Helper routine for get_gen routines

    Parameters
    ----------
    P : dict
        Parameters
    X : array-like
        Vector of between trial effects

    Returns
    -------
    Q : dict
        Trial or condition-specific parameters

    This routine computes the parameters of a DCM for a given trial, where
    trial-specific effects are deployed according to a design vector X. The
    parameterisation follows a standard naming protocol where, for example,
    X[0]*P['B'][0] + X[1]*P['B'][1]... adjusts P['A'] for all (input) effects encoded
    in P['B']. P['BN'] and P['AN'] operate at NMDA receptors along extrinsic connections.
    """
    P = copy.deepcopy(P)
    X = copy.deepcopy(X)
    # Condition or trial specific parameters
    if "B" in P:
        Q = {k: v for k, v in P.items() if k != "B"}
    else:
        Q = copy.deepcopy(P)

    # Trial-specific effects on C (first effect only)
    try:
        Q["C"] = Q["C"][:, :, 0] + X[0] * P["C"][:, :, 1]
    except IndexError:
        pass
    if np.isscalar(X):
        X = np.array([X])
    else:
        X = np.asarray(X)

    if not isinstance(P["B"], (dict, list)):
        P["B"] = [P["B"]]
    for i, x in enumerate(X):
        # Extrinsic (driving) connections
        for j, a in enumerate(Q["A"]):
            Q["A"][j] = a + x * P["B"][i]

            # CMM-NMDA specific modulation on extrinsic NMDA connections
            if "AN" in P:
                Q["AN"][j] = Q["AN"][j] + x * P["BN"][i]

        # Modulatory connections
        if "M" in P:
            Q["M"] = Q["M"] + x * P["N"][i]

        # Intrinsic connections
        if "G" in Q:
            Q["G"][:, 0] = Q["G"][:, 0] + x * np.diag(P["B"][i])

        # Intrinsic connections
        if "int" in Q:
            for j, q_int in enumerate(Q["int"]):
                if "B" in q_int and q_int["B"] is not None:
                    Q["int"][j]["G"] = q_int["G"] + x * q_int["B"]
                else:
                    Q["int"][j]["G"][:, 1] = q_int["G"][:, 1] + x * P["B"][i][j, j]

    return Q


def get_erp_u(t, P, M):
    """
    Input for EEG models (Gaussian function)

    Parameters:
    t (array-like): PST (seconds)
    P (dict): Parameter structure
        P['R'] (array-like): Scaling of [Gaussian] parameters
    M (dict): Model structure
        M['dur'] (array-like): Durations
        M['ons'] (array-like): Onsets
        M['sus'] (array-like): Sustained input (0,1)

    Returns:
    u (ndarray): Stimulus-related (subcortical) input
    """
    M = copy.deepcopy(M)
    # Preliminaries - check durations (ms)
    if np.isscalar(M["dur"]):
        M["dur"] = [M["dur"]]
    if np.isscalar(M["ons"]):
        M["ons"] = [M["ons"]]

    try:
        if len(M["dur"]) != len(M["ons"]):
            M["dur"] = [M["dur"][0]] * len(M["ons"])
    except KeyError:
        M["dur"] = [32] * len(M["ons"])

    # Check sustained input (0,1)
    try:
        if len(M["sus"]) != len(M["ons"]):
            M["sus"] = [M["sus"][0]] * len(M["ons"])
    except KeyError:
        M["sus"] = [0] * len(M["ons"])

    # Stimulus - Gaussian (subcortical) impulse
    nu = len(M["ons"])
    u = np.zeros((len(t), nu))
    t = np.array(t) * 1000
    for i in range(nu):

        # Gaussian bump function
        delay = M["ons"][i] + 128 * P["R"][i, 0]
        scale = M["dur"][i] * np.exp(P["R"][i, 1])
        U = np.exp(-((t - delay) ** 2) / (2 * scale**2))

        # Sustained inputs
        try:
            prop = M["sus"][i] * np.exp(P["R"][i, 2])
        except IndexError:
            prop = M["sus"][i]
        U = prop * np.cumsum(U) / np.sum(U) + U * (1 - prop)
        u[:, i] = 32 * U

    return u


def get_gen_erp(P, M, U, pst=False):
    """
    Generate a prediction of trial-specific source activity.

    Parameters
    ----------
    P : dict
        Parameters.
    M : dict
        Neural-mass model structure.
    U : dict
        Trial-effects.
        U['X'] : array-like
            Between-trial effects (encodes the number of trials).
        U['dt'] : float
            Time bins for within-trial effects.

    Returns
    -------
    y : list of ndarray
        Predictions for nx states {trials} for ns samples.
    pst : ndarray
        Peristimulus time (seconds).
    """

    # Default inputs - one trial (no between-trial effects)
    U = copy.deepcopy(U)
    if "X" not in U:
        U["X"] = np.zeros((1, 0))

    # Check input u = f(t,P,M) and switch off full delay operator
    if "fu" not in M:
        M["fu"] = get_erp_u
    if "ns" not in M:
        M["ns"] = 128
    if "N" not in M:
        M["N"] = 0.0
    if "dt" not in U:
        U["dt"] = 0.004

    # Within-trial (exogenous) inputs
    if "u" not in U:
        U["u"] = get_erp_u(np.arange(1, M["ns"] + 1) * U["dt"], P, M)

    if "u" in M:
        M.pop("u")

    # Between-trial (experimental) inputs
    if "X" in U:
        X = U["X"].reshape((-1, 1))
    else:
        X = np.zeros((1, 0))

    if X.shape[0] == 0:
        X = np.zeros((1, 0))

    # Cycle over trials
    y = []
    for c in range(X.shape[0]):

        # Condition-specific parameters
        Q = get_gen_Q(P, X[c, :])

        # Integrate DCM - for this condition
        y.append(get_int_L(Q, M, U))
    # Peristimulus time
    if pst:
        pst = (np.arange(1, M["ns"] + 1) * U["dt"]) - (M["ons"] / 1000)
        return np.concatenate(y, axis=0), pst
    return np.concatenate(y, axis=0)


def get_lx_erp(P, dipfit):
    """
    Observer matrix for a neural mass model: y = G*x

    Parameters
    ----------
    P : dict
        Parameters, which may include an explicit gain matrix 'LG' or source contributions 'J'.
    dipfit : dict
        Spatial model specification.

    Returns
    -------
    L : ndarray
        Lead field matrix where y = L*x; G = dy/dx.
    """
    if "LG" in P:
        return P["LG"]

    # Extract dipfit from model if necessary
    if "dipfit" in dipfit:
        dipfit = dipfit["dipfit"]
    if "type" not in dipfit:
        dipfit = "LFP"

    # Parameterised lead field times source contribution to ECD
    L = get_erp_L(P, dipfit)

    if isinstance(P["J"], np.ndarray):
        L = np.kron(P["J"], L)
    else:
        # Construct lead field for each source
        G = []
        for i in range(len(P["J"])):
            G.append(L[:, i] * P["J"][i])
        L = concatenate_dod_lol(G)

    return L


def get_expm(J, x=None):
    """
    Approximate matrix exponential using a Taylor expansion.

    Parameters:
    J (numpy.ndarray): The input matrix.
    x (numpy.ndarray, optional): The vector to multiply with the matrix exponential.

    Returns:
    numpy.ndarray: The result of expm(J) * x if x is provided, otherwise expm(J).
    """
    J = copy.deepcopy(J)
    I = np.eye(J.shape[0])
    _, e = np.frexp(np.linalg.norm(J, ord=np.inf))
    s = max(0, e + 1)
    J = J / (2**s)
    X = copy.deepcopy(J)
    c = 1 / 2
    E = I + c * J
    D = I - c * J
    q = 6
    p = 1

    for k in range(2, q + 1):
        c = c * (q - k + 1) / (k * (2 * q - k + 1))
        X = J @ X
        cX = c * X
        E = E + cX
        if p:
            D = D + cX
        else:
            D = D - cX
        p = not p

    E = solve(D, E)

    # Undo scaling by repeated squaring E = E^(2^s)
    for _ in range(s):
        E = E @ E

    # Multiply by x if necessary
    if x is not None:
        if x.isscalar():
            return E * x
        else:
            return E @ x
    else:
        return E


def set_dtype_to_float64(data):
    """
    Recursively set all NumPy arrays' data types to float64 in a nested structure.
    Leave scalars that are integers unchanged.

    Parameters:
    data (dict or list): The input nested dictionary or list.

    Returns:
    dict or list: The modified structure with updated data types.
    """
    data = copy.deepcopy(data)
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = set_dtype_to_float64(value)
    elif isinstance(data, list):
        for i in range(len(data)):
            data[i] = set_dtype_to_float64(data[i])
    elif isinstance(data, np.ndarray):
        data = data.astype(np.float64)
    elif isinstance(data, (int, float, str)):
        pass
    else:
        pass
    return data


def get_dcm_neural_x(Q, M):
    return M["x"]


def get_nlsi_N(M, U, Y):
    """
    Bayesian inversion of a linear-nonlinear model of the form F(p)*G(g)'

    Parameters
    ----------
    M : dict
        Generative model parameters:
        - IS : function
            A prediction generating function name; usually an integration scheme for state-space models of the form
            - f : function
                State equation: dxdt = f(x, u, p, M) that returns hidden states - x; however, it can be any nonlinear function of the inputs u. I.e., x = IS(p, M, U)
        - G : function
            Linear observer: y = (x - M.x')*G(g, M)'
        - FS : function, optional
            Feature selection function name f(y, M). This function performs feature selection assuming the generalized model y = FS(y, M) = FS(x*G', M) + X0*P0 + e
        - x : array-like
            The expansion point for the states (i.e., the fixed point)
        - P : array-like, optional
            Starting estimates for model parameters [states]
        - Q : array-like, optional
            Starting estimates for model parameters [observer]
        - pE : array-like
            Prior expectation of model parameters - f(x, u, p, M)
        - pC : array-like
            Prior covariance of model parameters - f(x, u, p, M)
        - gE : array-like
            Prior expectation of model parameters - G(g, M)
        - gC : array-like
            Prior covariance of model parameters - G(g, M)
        - hE : array-like
            Prior expectation of log-precision parameters
        - hC : array-like
            Prior covariance of log-precision parameters

    U : dict
        Inputs:
        - u : array-like
            Inputs
        - dt : float
            Sampling interval

    Y : dict
        Outputs:
        - y : list of array-like
            [ns] samples x [nx] channels x {trials}
        - X0 : array-like
            Confounds or null space
        - dt : float
            Sampling interval for outputs
        - Q : array-like
            Error precision components

    Returns
    -------
    Ep : array-like
        (p x 1) Conditional expectation E{p|y}
    Cp : array-like
        (p x p) Conditional covariance Cov{p|y}
    Eg : array-like
        (p x 1) Conditional expectation E{g|y}
    Cg : array-like
        (p x p) Conditional covariance Cov{g|y}
    S : array-like
        (v x v) [Re]ML estimate of error Cov{e(h)}
    F : float
        [-ve] free energy F = log evidence = p(y|m)
    L : list of float
        Log evidence components:
        - L(1) : float
            Accuracy of states
        - L(2) : float
            Accuracy of parameters (f)
        - L(3) : float
            Accuracy of parameters (g)
        - L(4) : float
            Accuracy of parameters (u)
        - L(5) : float
            Accuracy of precisions (u)
        - L(6) : float
            Constant
        - L(7) : float
            Precision
        - L(8) : float
            Parameter complexity
        - L(9) : float
            Precision complexity

    Notes
    -----
    Returns the moments of the posterior p.d.f. of the parameters of a nonlinear model specified by IS(P, M, U) under Gaussian assumptions. Usually, IS would be an integrator of a dynamic MIMO input-state-output model:
        dx/dt = f(x, u, p)
        y = G(g)*x + X0*B + e

    The E-Step uses a Fisher-Scoring scheme and a Laplace approximation to estimate the conditional expectation and covariance of P. If the free-energy starts to increase, a Levenberg-Marquardt scheme is invoked. The M-Step estimates the precision components of e, in terms of [Re]ML point estimators of the log-precisions. An optional feature selection can be specified with parameters M.FS.

    References
    ----------
    Karl Friston
    Copyright (C) 2009-2022 Wellcome Centre for Human Neuroimaging
    """
    # Ensure default values for M
    M = copy.deepcopy(M)
    U = copy.deepcopy(U)
    Y = copy.deepcopy(Y)
    M.setdefault("nograph", 0)
    M.setdefault("Nmax", 200)
    M.setdefault("Gmax", 8)
    M.setdefault("Hmax", 4)

    # Check observer has not been accidentally specified
    M.pop("g", None)

    # Composition of feature selection and prediction (usually an integrator)
    if "FS" in M:
        try:
            y = M["FS"](Y["y"], M)
            FS = M["FS"]
        except TypeError:
            y = M["FS"](Y["y"])
            FS = lambda y: M["FS"](Y["y"])
    else:
        y = Y["y"]
        FS = lambda y, M: y

    if isinstance(y, list):
        # Concatenate samples over list, ensuring the same for predictions
        ns = y[0].shape[0]
        y = concatenate_dod_lol([[y[0]], [y[1]]])
        IS = M["IS"]
    else:
        ns = y.shape[0]

    ny = len(vectorise_object(y))
    nr = ny // ns
    M["ns"] = ns

    # Initial states
    # --------------------------------------------------------------------------
    try:
        M["x"]
    except KeyError:
        try:
            M["n"]
        except KeyError:
            M["n"] = 0
        M["x"] = np.zeros((M["n"], 1))

    # Input
    # --------------------------------------------------------------------------
    try:
        U
    except NameError:
        U = []

    # Initial parameters
    # --------------------------------------------------------------------------
    try:
        vectorise_object(M["P"]) - vectorise_object(M["pE"])
    except KeyError:
        M["P"] = M["pE"].copy()

    try:
        vectorise_object(M["Q"]) - vectorise_object(M["gE"])
    except KeyError:
        M["Q"] = M["gE"].copy()

    # Time-step
    # --------------------------------------------------------------------------
    try:
        Y["dt"]
    except KeyError:
        Y["dt"] = 1

    # Precision components Q
    # --------------------------------------------------------------------------
    try:
        Q = [Y["Q"]]
    except KeyError:
        Q = get_Ce(t="ar", v=[ns] * nr)
    nh = len(Q)
    nt = len(Q[0])
    nq = nr * ns // nt

    # Confounds (if specified)
    # --------------------------------------------------------------------------
    try:
        if len(Y["X0"]) == 0:
            Y["X0"] = np.zeros((ns, 0))
        dgdu = np.kron(np.eye(nr), Y["X0"].reshape((-1, 1)))
    except KeyError:
        dgdu = np.zeros((ns * nr, 0))

    # hyperpriors - expectation (and initialize hyperparameters)
    # --------------------------------------------------------------------------
    try:
        hE = M["hE"]
        if np.isscalar(hE):
            len_hE = 1
        else:
            len_hE = len(hE)
        if len_hE != nh:
            hE = hE + np.zeros((nh, 1))
    except KeyError:
        hE = np.zeros((nh, 1)) - np.log(np.var(vectorise_object(y))) + 4
    h = copy.deepcopy(hE)

    # hyperpriors - covariance
    # --------------------------------------------------------------------------
    try:
        ihC = get_inv(M["hC"])
        if np.isscalar(ihC):
            len_hc = 1
        else:
            len_hc = len(ihC)
        if len_hc != nh:
            ihC = ihC * np.eye(nh, nh)
    except KeyError:
        ihC = np.eye(nh, nh) * np.exp(4)

    # unpack prior covariances
    # --------------------------------------------------------------------------
    if isinstance(M["pC"], dict):
        M["pC"] = get_diag(vectorise_object(M["pC"]))
    if isinstance(M["gC"], dict):
        M["gC"] = get_diag(vectorise_object(M["gC"]))
    if isinstance(M["pC"], np.ndarray) and M["pC"].ndim == 1:
        M["pC"] = get_diag(M["pC"])
    if isinstance(M["gC"], np.ndarray) and M["gC"].ndim == 1:
        M["gC"] = get_diag(M["gC"])

    # dimension reduction of parameter space
    # --------------------------------------------------------------------------

    Vp, *_ = truncate_svd(M["pC"], 0)
    Vg, *_ = truncate_svd(M["gC"], 0)
    np_ = Vp.shape[1]
    ng = Vg.shape[1]
    nu = dgdu.shape[1]

    # prior moments
    # --------------------------------------------------------------------------
    pE = M["pE"]
    gE = M["gE"]
    uE = np.zeros((nu, 1))

    # second-order moments (in reduced space)
    # --------------------------------------------------------------------------
    pC = Vp.T @ M["pC"] @ Vp
    gC = Vg.T @ M["gC"] @ Vg
    uC = np.eye(nu, nu) * np.exp(16)
    ipC = get_inv(pC)
    igC = get_inv(gC)
    iuC = get_inv(uC)
    ibC = concatenate_dod_lol(get_diag([ipC, igC, iuC]))
    # all parameters
    bC = np.eye(ibC.shape[0]) * np.exp(-16)

    # initialize conditional density
    # --------------------------------------------------------------------------
    Ep = M["P"]
    Ep["S"] = Ep["S"].astype(float)
    Ep["A"][0] = Ep["A"][0].astype(float)
    Ep["A"][1] = Ep["A"][1].astype(float)
    Ep["A"][2] = Ep["A"][2].astype(float)
    Ep["B"] = Ep["B"].astype(float)
    Ep["C"] = Ep["C"].astype(float)
    Eg = M["Q"]
    Eg["L"] = Eg["L"].astype(float)
    Eu = get_pinv(dgdu) @ vectorise_object(y)

    # expansion point
    # --------------------------------------------------------------------------
    if M["x"] is not None:
        x0 = np.ones((y.shape[0], 1)) * vectorise_object(M["x"]).T
    else:
        x0 = 0

    # EM
    # ==========================================================================
    criterion = [0, 0, 0, 0]

    C = {"F": -np.inf}
    v = -4
    dgdp = np.zeros((ny, np_))
    dgdg = np.zeros((ny, ng))
    dFdh = np.zeros(nh)
    dFdhh = np.zeros((nh, nh))
    # Initialize EP

    for ip in range(M["Nmax"]):
        # time
        tStart = time.time()
        # Predicted hidden states (x) and dxdp
        dxdp, x = get_diff(IS, Ep, M, U, np.array([1]), [Vp])

        # Check for initial iterations and dissipative dynamics
        if np.all(np.isfinite(vectorise_object(x))):
            Gmax = M["Gmax"]
            if ip < 8:
                vg = -4
            else:
                vg = 2
        else:
            Gmax = 0
        # Optimize g: parameters of G(g)
        # ======================================================================
        for ig in range(1, Gmax):

            # prediction yp = G(g)*x
            # ------------------------------------------------------------------
            dGdg, G = get_diff(M["G"], Eg, M, np.array([1]), [Vg])
            yp = FS((x - x0) @ G.T, M)

            # prediction errors - states
            # ==================================================================
            ey = vectorise_object(y) - vectorise_object(yp) - dgdu @ Eu

            # prediction errors - parameters
            # ------------------------------------------------------------------
            ep = Vp.T @ (vectorise_object(Ep) - vectorise_object(pE))
            eg = Vg.T @ (vectorise_object(Eg) - vectorise_object(gE))
            eu = vectorise_object(Eu) - vectorise_object(uE)

            # gradients
            # ------------------------------------------------------------------
            for i in range(np_):
                dgdp[:, i] = vectorise_object(FS(dxdp[i] @ G.T, M))

            try:
                for i in range(ng):
                    dgdg[:, i] = vectorise_object(FS((x - x0) @ dGdg[i].T, M))
            except Exception:
                dgdg = FS((x - x0) @ dGdg, M)

            # Optimize F(h): parameters of iS(h)
            # ==================================================================
            dgdb = np.concatenate([dgdp, dgdg, dgdu], axis=1)

            for ih in range(M["Hmax"]):

                # precision
                # --------------------------------------------------------------
                iS = np.eye(nt) * np.exp(-32)
                if nh == 1:
                    iS = Q[0] * np.exp(h)
                else:
                    for i in range(nh):
                        iS += Q[i] * np.exp(h[i])
                S = get_inv(iS)
                iS = np.kron(np.eye(nq), iS)
                dFdbb = dgdb.T @ iS @ dgdb + ibC
                Cb = get_inv(dFdbb) + bC

                # precision operators for M-Step
                # --------------------------------------------------------------
                P = [None] * nh
                PS = [None] * nh
                if nh == 1:
                    P[0] = Q[0] * np.exp(h)
                    PS[0] = P[0] @ S
                    P[0] = np.kron(np.eye(nq), P[0])
                else:
                    for i in range(nh):
                        P[i] = Q[i] * np.exp(h[i])
                        PS[i] = P[i] @ S
                        P[i] = np.kron(np.eye(nq), P[i])

                # derivatives: dLdh = dL/dh,...
                # --------------------------------------------------------------
                for i in range(nh):
                    dFdh[i] = (
                        np.trace(PS[i]) * nq / 2
                        - np.real(ey.T @ P[i] @ ey) / 2
                        - get_trace(Cb, dgdb.T @ P[i] @ dgdb) / 2
                    )
                    for j in range(i, nh):
                        dFdhh[i, j] = -get_trace(PS[i], PS[j]) * nq / 2
                        dFdhh[j, i] = dFdhh[i, j]

                dFdhh += np.diag(dFdh)

                eh = h - hE
                dFdh -= ihC * eh
                dFdhh -= ihC
                Ch = get_inv(-dFdhh)

                # M-Step: update ReML estimate of h
                # --------------------------------------------------------------
                dh = get_dx(dFdhh, dFdh, 4)
                h += np.clip(dh, -2, 2)

                # convergence
                # --------------------------------------------------------------
                if dFdh.T @ dh < np.exp(-2):
                    break

            # E-step: optimise F(g,u)
            # ==================================================================

            # update gradients and curvature - confounds
            # ------------------------------------------------------------------
            dFdu = dgdu.T @ iS @ ey - iuC @ eu
            dFduu = -dgdu.T @ iS @ dgdu - iuC

            # Conditional updates of confounds (u)
            # ------------------------------------------------------------------
            du = get_dx(dFduu, dFdu, [4])
            Eu = Eu + du

            # update gradients and curvature - parameters
            # ------------------------------------------------------------------
            dFdg = dgdg.T @ iS @ ey - igC @ eg
            dFdgg = -dgdg.T @ iS @ dgdg - igC

            # Conditional updates of parameters (g)
            # ------------------------------------------------------------------
            dg = get_dx(dFdgg, dFdg, [vg])
            Eg = unvectorise_object(vectorise_object(Eg) + Vg @ dg, Eg)

            # convergence
            # ------------------------------------------------------------------
            dG = dFdg.T @ dg
            if ig > 1 and dG < np.exp(-2):
                break

        # Optimize objective function: F(p) = log-evidence - divergence
        # ======================================================================
        L = [0] * 9
        L[0] = -ey.T @ iS @ ey / 2
        L[1] = -ep.T @ ipC @ ep / 2
        L[2] = -eg.T @ igC @ eg / 2
        L[3] = -eu.T @ iuC @ eu / 2
        L[4] = -eh.T[0] * ihC * eh[0] / 2
        L[5] = -ns * nr * np.log(8 * np.arctan(1)) / 2
        L[6] = -nq * get_logdet(S) / 2
        L[7] = get_logdet(ibC @ Cb) / 2
        L[8] = get_logdet(ihC * Ch) / 2
        F = sum(L)

        # Record increases and reference log-evidence for reporting
        # ----------------------------------------------------------------------
        try:
            F0
            print(f' actual: {F - C["F"]:.3e} ({time.time() - tStart:.2f} sec)')
        except NameError:
            F0 = F

        # If F has increased, update gradients and curvatures for E-Step
        # ----------------------------------------------------------------------
        if F > C["F"] or ip < 4:

            # Update gradients and curvature
            # ------------------------------------------------------------------
            dFdp = dgdp.T @ iS @ ey - ipC @ ep
            dFdpp = -dgdp.T @ iS @ dgdp - ipC

            # Decrease regularization
            # ------------------------------------------------------------------
            v = min(v + 1 / 2, 4)
            str_EM = "EM(+)"

            # Accept current estimates
            # ------------------------------------------------------------------
            C["Cb"] = Cb
            C["Ep"] = Ep
            C["Eg"] = Eg
            C["Eu"] = Eu
            C["h"] = h
            C["F"] = F
            C["L"] = L
        else:
            # Reset expansion point
            # ------------------------------------------------------------------
            Cb = C["Cb"]
            Ep = C["Ep"]
            Eg = C["Eg"]
            Eu = C["Eu"]
            h = C["h"]
            # And increase regularization
            # ------------------------------------------------------------------
            v = min(v - 2, -4)
            str_EM = "EM(-)"

        # Optimize p: parameters of f(x,u,p)
        # ======================================================================
        dp = get_dx(dFdpp, dFdp, [v])
        Ep = unvectorise_object(vectorise_object(Ep) + Vp @ dp, Ep)

        # Convergence
        # ----------------------------------------------------------------------
        dF = dFdp.T @ dp
        ig = max(0, ig)
        print(f'{str_EM}: {ip} ({ig},{ih}) F: {C["F"] - F0:.3e} dF predicted: {dF:.3e}')
        criterion = [(dF < 1e-1)] + criterion[:-1]
        if all(criterion):
            print(" convergence")
            break

    Ep = C["Ep"]
    Eg = C["Eg"]
    Cp = Vp @ C["Cb"][:np_, :np_] @ Vp.T
    Cg = Vg @ C["Cb"][np_ : np_ + ng, np_ : np_ + ng] @ Vg.T
    F = C["F"]
    L = C["L"]
    return Ep, Eg, Cp, Cg, S, F, L


def get_match_str(a, b):
    """
    get_match_str looks for matching labels in two lists of strings
    and returns the indices into both the 1st and 2nd list of the matches.
    They will be ordered according to the first input argument.

    Parameters:
    a (list of str): First list of strings.
    b (list of str): Second list of strings.

    Returns:
    sel1 (list of int): Indices in the first list.
    sel2 (list of int): Indices in the second list.
    """
    # Ensure that both are lists of strings
    if a is None:
        a = []
    elif not isinstance(a, list):
        a = list(a)

    if b is None:
        b = []
    elif not isinstance(b, list):
        b = list(b)

    # Ensure that both are column vectors (lists in Python)
    a = [str(item) for item in a]
    b = [str(item) for item in b]

    # Replace all unique strings by a unique number and use the fact that
    combined = a + b
    _, c = np.unique(combined, return_inverse=True)
    a = c[: len(a)]
    b = c[len(a) :]

    sel1 = []
    sel2 = []
    for i, val in enumerate(a):
        s = np.where(val == b)[0]  # for numeric comparison
        sel1.extend([i] * len(s))
        sel2.extend(s)

    return sel1, sel2


def get_dcm_erp(DCM):
    """
    Estimate parameters of a DCM model (Variational Laplace)

    Parameters
    ----------
    DCM : dict
        Dictionary containing the following keys:
        - name : str
            Name string
        - Lpos : array-like
            Source locations
        - xY : dict
            Data structure
        - xU : dict
            Design structure
        - Sname : list of str
            Cell of source name strings
        - A : list of array-like
            Connection constraints, e.g., [nr x nr double], [nr x nr double], [nr x nr double]
        - B : list of array-like
            Connection constraints, e.g., [nr x nr double], ...
        - C : array-like
            Connection constraints, e.g., [nr x 1 double]
        - options : dict
            Dictionary containing the following keys:
            - trials : list of int
                Indices of trials
            - Tdcm : list of int
                [start, end] time window in ms
            - D : int
                Time bin decimation (usually 1 or 2)
            - h : int
                Number of DCT drift terms (usually 1 or 2)
            - Nmodes : int
                Number of spatial models to invert
            - analysis : str
                'ERP', 'SSR' or 'IND'
            - model : str
                'ERP', 'SEP', 'CMC', 'CMM', 'NMM' or 'MFM'
            - spatial : str
                'ECD', 'LFP' or 'IMG'
            - onset : int
                Stimulus onset (ms)
            - dur : int
                Dispersion (sd)
            - CVA : int
                Use CVA for spatial modes [default = 0]
            - Nmax : int
                Maximum number of iterations [default = 64]

    Returns
    -------
    DCM : dict
        Updated DCM structure with estimated parameters.
    dipfit : dict
        Dipole structure for electromagnetic forward model. This field is removed from DCM.M to save memory and is offered as an output argument if needed.

    Notes
    -----
    The scheme can be initialized with parameters for the neuronal model and spatial (observer) model by specifying the fields DCM.P and DCM.Q, respectively. If previous priors (DCM.M.pE and pC or DCM.M.gE and gC or DCM.M.hE and hC) are specified, they will be used. Explicit priors can be useful for Bayesian parameter averaging but would not normally be called upon because prior constraints are specified by DCM.A, DCM.B, etc.

    References
    ----------
    Karl Friston
    Copyright (C) 2005-2022 Wellcome Centre for Human Neuroimaging
    """

    DCM = copy.deepcopy(DCM)
    name = f"DCM_{date.today()}"

    # Filename and options
    # --------------------------------------------------------------------------
    DCM["name"] = DCM.get("name", name)
    DCM["xU"] = DCM.get("xU", {"X": np.array([[]])})
    Nm = DCM.get("options", {}).get("Nmodes", 8)
    onset = DCM.get("options", {}).get("onset", 60)
    dur = DCM.get("options", {}).get("dur", 16)
    model = DCM.get("options", {}).get("model", "CMC")
    lock = DCM.get("options", {}).get("lock", 0)
    multC = DCM.get("options", {}).get("multiC", 0)
    symm = DCM.get("options", {}).get("symmetry", 0)
    CVA = DCM.get("options", {}).get("CVA", 0)
    Nmax = DCM.get("options", {}).get("Nmax", 400)
    DATA = DCM.get("options", {}).get("DATA", 1)
    # symmetry constraints for ECD models only
    # --------------------------------------------------------------------------
    if DCM["options"].get("spatial") != "ECD":
        symm = 0

    # disallow IMG solutions for generic DCMs
    # --------------------------------------------------------------------------
    if isinstance(model, dict) and DCM["options"].get("spatial") == "IMG":
        DCM["options"]["spatial"] = "ECD"
    # Data and spatial model
    # ==========================================================================
    if DATA:
        file_path = "Data/DCM_recompute_by_DATA.mat"
        data_mat0 = sio.loadmat(file_path, simplify_cells=True)
        data_mat0.keys()
        data_mat0 = convert_sparse_to_dense(data_mat0)
        DCM = set_dtype_to_float64(data_mat0["DCM"])

    xY = copy.deepcopy(DCM["xY"])
    xU = copy.deepcopy(DCM["xU"])
    M = copy.deepcopy(DCM["M"])

    if "X0" not in xY:
        xY["X0"] = np.zeros((xY["y"][0].shape[0], 0))
    if "X" not in xU:
        xU["X"] = np.zeros((1, 0))
    if "scale" not in xY:
        xY["scale"] = 1

    # dimensions
    # --------------------------------------------------------------------------
    Nt = len(xY["y"])
    Nr = DCM["C"].shape[0]
    Nu = len(DCM["C"]) // Nr
    Ns = xY["y"][0].shape[0]
    Nc = xY["y"][0].shape[1]
    Nx = len(xU["X"]) // xU["X"].shape[0]

    # check the number of modes is greater or equal to the number of sources
    # --------------------------------------------------------------------------
    Nm = max(Nm, Nr)

    # confounds - residual forming matrix
    # --------------------------------------------------------------------------
    if "R" in xY:
        M["R"] = copy.deepcopy(xY["R"])
    else:
        X0 = xY["X0"]
        M["R"] = np.eye(Ns) - X0 @ solve(X0.T @ X0, X0.T)

    # Serial correlations (precision components) AR model
    # --------------------------------------------------------------------------
    xY["Q"] = get_Q(np.array([1 / 2]), Ns, 1)
    # Inputs
    # ==========================================================================

    # between-trial effects
    # --------------------------------------------------------------------------
    try:
        if len(DCM["B"]) < Nx:
            for i in range(Nx):
                DCM["B"].append(np.zeros((Nr, Nr)))
    except Exception:
        xU["X"] = np.zeros((1, 0))
        DCM["B"] = []

    # within-trial effects: adjust onset relative to PST
    # --------------------------------------------------------------------------
    M["ons"] = onset - xY["pst"][0]
    M["dur"] = dur
    xU["dt"] = xY["dt"]

    # Model specification and nonlinear system identification
    # ==========================================================================
    M.pop("g", None)

    # prior moments on parameters
    # --------------------------------------------------------------------------
    pE, pC = set_erp_priors(DCM["A"], DCM["B"], DCM["C"])
    # check for trial specific inputs
    # --------------------------------------------------------------------------
    if multC:
        pE["C"] = np.concatenate((pE["C"], pE["C"]), axis=2)
        pC["C"] = np.concatenate((pC["C"], pC["C"]), axis=2)

    # priors on spatial model
    # --------------------------------------------------------------------------
    M["dipfit"]["model"] = model
    gE, gC = set_L_priors(M["dipfit"])

    # hyperpriors (assuming a high signal to noise)
    # --------------------------------------------------------------------------
    hE = 6.0
    hC = 1.0 / 128

    # check for previous priors
    # --------------------------------------------------------------------------
    try:
        pE = M["pE"]
        pC = M["pC"]
        print("Using specified priors (for neural model)")
    except KeyError:
        pass

    try:
        gE = M["gE"]
        gC = M["gC"]
        print("Using specified priors (for spatial model)")
    except KeyError:
        pass

    try:
        hE = M["hE"]
        hC = M["hC"]
        print("Using specified priors (for noise precision)")
    except KeyError:
        pass

    # Feature selection using (canonical) eigenmodes of lead-field
    # ==========================================================================
    if CVA:
        M["U"] = get_dcm_eeg_channelmodes(M["dipfit"], Nm, xY)
    else:
        M["U"] = get_dcm_eeg_channelmodes(M["dipfit"], Nm)

    # scale data features
    # --------------------------------------------------------------------------
    scale = np.std(vectorise_object(define_fy_erp(xY["y"], M)), ddof=1)
    xY["y"] = unvectorise_object(vectorise_object(xY["y"]) / scale, xY["y"])
    xY["scale"] = xY["scale"] / scale

    # likelihood model
    # ==========================================================================

    # Use TFM integration scheme (with plasticity) if indicated
    # --------------------------------------------------------------------------
    if "TFM" in M:
        IS = None
    else:
        IS = get_gen_erp

    # initial states and equations of motion
    # --------------------------------------------------------------------------
    x, f, h = get_dcm_x_neural(pE, model)

    M["FS"] = define_fy_erp
    M["G"] = get_lx_erp
    M["IS"] = IS
    M["f"] = f
    M["h"] = h
    M["x"] = x
    M["pE"] = pE
    M["pC"] = pC
    M["gE"] = gE
    M["gC"] = gC
    M["hE"] = hE
    M["hC"] = hC
    M["m"] = Nu
    M["n"] = len(vectorise_object(M["x"]))
    M["l"] = Nc
    M["ns"] = Ns
    M["Nmax"] = Nmax

    # re-initialise states
    # --------------------------------------------------------------------------
    M["x"] = get_dcm_neural_x(pE, M)

    # EM: inversion
    # ==========================================================================
    Qp, Qg, Cp, Cg, Ce, F, LE = get_nlsi_N(M, xU, xY)
    # # temporarily store results
    variables_to_save = {
        "Qp": Qp,
        "Qg": Qg,
        "Cp": Cp,
        "Cg": Cg,
        "Ce": Ce,
        "F": F,
        "LE": LE,
    }

    # Save to a .mat file
    sio.savemat("Data/temporal_result_nlsi_N.mat", variables_to_save)

    dp = vectorise_object(Qp) - vectorise_object(pE)
    Pp = unvectorise_object(1 - get_Ncdf(0, np.abs(dp), np.sqrt(np.diag(Cp))), Qp)

    # neuronal and sensor responses (x and y)
    # --------------------------------------------------------------------------
    L = M["G"](Qg, M)
    x = M["IS"](Qp, M, xU)
    # turn it to list
    nn = x.shape[0] // Nt
    x = [x[(nn * i) : (nn * (i + 1)), :] for i in range(Nt)]

    # trial-specific responses (in mode, channel and source space)
    # --------------------------------------------------------------------------
    try:
        j = np.where(np.kron(Qg["J"], np.ones((1, Nr))).flatten())[0]
    except:
        j = np.where(vectorise_object(Qg["J"]).flatten())[0]

    x0 = np.ones((Ns, 1)) * vectorise_object(M["x"]).T
    K = []
    y = []
    r = []

    for i in range(Nt):
        K_i = x[i] - x0
        y_i = M["R"] @ K_i @ L.T @ M["U"]
        r_i = M["R"] @ xY["y"][i] @ M["U"] - y_i
        K_i = K_i[:, j]
        K.append(K_i)
        y.append(y_i)
        r.append(r_i)

    # store estimates in DCM
    # --------------------------------------------------------------------------
    DCM["M"] = M
    DCM["xY"] = xY
    DCM["xU"] = xU
    DCM["Ep"] = Qp
    DCM["Cp"] = Cp
    DCM["Eg"] = Qg
    DCM["Cg"] = Cg
    DCM["Ce"] = Ce
    DCM["Pp"] = Pp
    DCM["H"] = y
    DCM["K"] = K
    DCM["x"] = x
    DCM["R"] = r
    DCM["F"] = F
    DCM["L"] = LE

    DCM["options"]["Nmodes"] = M["U"].shape[1]
    DCM["options"]["onset"] = onset
    DCM["options"]["dur"] = dur
    DCM["options"]["model"] = model
    DCM["options"]["lock"] = lock
    DCM["options"]["symm"] = symm
    DCM["options"]["analysis"] = "ERP"

    if M["dipfit"]["type"] == "IMG" and DATA:

        # Assess accuracy; signal to noise (over sources), SSE and log-evidence
        # ----------------------------------------------------------------------
        SSR = np.zeros(Nt)
        SST = np.zeros(Nt)
        for i in range(Nt):
            SSR[i] = np.sum(np.var(r[i], axis=0, ddof=1))
            SST[i] = np.sum(np.var(y[i] + r[i], axis=0, ddof=1))
        R2 = 100 * (np.sum(SST - SSR)) / np.sum(SST)

        # reconstruct sources in dipole space
        # ----------------------------------------------------------------------
        Nd = M["dipfit"]["Nd"]
        G = np.zeros((Nd, Nr))

        # one dipole per subpopulation (p)
        # ----------------------------------------------------------------------
        if isinstance(Qg["L"], list):
            for p in range(len(Qg["L"])):
                for i in range(Nr):
                    G = np.hstack((G, M["dipfit"]["U"][i] @ Qg["L"][p][:, i]))
        else:
            for i in range(Nr):
                G[M["dipfit"]["Ip"][i].astype(np.int32), i] = (
                    M["dipfit"]["U"][i] @ Qg["L"][:, i]
                )

            G = np.kron(Qg["J"], G)

        Is = np.where(np.any(G, axis=1))[0]
        Ix = np.where(np.any(G, axis=0))[0]
        G = G[Is, :][:, Ix]
        J = [G @ K[i].T for i in range(Nt)]

        # get D and dipole space lead field
        # ----------------------------------------------------------------------
        try:
            val = DCM["val"]
        except KeyError:
            val = 1

        D = sio.loadmat(
            "Data/SPMgainmatrix_mfaeffspmeeg_examplecontrolsubject_5.mat",
            simplify_cells=True,
        )
        G_ = D["G"]
        labels = D["label"]
        channels = DCM["xY"]["name"]
        _, sel2 = get_match_str(channels, labels)
        L = G_[sel2, :][:, (Is - 1)]
        L = M["U"].T @ L

        # reduced data (for each trial)
        # ----------------------------------------------------------------------
        Y = [M["U"].T @ xY["y"][i].T @ M["R"] for i in range(Nt)]

        # fill in fields of inverse structure
        # ----------------------------------------------------------------------
        inverse = {
            "trials": DCM["options"]["trials"],
            "modality": [DCM["xY"]["modality"]],
            "type": "DCM",
            "J": J,
            "L": L,
            "R": np.eye(Nc),
            "T": M["R"],
            "U": M["U"],
            "Is": Is - 1,
            "It": DCM["xY"]["It"],
            "Ic": DCM["xY"]["Ic"],
            "Y": Y,
            "Nd": Nd,
            "Nt": Nt,
            "pst": xY["pst"],
            "F": DCM["F"],
            "R2": R2,
            "dipfit": M["dipfit"],
        }

    # remove dipfit structure to save memory
    # --------------------------------------------------------------------------
    dipfit = DCM["M"]["dipfit"]
    del DCM["M"]["dipfit"]
    del DCM["M"]["g"]

    # Assuming DCM and inverse are defined somewhere in your code
    with open("Data/DCM1.pkl", "wb") as dcm_file:
        pickle.dump(DCM, dcm_file)

    # Save the inverse dictionary
    with open("Data/inverse1.pkl", "wb") as inverse_file:
        pickle.dump(inverse, inverse_file)

    return DCM, inverse, dipfit
