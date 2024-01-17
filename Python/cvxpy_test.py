# %%
from cProfile import Profile
from pstats import SortKey, Stats

import core_utilities as core
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np

# please generate a random data on your side.
# data = np.load("cvxpy_test.npz")
# epoch = data["epoch"]
# cov_empty = data["cov_empty"]
# G = data["gain_matrix"]
# bb_truncation = data["bb_truncation"]
# g_truncation = data["g_truncation"]
# job_count = data["job_count"]

# %%
# with Profile() as prof:
# whitening the covariance
cov = np.cov(core.whiten(cov_empty, epoch))
BB = cov
# SVD decomposition of the covariance and gain matrix.
Ub, sb, _ = np.linalg.svd(BB, full_matrices=False)
Ug, sg, VgT = np.linalg.svd(G, full_matrices=False)
plt.plot(sb)
plt.plot(sg)
# %%
# check the rank.
if bb_truncation is not None:
    # Get rank of matrices
    bb_rank = np.linalg.matrix_rank(BB)

    # No point in using truncation if it is smaller than rank and if values
    # of singular values are smaller then tolerance for that rank/tolerance
    if bb_rank < bb_truncation:
        bb_truncation = bb_rank

    Ub = Ub[:, :bb_truncation]
    sb = sb[:bb_truncation]
    # VbT = VbT[:bb_truncation, :]  # unused

if g_truncation is not None:
    g_rank = np.linalg.matrix_rank(G)

    if g_rank < g_truncation:
        g_truncation = g_rank

    Ug = Ug[:, :g_truncation]
    sg = sg[:g_truncation]
    VgT = VgT[:g_truncation, :]

# %%
# create diagonal matrix from singular values
Sb = np.diag(np.sqrt(sb))
Sg = np.diag(sg)

UgT = Ug.transpose()
Vg = VgT.transpose()

# %%
# test Ug @ UgT
# np.diag(UgT @ Ug)
# np.matmul(Vg, VgT)
# %%
w = np.sqrt(np.diag(Vg @ VgT))
# constraint A@x=D
UbSb = Ub @ Sb
D = UgT @ UbSb
A = Sg @ VgT

# %%
# compare l1_cvxpy_l1 with l1_cvxpy_l1_adjust
# take the first column of D
b = D[:, 0]
result0 = core.l1_cvxpy_l1(A, b, w)
result1 = core.l1_cvxpy_l1_adjust(A, b, w)
np.sum(result0 >= 0)
np.sum(result1 >= 0)  # apparently, the result is different.
min0 = np.abs(w.T @ result0)
min1 = w.T @ np.abs(result1)
print(min0, min1)
# Stats(prof).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
