"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2024-06-01 21:22:36
Last Edited : 2024-07-16 07:57:29
Last Author : Jie Li
File Path   : /DCM/Python/dcm/dcm_estimate_test.py
Description : This script is used to test the DCM estimation algorithm based on the .mat file in ~/attention/GLM/DCM_mod_bwd.mat








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

# %%

import numpy as np
import scipy.io as sio

# scipy for detrend function
from utils import *

file_path = "/Users/jie/attention/GLM/DCM_mod_bwd.mat"
data_mat = sio.loadmat(file_path, simplify_cells=True)
data_mat = convert_sparse_to_dense(data_mat)
dcm_dictionary = data_mat["DCM"]  # DCM structure, dictionary

# %%
dcm_static = estimate_dcm(dcm_dictionary)
print(dcm_static["F"])
print(dcm_static["Ep"]["A"])

# %%
dcm_dictionary["options"]["stochastic"] = 1
dcm_stochastic = estimate_dcm(dcm_dictionary)
print(dcm_stochastic["F"])
print(dcm_stochastic["Ep"]["A"])
