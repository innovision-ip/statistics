"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2024-08-08 21:10:49
Last Edited : 2024-08-15 12:18:40
Last Author : Jie Li
File Path   : /DCM/Users/Jie/Documents/dcm_IP/dcm/spm_dcm_erp_test.py
Description :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import copy

# and save
import pickle
from datetime import date

import numpy as np
import scipy.io as sio

from utils import *
from utils_erp import *

file_path = "/Users/jie/dcm-erp-tyrer/DCM-Model-ERP-specification.mat"
data_mat = sio.loadmat(file_path, simplify_cells=True)
data_mat.keys()
data_mat = convert_sparse_to_dense(data_mat)
DCM = data_mat["DCM"]
DCM = set_dtype_to_float64(DCM)

a, b, c = get_dcm_erp(DCM)
# %%
b["R2"]
b["F"]
