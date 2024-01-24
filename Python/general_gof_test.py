"""
Author         : Jie Li, Innovision IP Ltd and School of Mathematics, Statistics and Actuarial Science, University of Kent.
Date           : 2024-01-24 21:16:12
Last Revision  : 2024-01-24 21:16:23
Last Author    : Jie Li
File Path      : /KTP-Mini-Project/statistics/Python/general_gof_test.py
Description    :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""
# %%
from core_utilities import general_gof
from skewt_scipy.skewt import skewt

# %%

data = skewt.rvs(a=20, df=4, loc=2, scale=3, size=200, random_state=123)
res = general_gof(data)
print(res)
