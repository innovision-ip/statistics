"""
Author         : Jie Li, Innovision IP Ltd and School of Mathematics, Statistics and Actuarial Science, University of Kent.
Date           : 2024-01-30 09:04:36
Last Revision  : 2024-01-30 09:04:50
Last Author    : Jie Li
File Path      : /KTP-Mini-Project/statistics/Python/test_gof_realdata.py
Description    :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

# %%
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from core_utilities import general_gof
from dfply import X, mask
from skewt_scipy.skewt import skewt
from tqdm import tqdm

# %% parser the arguments in command line
parser = argparse.ArgumentParser(
    description="Compute the Band result given the test subject."
)
parser.add_argument("-s", "--subject", dest="test_subject", help="Subject ID")
parser.add_argument("-b", "--band", dest="band", help="Band name")
# args = parser.parse_args()
# test_subject = args.test_subject
# band = args.band
test_subject = "S10149"
band = "gamma"
band_name = band

# %% set paths
volume_dir = "/Volumes/lijie4tbtmp/KTPGary"
controls_results_directory = volume_dir + "/control_analysis_27_37F/"
controls_path = volume_dir + "/controls_27_37F.yaml"
test_results_directory = (
    volume_dir + "/" + test_subject + "/results/innovision/" + test_subject + "/"
)

# %%
n_epochs = 100
path_file = f"{test_results_directory}{band_name}/AllData.csv"
df = pd.read_csv(path_file)
area_labels = []
label_list = df["Area"].unique()
for label in label_list:
    area_labels.append(label)
# %% use dfply
# the number of controls is 103, and the number of epochs is 100, and the number of area is 68. The size of control_values_3d is (103, 100, 68).
sub_values_2d = np.zeros((n_epochs, len(area_labels)))
no_controls_real = int(df.shape[0] / n_epochs / len(area_labels)) - 1
control_values_3d = np.zeros((no_controls_real, n_epochs, len(area_labels)))

for i, label in enumerate(area_labels):
    print(label)
    print(i)
    temp = df >> mask(X.Area == label)
    sub_temp = temp >> mask(X.condition == "test")
    sub_values_2d[:, i] = np.array(sub_temp["Value"])
    controls = temp >> mask(X.condition != "test")
    control_values_3d[:, :, i] = np.array(controls["Value"]).reshape(
        (no_controls_real, n_epochs)
    )
# %%
# take log transformation
sub_values_2d = np.log(sub_values_2d)
control_values_3d = np.log(control_values_3d)
# estimate the skewness, location and scale parameters of skew t for each area each subject.
para_test = np.zeros((len(area_labels), 6))
para_control = np.zeros((no_controls_real, len(area_labels), 6))
# %%
for i in tqdm(range(len(area_labels))):
    # test whether it is skew t distribution for test
    try:
        res = general_gof(sub_values_2d[:, i])
        para_test[i, 0] = res[0].get(0.05)
        para_test[i, 1] = res[0].get(0.01)
        para_test[i, 2:] = res[1:5]
    except Exception:
        pass
    for j in tqdm(range(no_controls_real)):
        # test whether it is skew normal distribution for control
        # p_value_control[j, i] = SWtest(control_values_3d[j, :, i])
        try:
            res = general_gof(control_values_3d[j, :, i])
            para_control[j, i, 0] = res[0].get(0.05)
            para_control[j, i, 1] = res[0].get(0.01)
            para_control[j, i, 2:] = res[1:5]
        except Exception:
            pass
# %%
# save the variables: sub_values_2d, control_values_3d, p_value_test, para_test, p_value_control, para_control
path_file = f"../Data/{test_subject}/{band_name}/"
Path(path_file).mkdir(parents=True, exist_ok=True)
file = path_file + "skewtresults.pkl"
# Open the file in write-binary mode and save the variables
with open(file, "wb") as f:
    pickle.dump(
        (
            sub_values_2d,
            control_values_3d,
            # p_value_test,
            para_test,
            # p_value_control,
            para_control,
        ),
        f,
    )


# %% combine the control groups across the subjects
para_control_group = np.zeros((len(area_labels), 6))
for i in tqdm(range(len(area_labels))):
    # test whether it is skew t distribution for test
    try:
        res = general_gof(control_values_3d[:, :, i].ravel())
        para_control_group[i, 0] = res[0].get(0.05)
        para_control_group[i, 1] = res[0].get(0.01)
        para_control_group[i, 2:] = res[1:5]
    except Exception:
        pass

file = path_file + "skewtresultsNew.pkl"
# Open the file in write-binary mode and save the variables
with open(file, "wb") as f:
    pickle.dump(
        (
            sub_values_2d,
            control_values_3d,
            # p_value_test,
            para_test,
            para_control,
            para_control_group,
        ),
        f,
    )
# %%
np.sum(para_control[:, :, 1]) / 103 / 68
np.sum(para_control[:, :, 0]) / 103 / 68
# %%
bb = para_control[:, :, 1]
# %% rejected skew-t
i = 0
j = 4
a, df, loc, scale = para_control[i, j, 2:]
x = np.sort(control_values_3d[j, :, i])
fig, ax = plt.subplots(1, 1)
r = np.sort(skewt.rvs(a=a, df=df, loc=loc, scale=scale, size=1000))
ax.plot(
    r,
    skewt.pdf(r, a=a, df=df, loc=loc, scale=scale),
    "r-",
    lw=2.5,
    alpha=0.6,
    label="skewt pdf",
)
ax.hist(x, density=True, bins="auto", histtype="stepfilled", alpha=0.1)
ax.set_xlim([r[0], r[-1]])
ax.legend(loc="best", frameon=False)
plt.show()
# %%
i = 0
j = 0
a, df, loc, scale = para_control[i, j, 2:]
x = np.sort(control_values_3d[j, :, i])
fig, ax = plt.subplots(1, 1)
ax.plot(
    x,
    skewt.pdf(x, a=a, df=df, loc=loc, scale=scale),
    "r-",
    lw=2.5,
    alpha=0.6,
    label="skewt pdf",
)
ax.hist(x, density=True, bins="auto", histtype="stepfilled", alpha=0.1)
ax.set_xlim([x[0], x[-1]])
ax.legend(loc="best", frameon=False)
plt.show()

# %%
