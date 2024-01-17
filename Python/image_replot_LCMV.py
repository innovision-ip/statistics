# %%

import argparse
import datetime as dt
from pathlib import Path

import core_utilities as core
import h5py as h5
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from nilearn import plotting

# Define the filename
sub_id = "sub-CC110069"
results_dir = "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
# %%
band_name = "beta"
no_epochs = 100
filename_ep = f"{results_dir}{sub_id}_{band_name}_stat_image_{no_epochs}_ep.nii"
filename_sh = f"{results_dir}{sub_id}_{band_name}_map_image_{no_epochs}_sh.nii"
filename_lw = f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_lw.nii"

# Load the NIfTI image
img_ep = nib.load(filename_ep)
img_sh = nib.load(filename_sh)
img_lw = nib.load(filename_lw)


if sub_id[:2] == "su":
    scale_factor = 1e2
    vmax = 24
    threshold = 1.5
else:
    scale_factor = 10
    vmax = 170
    threshold = 10

# %% Empirical analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_ep, scale_factor=scale_factor
)

plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=threshold,
    cmap="jet",
    title=f"Empirical, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_ep.png")

# %% Shrinkage analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_sh, scale_factor=scale_factor
)
plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=threshold,
    cmap="jet",
    title=f"Shrinkage, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_sh.png")

# %% ledoit_wolf analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_lw, scale_factor=scale_factor
)
plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=threshold,
    cmap="jet",
    title=f"Ledoit-Wolf, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_lw.png")

# %%
