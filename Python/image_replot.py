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
band_name = "alpha"
no_epochs = 100
filename = f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}.nii"

# Load the NIfTI image
img_fsaverage = nib.load(filename)
# Get the data from the Nifti image
data = img_fsaverage.get_fdata()

# # Find the index of the maximum value in the flattened array
z_max = np.max(data)

coords = plotting.find_xyz_cut_coords(img_fsaverage, activation_threshold=z_max - 0.1)
# plotting.show()
plotting.plot_stat_map(
    img_fsaverage,
    cut_coords=coords,
    vmax=9.7,
    threshold=4,
    cmap="jet",
    title=f"Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}.png")

# %%
