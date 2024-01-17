# %% imports

import argparse
import datetime as dt
from pathlib import Path

import core_utilities as core
import h5py as h5
import matplotlib.pyplot as plt
import mne
import nibabel as nib
import numpy as np
from mne.beamformer import apply_lcmv, apply_lcmv_cov, make_lcmv
from nilearn import image, plotting

# %% set up
mne.utils.use_log_level("error")

# get the arguments from the command line
# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", dest="test_subject", help="Subject ID")
parser.add_argument("-b", "--band", dest="band_name", help="band name")
parser.add_argument(
    "-t", "--test", dest="sim_data", help="simulated data True or False"
)

args = parser.parse_args()
test_sub = args.test_subject
band_name = args.band_name
simulated = args.sim_data

# test_sub = "S10149"
# band_name = "delta"
# simulated = False

if test_sub[:2] == "CC":
    sub_id = "sub-" + test_sub

else:
    sub_id = test_sub

# define the directories where the data is stored
camcam_path = "/Users/Jie/KTP_Project/Gary_Shared/20231130/camcam/"
innovision_path = "/Users/Jie/KTP_Project/Gary_Shared/20231130/innovision/"
camcam_empty_room_path = camcam_path + "meg_emptyroom/"
innovision_empty_room_path = innovision_path + "meg/"
camcam_mri_subjects_path = camcam_path + "freesurfer/"
innovision_mri_subjects_path = innovision_path + "freesurfer/"
camcam_transform_path = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/camcam/coreg/trans/"
)
innovision_transform_path = innovision_path + "coreg/trans/"
camcam_meg_path = camcam_path + "meg/"
innovision_meg_path = innovision_path + "meg/"
camcam_hdf5_path = "/Users/Jie/KTP_Project/Gary_Shared/20231130/testanalysis/camcam/"
innovision_hdf5_path = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/testanalysis/innovision/"
)

simulated_data_path = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
)

if simulated == "True":
    Path(simulated_data_path).mkdir(parents=True, exist_ok=True)

results_dir = "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
Path(results_dir).mkdir(parents=True, exist_ok=True)

if test_sub[:2] == "CC":
    mri_subjects_path = camcam_mri_subjects_path
    mri_t1_file = camcam_path + "anat/" + test_sub + "/" + sub_id + "_T1w.nii.gz"
    mri_surface_path = camcam_mri_subjects_path + sub_id + "/surf/"
    mri_bem_path = camcam_mri_subjects_path + sub_id + "/bem/"
    transform_file = camcam_transform_path + sub_id + "-trans.fif"
    meg_file = (
        camcam_meg_path
        + sub_id
        + "/ses-rest/"
        + sub_id
        + "_ses-rest_task-rest_proc-sss.fif"
    )
    empty_room_file = (
        camcam_empty_room_path + sub_id + "/emptyroom_" + sub_id[4:] + ".fif"
    )
else:
    mri_subjects_path = innovision_mri_subjects_path
    # mri_t1_file = innovision_path + "anat/" + sub_id + "/anat/" + sub_id + "_T1w.nii"
    mri_t1_file = innovision_path + "anat/" + sub_id + "/T1_biascorr.nii.gz"
    mri_surface_path = innovision_mri_subjects_path + sub_id + "/surf/"
    mri_bem_path = innovision_mri_subjects_path + sub_id + "/bem/"
    transform_file = innovision_transform_path + sub_id + "-trans.fif"
    meg_file = innovision_meg_path + sub_id + "/resting_eyesclosed1.fif"
    empty_room_file = innovision_empty_room_path + sub_id + "/emptyroom.fif"
    bad_channels_filename = (
        innovision_empty_room_path + sub_id + "/bad_chans.txt"
    )  # can not  find this file

if simulated == "True":
    meg_file = f"{simulated_data_path}{sub_id}_sim_noise.fif"

# %% some defs

bands = {
    "delta": [1.0, 4.0],
    "theta": [4.0, 8.0],
    "alpha": [8.0, 12.0],
    "beta": [12.0, 30.0],
    "gamma": [30.0, 45.0],
    "wide": [1.0, 45.0],
}

bem_ico = 4
bem_conductivity = [0.3]
fwd_grid_spacing = 8.0
fwd_grid_exclude = 20.0
fwd_mindist = 2.0
g_truncation = 45
bb_truncation = 16
job_count = 8

epoch_duration = 2.0
no_epochs = 100  # can be at least 250 if required


# %% load the data
print(meg_file)
meg_info, meg_data, n_sensors, temp = core.load_meg_info_and_data(meg_file, None)
bads = meg_info["bads"]


picks = mne.pick_types(meg_info, meg=True)
n_sig_channels = len(picks)

# filter the data

lf = bands[band_name][0]
hf = bands[band_name][1]
meg_data = meg_data.filter(lf, hf, n_jobs=job_count)


# %% get emptyroom data. This is needed for some inverse methods

room_info, room_data, n_sensors, temp = core.load_meg_info_and_data(
    empty_room_file, None
)

filtered_empty_room_data = room_data.filter(lf, hf, n_jobs=job_count)

# create a covariance of the whole emptyroom data
# empty_room_cov = core.calc_cov(filtered_empty_room_data)
empty_room_cov = mne.compute_raw_covariance(filtered_empty_room_data, rank="info")


# now set up information for inverse method
# Create the BEM model of the subject's head. This requires access to
# the freesurfer brain extractions

surfs = mne.make_bem_model(
    sub_id, ico=bem_ico, conductivity=bem_conductivity, subjects_dir=mri_subjects_path
)

bem = mne.make_bem_solution(surfs)

# Calculate the source space
src_space = mne.setup_volume_source_space(
    sub_id,
    pos=fwd_grid_spacing,
    bem=bem,
    subjects_dir=mri_subjects_path,
    exclude=fwd_grid_exclude,
)

# Calculate the forward model based on the BEM solution
fwd = mne.make_forward_solution(
    meg_info,
    trans=transform_file,
    src=src_space,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=fwd_mindist,
    n_jobs=job_count,
)
# %%
# create epochs
events, epochs = core.generate_epochs_and_events(meg_data, epoch_duration)
evoked = epochs.average()

data_cov_empirical = mne.compute_covariance(epochs, tmin=0, tmax=2, method="empirical")
# shrinkage, cross-validation
data_cov_shrunk = mne.compute_covariance(epochs, tmin=0, tmax=2, method="shrunk")
# Ledoit Wolf
data_cov_ledoit_wolf = mne.compute_covariance(
    epochs, tmin=0, tmax=2, method="ledoit_wolf"
)
# data_cov_empirical.plot(epochs.info)
# data_cov_shrunk.plot(epochs.info)
# data_cov_ledoit_wolf.plot(epochs.info)
# empty_room_cov.plot(epochs.info)
# %% computing the spatial filter
filters_empirical = make_lcmv(
    evoked.info,
    fwd,
    data_cov_empirical,
    reg=0.05,
    noise_cov=empty_room_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank="info",
)
filters_shrunk = make_lcmv(
    evoked.info,
    fwd,
    data_cov_shrunk,
    reg=0.05,
    noise_cov=empty_room_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank="info",
)
filters_ledoit_wolf = make_lcmv(
    evoked.info,
    fwd,
    data_cov_ledoit_wolf,
    reg=0.05,
    noise_cov=empty_room_cov,
    pick_ori="max-power",
    weight_norm="unit-noise-gain",
    rank="info",
)

# You can save the filter for later use with:
# filters.save('filters-lcmv.h5')
# %% apply the spatial filter to covariance matrix
stc_empirical = core.gen_lcmv(data_cov_empirical, empty_room_cov, filters_empirical)
stc_shrunk = core.gen_lcmv(data_cov_shrunk, empty_room_cov, filters_shrunk)
stc_ledoit_wolf = core.gen_lcmv(
    data_cov_ledoit_wolf, empty_room_cov, filters_ledoit_wolf
)

# %%
# Morph the images into the fsaverage space for group work
morph = mne.compute_source_morph(
    src_space,
    subject_from=sub_id,
    subject_to="fsaverage",
    subjects_dir=mri_subjects_path,
)

img_fsaverage_empirical = morph.apply(
    stc_empirical, mri_resolution=False, output="nifti1"
)
img_fsaverage_shrunk = morph.apply(stc_shrunk, mri_resolution=False, output="nifti1")
img_fsaverage_ledoit_wolf = morph.apply(
    stc_ledoit_wolf, mri_resolution=False, output="nifti1"
)

nib.nifti1.save(
    img_fsaverage_empirical,
    filename=f"{results_dir}{sub_id}_{band_name}_stat_image_{no_epochs}_ep",
)
nib.nifti1.save(
    img_fsaverage_shrunk,
    filename=f"{results_dir}{sub_id}_{band_name}_map_image_{no_epochs}_sh",
)
nib.nifti1.save(
    img_fsaverage_ledoit_wolf,
    filename=f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_lw",
)

if test_sub[:2] == "CC":
    scale_factor = 1e2
    vmax = 4
else:
    scale_factor = 10
    vmax = 150

# %% Empirical analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_fsaverage_empirical, scale_factor=scale_factor
)

plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=1.5,
    cmap="jet",
    title=f"Empirical, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_ep.png")

# %% Shrinkage analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_fsaverage_shrunk, scale_factor=scale_factor
)
plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=1.5,
    cmap="jet",
    title=f"Shrinkage, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_sh.png")

# %% ledoit_wolf analysis
rescaled_img, coords, z_max = core.rescale_and_find_cut_coords(
    img_fsaverage_ledoit_wolf, scale_factor=scale_factor
)
plotting.plot_stat_map(
    rescaled_img,
    cut_coords=coords,
    vmax=vmax,
    threshold=1.5,
    cmap="jet",
    title=f"Ledoit-Wolf, Max Z: {z_max:.2f}",
)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}_lw.png")
