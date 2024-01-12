"""
Created on Sun Nov 19 11:55:01 2023

@author: gary

programme to demonstrate how to read a meg file; how to analyse
the data to obtain an estimate of source activity;

__copyright__ = 'Copyright (C) 2020, Innovision IP Ltd'
__license__ = Flimal - Innovision IP Limited

(C) Innovision IP Limited 2020. All Rights Reserved.

This confidential software and the copyright therein are the property of
Innovision IP Limited and may not be used, copied or disclosed to any third
party without the express written permission of Innovision IP Limited.



__email__ = 'enquiries@innovision-ip.co.uk'
__author__ = 'Gary Green, Mark Hymers
"""


# imports

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

# set up
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

# test_sub = "CC110069"
# band_name = "alpha"
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

##some defs

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


# load the data
print(meg_file)
meg_info, meg_data, n_sensors, temp = core.load_meg_info_and_data(meg_file, None)
bads = meg_info["bads"]


picks = mne.pick_types(meg_info, meg=True)
n_sig_channels = len(picks)

# filter the data

lf = bands[band_name][0]
hf = bands[band_name][1]
meg_data = meg_data.filter(lf, hf, n_jobs=job_count)


# get emptyroom data. This is needed for some inverse methods

room_info, room_data, n_sensors, temp = core.load_meg_info_and_data(
    empty_room_file, None
)

filtered_empty_room_data = room_data.filter(lf, hf, n_jobs=job_count)

# create a covariance of the whole emptyroom data
empty_room_cov = core.calc_cov(filtered_empty_room_data)


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

# The Gain Matrix (`G`) is the important forward lead field gain matrix
# we first remove the bad channels
fwd_solution = mne.pick_channels_forward(fwd, exclude="bads")
gain_matrix = fwd_solution["sol"]["data"]
n_sensors = np.shape(gain_matrix)[0]

# Note that `G` is in three directions so three times number of sources.
# We want to use 2 directions in the forward model and whiten the matrix
gain_matrix = core.generate_2_directions(
    gain_matrix, n_sensors, fwd_solution["nsource"]
)
gain_matrix = core.whiten(empty_room_cov.data, gain_matrix)


# create epochs
events, epochs = core.generate_epochs_and_events(meg_data, epoch_duration)

# The numbers of solutions is the number of locations multiplied by the
# number of directions
source_map = np.zeros([gain_matrix.shape[1], no_epochs])

# Used to return a list of booleans indicating which epochs are with or
# without solutions
epoch_soln_map = []

soln_idx = 0

soln_idx = 0
# %%
# save the 21th epoch to do the test
epoch = epochs.get_data(copy=True)[20, :, :]
cov_empty = empty_room_cov.data
np.savez(
    "cvxpy_test.npz",
    epoch=epoch,
    cov_empty=cov_empty,
    gain_matrix=gain_matrix,
    bb_truncation=bb_truncation,
    g_truncation=g_truncation,
    job_count=job_count,
)

for idx_epoch, epoch in enumerate(epochs):
    # Whiten the covariance of the B (MEG) data based on the spectrum of the
    # empty room data
    print(epoch.shape)
    cov = np.cov(core.whiten(empty_room_cov.data.data, epoch))

    # I've done some performance tests. This method is fastest, despite what
    # is claimed here
    # <https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy>
    if np.isnan(cov).any() or np.isnan(gain_matrix).any():
        print("Epoch %d: NaN detected in covariance or MEG data.", idx_epoch)
        epoch_soln_map.append(False)
    else:
        try:
            # Run the main algorithm using the whitened data covariance data
            # (BB), the leadfield (G) and the truncation values
            solution = core.compute_l1_minimisation(
                cov, gain_matrix, bb_truncation, g_truncation, job_count
            )
            print("Epoch ", idx_epoch, "Solution found")

            source_map[:, soln_idx] = solution
            epoch_soln_map.append(True)

            soln_idx += 1
            if soln_idx >= no_epochs:
                break

        except Exception as e:
            print("Epoch ", idx_epoch, " No solution ", e)
            epoch_soln_map.append(False)


no_solns_found = soln_idx

if no_solns_found < no_epochs:
    print(
        "Could not find sufficient solutions "
        f"({no_solns_found} is less than {no_epochs})"
    )


# create some maps
average_map = np.mean(source_map, axis=1)
std_map = np.std(source_map, axis=1)

# TODO: #180. Divide by 2 because we work in two directions. Would
# possibly be better to generalise this and pull the number of
# directions / vertices from `fwd_solution` but for now it's
# hard-coded so it's fine.
vector_image = np.zeros(int(len(average_map) / 2))
vector_std = np.zeros(int(len(std_map) / 2))
z_map = np.zeros(int(len(average_map) / 2))

# Find best vector
vector_image = core.vectorise(average_map, int(len(average_map) / 2))
vector_std = core.vectorise(std_map, int(len(average_map) / 2))
for i in range(int(len(average_map) / 2)):
    if vector_std[i] > 0.0:
        z_map[i] = vector_image[i, 0] / vector_std[i, 0]

# Extract the vertices information
vertices = [fwd_solution["src"][0]["vertno"]]

# save the relevant data into an HDF5 file
if simulated == "False":
    with h5.File(f"{results_dir}{sub_id}_{band_name}_{no_epochs}.hdf5", "w") as fout:
        grp_subject = fout.create_group(sub_id)

        grp_results = grp_subject.create_group("Results")
        grp_results.create_dataset("Source_map", data=source_map)
        grp_results.create_dataset("Average_image", data=average_map)
        grp_results.create_dataset("Vertices", data=vertices)
        grp_results.create_dataset("Gain_matrix", data=gain_matrix)
        grp_results.create_dataset("Epoch_list", data=epoch_soln_map)

        grp_meg = grp_subject.create_group("MEG")
        grp_meg.attrs.update(
            {"Time_updated": str(dt.date.today()), "MEG_data_file": str(meg_file)}
        )

        grp_mri = grp_subject.create_group("MRI")
        grp_mri.attrs.update(
            {
                "Time_updated": str(dt.date.today()),
                "MRI_data_file": str(mri_t1_file),
                "MRI_surface_directory": str(mri_surface_path),
                "MRI_bem_directory": str(mri_bem_path),
                "MRI_bem_surface": str("inner_skull.surf"),
            }
        )
else:
    with h5.File(
        f"{simulated_data_path}{sub_id}_{band_name}_{no_epochs}.hdf5", "w"
    ) as fout:
        grp_subject = fout.create_group(sub_id)

        grp_results = grp_subject.create_group("Results")
        grp_results.create_dataset("Source_map", data=source_map)
        grp_results.create_dataset("Average_image", data=average_map)
        grp_results.create_dataset("Vertices", data=vertices)
        grp_results.create_dataset("Gain_matrix", data=gain_matrix)
        grp_results.create_dataset("Epoch_list", data=epoch_soln_map)

        grp_meg = grp_subject.create_group("MEG")
        grp_meg.attrs.update(
            {"Time_updated": str(dt.date.today()), "MEG_data_file": str(meg_file)}
        )

        grp_mri = grp_subject.create_group("MRI")
        grp_mri.attrs.update(
            {
                "Time_updated": str(dt.date.today()),
                "MRI_data_file": str(mri_t1_file),
                "MRI_surface_directory": str(mri_surface_path),
                "MRI_bem_directory": str(mri_bem_path),
                "MRI_bem_surface": str("inner_skull.surf"),
            }
        )

# This section on constructing images needs serious review and is probably incorrect
# Save out each image
# Create some maps for saving out
stc = mne.VolSourceEstimate(z_map, vertices, 0, 1, sub_id)
stc_map = mne.VolSourceEstimate(vector_image, vertices, 0, 1, sub_id)

# Convert each of the volumes into nifti image objects - individual MRI space
img_stat = stc.as_volume(
    src=src_space, dest="mri", mri_resolution=True, format="nifti1"
)

img_map = stc_map.as_volume(
    src=src_space, dest="mri", mri_resolution=True, format="nifti1"
)

# Morph the images into the fsaverage space for group work
morph = mne.compute_source_morph(
    src_space,
    subject_from=sub_id,
    subject_to="fsaverage",
    subjects_dir=mri_subjects_path,
)

img_fsaverage = morph.apply(stc, mri_resolution=False, output="nifti1")


nib.nifti1.save(
    img_stat, filename=f"{results_dir}{sub_id}_{band_name}_stat_image_{no_epochs}"
)
nib.nifti1.save(
    img_map, filename=f"{results_dir}{sub_id}_{band_name}_map_image_{no_epochs}"
)
nib.nifti1.save(
    img_fsaverage, filename=f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}"
)

plotting.plot_stat_map(img_fsaverage)
plt.savefig(f"{results_dir}{sub_id}_{band_name}_fs_image_{no_epochs}.png")
# plotting.show()
