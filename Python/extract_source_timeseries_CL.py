"""
Created on Sun Nov 19 11:55:01 2023

@author: gary

programme to demonstrate how to read MEG results from an hdf5 file; how to reconstruct the source time series

__copyright__ = 'Copyright (C) 2020, Innovision IP Ltd'
__license__ = Flimal - Innovision IP Limited

(C) Innovision IP Limited 2020. All Rights Reserved.

This confidential software and the copyright therein are the property of
Innovision IP Limited and may not be used, copied or disclosed to any third
party without the express written permission of Innovision IP Limited.



__email__ = 'enquiries@innovision-ip.co.uk'
__author__ = 'Gary Green, Mark Hymers
"""

# %% import the required libraries
import argparse
import datetime as dt

import core_utilities as core
import h5py as h5
import matplotlib.pyplot as plt
import mne
import numpy as np
from joblib import Parallel, delayed

# %% set up
mne.utils.use_log_level("error")
plt.ion()

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
# band_name = "beta"
# simulated = "False"

if test_sub[:2] == "CC":
    sub_id = "sub-" + test_sub

else:
    sub_id = test_sub

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
job_count = 6

epoch_duration = 2.0
no_epochs = 100  # can be at least 250 if required

# define the directories where the data is stored
test_path = "/media/gary/Seagate/controls_29_39/"
camcam_path = "/Users/Jie/KTP_Project/Gary_Shared/20231130/camcam/"
innovision_path = "/Users/Jie/KTP_Project/Gary_Shared/20231130/innovision/"
camcam_empty_room_path = camcam_path + "meg_emptyroom/"
innovision_empty_room_path = innovision_path + "meg/"
camcam_mri_subjects_path = camcam_path + "freesurfer/"
innovision_mri_subjects_path = innovision_path + "freesurfer/"
camcam_transform_path = "/media/gary/Seagate/BIG_DATA/data/camcam/coreg/trans/"
innovision_transform_path = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/camcam/coreg/trans/"
)
camcam_meg_path = camcam_path + "meg/"
innovision_meg_path = innovision_path + "meg/"

results_dir = "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
analysis_data_filename = f"{results_dir}{sub_id}_{band_name}_{no_epochs}.hdf5"
simulated_data_path = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
)

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
    mri_t1_file = innovision_path + "anat/" + sub_id + "/T1_biascorr.nii.gz"
    mri_surface_path = innovision_mri_subjects_path + sub_id + "/surf/"
    mri_bem_path = innovision_mri_subjects_path + sub_id + "/bem/"
    transform_file = innovision_transform_path + sub_id + "-trans.fif"
    meg_file = innovision_meg_path + sub_id + "/resting_eyesclosed1.fif"
    empty_room_file = innovision_empty_room_path + sub_id + "/emptyroom.fif"
    bad_channels_filename = innovision_empty_room_path + sub_id + "/bad_chans.txt"
# %% get the data
if simulated == "True":
    meg_file = f"{simulated_data_path}{sub_id}_sim_noise.fif"

# get the area labels  and the results from a previous analysis


# get the original time series from the sensors
# load the data
meg_info, meg_data, n_sensors, temp = core.load_meg_info_and_data(meg_file, None)
bads = meg_info["bads"]

picks = mne.pick_types(meg_info, meg=True)
n_sig_channels = len(picks)

# %% filter the data

lf = bands[band_name][0]
hf = bands[band_name][1]
meg_data = meg_data.filter(lf, hf)
# Gray did not share the fsaverage of this subject to me, then I used the fsaverage in Freesurfer
# mri_subjects_path = "/Applications/freesurfer/7.4.1/subjects/"
area_labels = core.get_canonical_area_list(mri_subjects_path, subject_id=sub_id)

# %% get previous results from analysis - note the source map is for each epoch of original analysis
gain_matrix, vertices, source_map, epoch_list = core.read_analysis_results(
    sub_id, analysis_data_filename
)
# %% make somes epochs so that we recreate the source timeseries for an epoch
# create epochs

events, epochs = core.generate_epochs_and_events(meg_data, epoch_duration)

# %%
# Then find epochs which match those used to compute sptial projection matrices
# Compute the time-series per-vertex and then combine into an ROI form
# args = []
# idx = 0
# timeseries_array = None
# for epoch_no, epoch_state in enumerate(epoch_list):
#     if epoch_state:
#         # Set up the arguments for _run_single_conn_estimate
#         this_epoch_args = (
#             idx,
#             epochs[epoch_no],
#             area_labels,
#             gain_matrix,
#             vertices,
#             source_map,
#             epoch_list,
#             sub_id,
#         )

#         args.append(this_epoch_args)

#         idx += 1

# %%
# Run all of the jobs; use multi-processing to speed things up
# res = Parallel(n_jobs=job_count)(
#     delayed(core.run_single_timeseries_extraction)(*arg) for arg in args
# )
# res = np.array(res)

# max_pos = np.where(res == np.max(res))

no_epochs_list = len(epoch_list)
no_epochs = len(epochs)
part2 = np.repeat(False, no_epochs - no_epochs_list)
epoch_list = np.concatenate((epoch_list, part2))
epochs = epochs[epoch_list]
res = Parallel(n_jobs=job_count)(
    delayed(core.run_single_timeseries_extraction_jie)(
        epoch, gain_matrix, vertices, source
    )
    for epoch, source in zip(epochs, source_map.T)
)
res = np.array(res)

# %% using joblib, we do not need the transpose anymore
# Using dstack gives us [nlabels, ntimepts, nepochs], we want
# [nepochs, vertices, ntimepts] so transpose appropriately
# timeseries = res.transpose(2, 0, 1)

# %% save the relevant data into an HDF5 file
if simulated == "False":
    with h5.File(f"{results_dir}{sub_id}_{band_name}_timeseries.hdf5", "w") as f:
        grp = f.create_group("timeseries")
        grp.create_dataset("timeseries", data=res)
        grp.attrs.update({"Time_updated": str(dt.date.today())})
else:
    with h5.File(
        f"{simulated_data_path}{sub_id}_{band_name}_timeseries.hdf5", "w"
    ) as f:
        grp = f.create_group("timeseries")
        grp.create_dataset("timeseries", data=res)
        grp.attrs.update({"Time_updated": str(dt.date.today())})

# %%
