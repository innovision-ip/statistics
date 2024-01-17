"""
Edited on Sun Feb 16 12:07:35 2020

@author: ggrg1

This programme simulates a source in regions of the brain as determined
by the DKT atlas

I did not use the parser. Instead, I directly assign the variable.

It can do this for any of the subjects in the database

"""
# %% standard imports
import argparse
from pathlib import Path

# my imports
import core_utilities as core
import mne
import numpy as np

# %% set up
mne.utils.use_log_level("error")

# get the arguments from the command line
# Initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--subject", dest="test_subject", help="Subject ID")
parser.add_argument("-b", "--band", dest="band_name", help="band name")
parser.add_argument("-l", "--label", dest="label_name", help="label_name")

# args = parser.parse_args()
# test_sub = args.test_subject
# band_name = args.band_name
# label_name = args.label_name
test_sub = "S10149"
# test_sub = "CC110069"
label_name = "postcentral-lh"

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

# signal simulation parameters
epoch_duration = 2.0  # seconds
sig_duration = 60.0  # seconds

freq_source = 10.0  # Hz
amplitude = 25  # nanoamperemetre
print("amplitude is", amplitude, "nanoampmetres")

location = "center"  # Use the center of the region as a seed.
extent = 10.0  # Extent in mm of the region.

# duration of whole simulation
duration = 200.0

no_epochs = 100  # this is arbitrary and is not actually used

# standard defines for where the directories are where the data is stored
# test_path = "/media/gary/Seagate/controls_29_39/"
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

sim_results_dir = (
    "/Users/Jie/KTP_Project/Gary_Shared/20231130/test_results/" + sub_id + "/"
)
Path(sim_results_dir).mkdir(parents=True, exist_ok=True)
sim_data_filename = f"{sim_results_dir}{sub_id}_sim_noise_{int(freq_source)}Hz.fif"

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
# %%
#  ======================================================================
# get area labels
area_labels = core.get_canonical_area_list(mri_subjects_path)  # does not work
# =========================================================================
# load some data for this subject just to set up a structure for the simulation

meg_info, meg_data, n_sensors, temp = core.load_meg_info_and_data(meg_file, None)
bads = meg_info["bads"]

picks = mne.pick_types(meg_info, meg=True)
n_sig_channels = len(picks)

sfreq = meg_info["sfreq"]
tstep = 1 / sfreq
times = np.arange(int(duration * sfreq), dtype=np.float64) * tstep

# %%
# ========================================================================
# make the bem model
# ========================================================================

surfs = mne.make_bem_model(
    sub_id, ico=bem_ico, conductivity=bem_conductivity, subjects_dir=mri_subjects_path
)
# %%

bem = mne.make_bem_solution(surfs)
# ========================================================================
# create volume source space.
# ========================================================================

src_space = mne.setup_source_space(
    sub_id,
    spacing="oct5",
    surface="white",
    subjects_dir=mri_subjects_path,
    n_jobs=job_count,
    verbose="error",
)
# %%
# ========================================================================
# create forward model
# ========================================================================

fwd = mne.make_forward_solution(
    meg_info,
    trans=transform_file,
    src=src_space,
    bem=bem,
    meg=True,
    eeg=False,
    mindist=5.0,
    n_jobs=7,
    verbose="error",
)
# this is the important forward lead field gain matrix
G = fwd["sol"]["data"]

# %% get empty room information  only needed if we need empty room noise
# ========================================================================

# get emptyroom data. This is needed fis we need real noise
room_info, room_data, n_sensors, temp = core.load_meg_info_and_data(
    empty_room_file, None
)
# %%
# ========================================================================
# set up simulation
# ========================================================================


# get thelabel for that area
selected_label = mne.read_labels_from_annot(
    sub_id, regexp=label_name, subjects_dir=mri_subjects_path, verbose="error"
)[0]

selected_label.values.fill(1.0)

# Restrict the eligible vertices to be those on the surface under
# consideration and within the label.
# do this to find centre of label for display later

com = selected_label.center_of_mass(
    subject=sub_id, subjects_dir=mri_subjects_path, restrict_vertices=True, surf="white"
)

# for some bizarre reason I have to reread the label in mne
# selected_label = mne.read_labels_from_annot(
#     sub_id, regexp=label_name, subjects_dir=mri_subjects_path, verbose="error"
# )[0]

label = mne.label.select_sources(
    sub_id,
    selected_label,
    location=location,
    extent=extent,
    subjects_dir=mri_subjects_path,
)


###############################################################################
# Generate sinusoids in that  label
# --------------------------------------------------

# The known signal is all zero-s off of the labels of interest
signal = np.zeros((1, len(times)))
# here is where the signal becomes nanoampere metres
signal[0, :] = amplitude * 1e-9 * np.sin(freq_source * 2 * np.pi * times)


print("generating simulation")

# stc_gen = mne.simulation.simulate_stc(src, labels, signal, times[0], tstep,
# allow_overlap=True, value_fun=lambda x: x)

stc_gen = mne.simulation.SourceSimulator(src_space, tstep=tstep, duration=duration)
events = mne.make_fixed_length_events(meg_data, id=1, duration=epoch_duration)
stc_gen.add_data(label, signal[0, :], events)
stc_data = stc_gen.get_stc()

_, peak_time = stc_data.get_peak(hemi="lh")

# this works but others may have problem about xcb
brain = stc_data.plot(
    initial_time=peak_time,
    hemi="lh",
    time_label=None,
    backend="matplotlib",
    subjects_dir=mri_subjects_path,
)

###############################################################################
# Simulate sensor-space signals

# Use the forward solution and could add Gaussian noise to simulate sensor-space
# data from the known source-space signals. The amount of noise is
# controlled by `nave` (higher values imply less noise).

#           or add noise at sensor level, eg empty room data
# ========================================================================

print("generating simulation data")


sim_raw = mne.simulation.simulate_raw(
    meg_info, stc=stc_gen, forward=fwd, verbose="error"
)
# just get the meg data
cov = mne.make_ad_hoc_cov(sim_raw.info)
mne.simulation.add_noise(sim_raw, cov, iir_filter=[0.2, -0.2, 0.04], random_state=1)

cov = mne.compute_raw_covariance(
    sim_raw, tmin=0, tmax=None, method="ledoit_wolf", rank="info"
)
cov.plot(sim_raw.info, proj=True)

### now save data as fif file

sim_raw.save(sim_data_filename, overwrite=True)

# %%
