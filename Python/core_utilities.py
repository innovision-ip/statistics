#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


import cvxpy as cp
import h5py

# import multiprocessing # does not work on Mac OS
import joblib
import mne
import nibabel as nib
import numpy as np
from joblib import Parallel, delayed, wrap_non_picklable_objects
from mne.beamformer import apply_lcmv_cov
from nilearn import plotting
from sklearn.decomposition import PCA

# should not need editing below here
##################


def read_analysis_results(subject, results_file):
    """Reads the analysis results from an image file for a subject.
    Args:
        subject:
            The subject id (`str`).
        results_file:
            The results file we want the results from (`os.PathLike`).
    Returns:
        An `AnalysisResultsDC`."""
    with h5py.File(results_file, "r") as fin:
        gain_matrix = [fin[subject]["Results"]["Gain_matrix"][()][0]]
        vertices = [fin[subject]["Results"]["Vertices"][()][0]][0]
        source_map = np.array(fin[subject]["Results"]["Source_map"][()])
        epoch_list = fin[subject]["Results"]["Epoch_list"][()]
    return gain_matrix, vertices, source_map, epoch_list


def scale_grad(data):
    # scales the data from gradiometers
    data = data * 1.0e13
    return data


def scale_mag(data):
    # scales the data from magnetometers
    data = data * 1.0e15
    return data


def bad_channels(bad_channel_filename):
    """Find bad channels and their names

    Args:
        bad_channel_filename :
            a text file that contains list of numbers that are bad channels
    Returns:
        A tuple consisting of 2 parts:

            `(bad_channels, bad_names)`

        bad_channels:
            string list of bad_channels from the original text file
        bad_names:
            The string names of the meg bad channels for a neuromag machine
    """
    bad_names = []
    bad_channels_list = [np.loadtxt(bad_channel_filename, delimiter=",", ndmin=1)]

    bad_channels_list = [int(i) for i in bad_channels_list[0]]
    for i in bad_channels_list:
        if i < 999:
            bad_names.append("MEG0" + str(i))
        else:
            bad_names.append("MEG" + str(i))
    bad_names = list(set(bad_names))

    return bad_names


def load_meg_info_and_data(meg_filename, bad_channel_filename):
    """Load meg information and data

    Args:
        meg_filename:
            The MEG filename (`string`).
        bad_channel_filename:
            Filename containing a list of bad channels (optional `string`).

    Returns:
        A tuple consisting of 4 parts:

            `(meg_info, meg_data, n_sensors, bad_names)`

        meg_info:
            (`dictionary`).
        meg_data:
            (`dictionary`).
        n_sensors:
            The total number of sensors after bad channels have been
            removed (`int`).
        bad_names:
            List of bad channel names (`list`).
    """
    # load the information header
    meg_info = mne.io.read_info(meg_filename, verbose="error")
    # load the actual data
    meg_data = mne.io.read_raw_fif(
        meg_filename, preload=True, allow_maxshield=True, verbose="error"
    )

    # Get the gradiometer and magnetometer channels
    grad_channel_indices = mne.pick_types(meg_info, meg="grad")
    mag_channel_indices = mne.pick_types(meg_info, meg="mag")
    # Scale to allow computation without running into accuracy issues
    meg_data.apply_function(scale_grad, picks=grad_channel_indices, channel_wise=False)
    meg_data.apply_function(scale_mag, picks=mag_channel_indices, channel_wise=False)

    bad_names = []
    # Identifying bad channels and removing them
    if bad_channel_filename is not None:
        bad_names = bad_channels(bad_channel_filename)
    meg_info["bads"] += bad_names
    meg_data = meg_data.pick_types(meg=True, eeg=False, stim=False, exclude=bad_names)

    n_sensors = meg_info["nchan"]

    return meg_info, meg_data, n_sensors, bad_names


def generate_epochs_and_events(data, epoch_duration):
    """Generates epochs and events for MEG data.

    Args:
        data:
        MEG data (`mne.io.Raw`)
        epoch_duration:
        Epoch duration in seconds (`float`).

    Returns:
        An `EpochsEventsDC`.
    """
    events = mne.make_fixed_length_events(data, duration=epoch_duration, start=0)
    # pylint: disable=not-callable
    epochs = mne.Epochs(
        data,
        events=events,
        tmin=0,
        tmax=epoch_duration,
        baseline=(None, None),
        reject=None,
        preload=True,
    )

    return events, epochs


def get_canonical_area_list(path_mri_subjects_dir, parc_name="aparc", subject_id=None):
    """Extracts the canonical list of areas by examining the subject.

    Args:
        path_mri_subjects_dir:
        Freesurfer subjects' directory (`os.PathLike`).
        parc_name:
        The name of the parcellation to use (`str`: defaults to `aparc`)
        subject_id:
        The subject ID to use (`str`: defaults to `fsaverage`)

    Returns:
        The brain area labels (`[str]`).
    """
    if subject_id is None:
        subject_id = "fsaverage"

    labels_lh = mne.read_labels_from_annot(
        subject_id, parc=parc_name, subjects_dir=str(path_mri_subjects_dir), hemi="lh"
    )

    labels_rh = mne.read_labels_from_annot(
        subject_id, parc=parc_name, subjects_dir=str(path_mri_subjects_dir), hemi="rh"
    )
    labels = labels_lh + labels_rh

    # Remove label unknowns from list
    labels = [x for x in labels if not x.name.startswith("unknown")]

    label_list = []

    # Re-organise the list in the form that we want
    for label in labels:
        hemisphere = label.name[-2:]
        area = label.name[:-3]

        lab_name = "ctx-" + hemisphere + "-" + area

        label_list.append(lab_name)

    print("Loaded ", len(label_list), " area definitions from ", path_mri_subjects_dir)
    return label_list


def compute_time_series(G, projection_map, B_epoch, n_sources, alpha_level=0.05):
    """TODO: #59 Needs a (complete) docstring.

    Make the source time series from the meg data, the gain matrix
    and the spatial projection map, also remove the mean from each time series.

    Args:
        G:

        projection_map:

        B_epoch:

        n_sources:

        alpha_level:
        The alpha level for the statistical test.  Defaults to 0.05
    Returns:
        (`numpy.ndarray`).
    """

    Ja = np.ones([np.shape(B_epoch)[0], 1])
    tempr = np.zeros([1, 2 * n_sources])
    tempr[0, :] = projection_map
    JaA = np.matmul(Ja, tempr)

    # huang equation 10
    Ghash = np.multiply(G, JaA)

    # huang equation 11
    Ug, Sg, Vgh = np.linalg.svd(Ghash, full_matrices=False)
    N = np.shape(Sg)[0]
    alpha = alpha_level * np.max(Sg)
    Sg = np.linalg.inv(np.diag(Sg) + alpha * np.eye(N))
    temp = np.matmul(np.transpose(Vgh), Sg)

    # huang equation 12
    Ghashplus = np.matmul(temp, np.transpose(Ug))

    # this allows reconstruction at ALL locations in head in 2 directions
    GhashplusA = np.multiply(Ghashplus, np.matmul(tempr.T, Ja.T))

    # huang equation 13
    time_series = np.matmul(GhashplusA, B_epoch)

    # remove mean from each time_series

    result = time_series - np.mean(time_series, axis=1, keepdims=True)
    return result


def best_orientation(time_series, n_vertices):
    """TODO: #59 Needs a docstring.
    calc_timeseries_results(analysis_data, fwd, area_labels)
        Args:
            time_series:
            (`numpy.ndarray`).
            n_vertices:
            (`int`).
        Returns:
            (`numpy.ndarray`).
    """
    new_series = np.zeros([np.shape(time_series)[1], 2])
    result = np.zeros([n_vertices, np.shape(time_series)[1]])
    for idx in range(n_vertices):
        new_series[:, 0] = time_series[2 * idx, :]
        new_series[:, 1] = time_series[2 * idx + 1, :]
        pca = PCA(n_components=1)
        pca.fit(new_series)
        best_vector = pca.transform(new_series)
        result[idx, :] = best_vector[:, 0]
    return result


# @delayed
# @wrap_non_picklable_objects
def run_single_timeseries_extraction(*args):
    (
        idx,
        epoch_data,
        area_labels,
        gain_matrix,
        vertices,
        source_map,
        epoch_list,
        subject_id,
    ) = args

    # no_labels = len(area_labels)
    no_vertices = len(vertices)

    B_epoch = np.squeeze(epoch_data)
    timeseries = compute_time_series(
        gain_matrix, source_map[:, idx], B_epoch, no_vertices
    )

    return timeseries


def run_single_timeseries_extraction_jie(epoch_data, gain_matrix, vertices, source_map):
    # no_labels = len(area_labels)
    no_vertices = len(vertices)
    B_epoch = np.squeeze(epoch_data)
    timeseries = compute_time_series(gain_matrix, source_map, B_epoch, no_vertices)

    return timeseries


def calc_cov(raw):
    return mne.compute_raw_covariance(
        raw, tmin=0, tmax=None, method="shrunk", rank="full"
    )


def generate_2_directions(gain_matrix_3d, n_sensors, n_sources):
    """Takes a three dimensional gain matrix and returns the matrix for two
    orthogonal directions.

    Args:
        gain_matrix_3d:
        The 3D gain matrix (`numpy.ndarray[n_sensors, 3 * n_sources]`).
        n_sensors:
        The number of sensors (`int`).
        n_sources:
        The number of sources (`int`).

    Returns:
      2D version of the matrix (`numpy.ndarray[n_sensors, 2 * n_sources]`).
    """
    gain_2d = np.zeros([n_sensors, 2 * n_sources])

    # See `WhiteningMatrices.md` documentation for how this is a rotation from
    # 3D onto a plane for every location on the grid
    for i in range(n_sources):
        # TODO: #134 Can these indexes be names something better?
        temp_i_1 = np.array(np.arange(-1, 1)) + (i + 1) * 2 - 1
        temp_i_2 = np.array(np.arange(-2, 1)) + (i + 1) * 3 - 1
        gain = gain_matrix_3d[:, temp_i_2]

        _, temp_s, temp_v = np.linalg.svd(gain, full_matrices=False)
        temp_s = np.diag(temp_s)  # if it not used, then delete it
        # Review sign of columns of V
        temp_v = np.transpose(-temp_v)

        gain_2d[:, temp_i_1] = gain @ temp_v[:, :2]

    return gain_2d


def compute_l1_minimisation(BB, G, bb_truncation, g_truncation, job_count):
    """Computes a fast L1 minimisation algorithm.

    Set up the minimisation problem of minimising L1 norm of x subject to AD=c
    use BB, G and only takes certain number of eigenvectors and eigenvalues;
    tolerance determines speed.

    Args:
        BB:
        Meg covariance (`numpy.ndarray`).
        G:
        Forward solution (`numpy.ndarray`).
        bb_truncation:
        Truncation value to use for MEG covariance matrix (`int` or `None`).
        g_truncation:
        Truncation value to use for Gain matrix (`int` or `None`).
    Returns:
        The solution matrix, RMS over the spatial modes (`numpy.ndarray`).
    """

    # First find eigenvalues and eigenvectors. We do this in main program so
    # should not be done again.
    Ub, sb, _ = np.linalg.svd(BB, full_matrices=False)  # `VbT` unused (see below)
    Ug, sg, VgT = np.linalg.svd(G, full_matrices=False)

    if bb_truncation is not None:
        # Get rank of matrices
        bb_rank = np.linalg.matrix_rank(BB)

        # No point in using truncation if it is smaller than rank and if values
        # of singular values are smaller then tolerance for that rank/tolerance
        if bb_rank < bb_truncation:
            bb_truncation = bb_rank

        Ub = Ub[:, :bb_truncation]
        sb = sb[:bb_truncation]
        # VbT = VbT[:bb_truncation, :]  # unused

    if g_truncation is not None:
        g_rank = np.linalg.matrix_rank(G)

        if g_rank < g_truncation:
            g_truncation = g_rank

        Ug = Ug[:, :g_truncation]
        sg = sg[:g_truncation]
        VgT = VgT[:g_truncation, :]

    # Create diagonal matrix and obtain sqrt
    Sb = np.diag(np.sqrt(sb))
    Sg = np.diag(sg)

    UgT = Ug.transpose()
    Vg = VgT.transpose()

    # From this construct A, D and c
    c = np.sqrt(np.diag(Vg @ VgT))
    UbSb = Ub @ Sb
    D = UgT @ UbSb
    A = Sg @ VgT
    # REVIEW: using joblib parallel
    # Call the relevant optimisation function in a multiprocessing way
    # print('Using %d jobs for L1 minimisation', job_count)
    # with multiprocessing.Pool(processes=job_count) as pool:
    #     # Iterate through columns of D to get solutions at each spatial mode
    #     solutions = pool.starmap(
    #         l1_cvxpy_l1,
    #         [
    #             (
    #                 A,
    #                 b,
    #                 c,
    #             )
    #             for b in D.T
    #         ],
    #     )
    solutions = Parallel(n_jobs=job_count)(delayed(l1_cvxpy_l1)(A, b, c) for b in D.T)
    # Construct matrix from list of solutions
    mat_solns = np.array(np.vstack(solutions))

    # Calculate root mean squared source amplitude,omit T (the number of epochs, but does not matter)
    return np.sqrt(np.diag(mat_solns.T @ mat_solns))


def compute_l1_minimisation_adjust(BB, G, bb_truncation, g_truncation, job_count):
    """Computes a fast L1 minimisation algorithm.

    Set up the minimisation problem of minimising L1 norm of x subject to AD=c
    use BB, G and only takes certain number of eigenvectors and eigenvalues;
    tolerance determines speed.

    Args:
        BB:
        Meg covariance (`numpy.ndarray`).
        G:
        Forward solution (`numpy.ndarray`).
        bb_truncation:
        Truncation value to use for MEG covariance matrix (`int` or `None`).
        g_truncation:
        Truncation value to use for Gain matrix (`int` or `None`).
    Returns:
        The solution matrix, RMS over the spatial modes (`numpy.ndarray`).
    """

    # First find eigenvalues and eigenvectors. We do this in main program so
    # should not be done again.
    Ub, sb, _ = np.linalg.svd(BB, full_matrices=False)  # `VbT` unused (see below)
    Ug, sg, VgT = np.linalg.svd(G, full_matrices=False)

    if bb_truncation is not None:
        # Get rank of matrices
        bb_rank = np.linalg.matrix_rank(BB)

        # No point in using truncation if it is smaller than rank and if values
        # of singular values are smaller then tolerance for that rank/tolerance
        if bb_rank < bb_truncation:
            bb_truncation = bb_rank

        Ub = Ub[:, :bb_truncation]
        sb = sb[:bb_truncation]
        # VbT = VbT[:bb_truncation, :]  # unused

    if g_truncation is not None:
        g_rank = np.linalg.matrix_rank(G)

        if g_rank < g_truncation:
            g_truncation = g_rank

        Ug = Ug[:, :g_truncation]
        sg = sg[:g_truncation]
        VgT = VgT[:g_truncation, :]

    # Create diagonal matrix and obtain sqrt
    Sb = np.diag(np.sqrt(sb))
    Sg = np.diag(sg)

    UgT = Ug.transpose()
    Vg = VgT.transpose()

    # From this construct A, D and c
    c = np.sqrt(np.diag(Vg @ VgT))
    UbSb = Ub @ Sb
    D = UgT @ UbSb
    A = Sg @ VgT
    # REVIEW: using joblib parallel
    # Call the relevant optimisation function in a multiprocessing way
    # print('Using %d jobs for L1 minimisation', job_count)
    # with multiprocessing.Pool(processes=job_count) as pool:
    #     # Iterate through columns of D to get solutions at each spatial mode
    #     solutions = pool.starmap(
    #         l1_cvxpy_l1,
    #         [
    #             (
    #                 A,
    #                 b,
    #                 c,
    #             )
    #             for b in D.T
    #         ],
    #     )
    solutions = Parallel(n_jobs=job_count)(
        delayed(l1_cvxpy_l1_adjust)(A, b, c) for b in D.T
    )
    # Construct matrix from list of solutions
    mat_solns = np.array(np.vstack(solutions))

    # Calculate root mean squared source amplitude,omit T (the number of epochs, but does not matter)
    return np.sqrt(np.diag(mat_solns.T @ mat_solns))


def l1_cvxpy_l1(A, b, c):
    """Args:
        A:
        numpy.ndarray`).
        b:
        `numpy.ndarray`).
        c:
        (`numpy.ndarray`).
    Returns:
        (`numpy.ndarray`).
    Raises:
        `cvxpy.error.SolverError` is raised by `cvxpy` if a solution cannot be
        found.
    """
    result = cp.Variable(A.shape[1])
    # equation 12 ||w^T*x|| s.t. A*x = b
    prob = cp.Problem(cp.Minimize(cp.norm(c.T @ result, 1)), [A @ result == b])
    prob.solve(solver=cp.ECOS, verbose=False)

    return result.value


def l1_cvxpy_l1_adjust(A, b, c):
    """Args:
        A:
        numpy.ndarray`).
        b:
        `numpy.ndarray`).
        c:
        (`numpy.ndarray`).
    Returns:
        (`numpy.ndarray`).
    Raises:
        `cvxpy.error.SolverError` is raised by `cvxpy` if a solution cannot be
        found.
    """
    result = cp.Variable(A.shape[1])
    # REVIEW: exactly following Huang's algorithm
    # prob = cp.Problem(cp.Minimize(c.T @ cp.abs(result)), [A @ result == b])
    # prob.solve(solver=cp.ECOS, verbose=False)
    # equation 12 ||w^T*x|| s.t. A*x = b
    prob = cp.Problem(cp.Minimize(cp.norm(c.T @ result, 1)), [A @ result == b])
    prob.solve(solver=cp.ECOS, verbose=False)

    return result.value


def vectorise(scalar_map, n_vertices):
    """TODO: #59 Needs a docstring.

    Args:
        scalar_map:

        n_vertices:

    Returns:
    """
    result = np.zeros([n_vertices, 1])
    for idx in range(n_vertices):
        result[idx, 0] = np.sqrt(
            scalar_map[2 * idx] * scalar_map[2 * idx]
            + scalar_map[2 * idx + 1] * scalar_map[2 * idx + 1]
        )
    return result


def whiten(covariance_matrix, input_matrix):
    """Whitens the data. See `WhiteningMatrices.md` for more information.

    Args:
        covariance_matrix:
        The matrix to use to define whitening (`numpy.ndarray`).
        input_matrix:
        `numpy.ndarray` to be whitened.

    Returns:
        A whitened `numpy.ndarray`.
    """

    u, s, _ = np.linalg.svd(covariance_matrix, full_matrices=False)

    principal_components = np.dot(np.dot(u, np.diag(1.0 / np.sqrt(s + 10e-8))), u.T)
    return np.dot(principal_components, input_matrix)


def gen_lcmv(active_cov, baseline_cov, filters):
    stc_base = apply_lcmv_cov(baseline_cov, filters)
    stc_act = apply_lcmv_cov(active_cov, filters)
    stc_act /= stc_base
    return stc_act


def rescale_and_find_cut_coords(img, scale_factor=1e2, threshold_offset=0.5):
    # Get the image data as a NumPy array
    data = img.get_fdata()

    # Rescale the data
    rescaled_data = data * scale_factor

    # Create a new NIfTI image with the rescaled data and the same affine as the original image
    rescaled_img = nib.Nifti1Image(rescaled_data, img.affine)

    # Find the maximum value in the rescaled data
    z_max = np.max(rescaled_data)

    # Find the cut coordinates
    coords = plotting.find_xyz_cut_coords(
        rescaled_img, activation_threshold=z_max - threshold_offset
    )

    return rescaled_img, coords, z_max
