"""
Author      : Jie Li, Innovision IP Ltd., and School of Mathematics Statistics
				and Actuarial Science, University of Kent.
Date        : 2024-06-02 12:53:20
Last Edited : 2024-08-15 12:23:19
Last Author : Jie Li
File Path   : /undefined/Users/Jie/Documents/dcm_IP/dcm/defaults.py
Description :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""

import numpy as np

defaults = {
    "cmdline": 0,
    "ui_monitor": float("nan"),
    "ui_colour": np.array([0.58, 0.77, 0.57]),
    "ui_fs": 14,
    "ui_print": "ps",
    "renderer": "opengl",
    "images_format": "nii",
    "images_tol_orient": 1e-4,
    "mat_format": "-v6",
    "tbx_dir": None,  # This should be set to the actual directory
    "tbx_mb_data": None,  # This should be set to the actual directory
    "dicom_root": "flat",
    "stats_fmri_t": 16,
    "stats_fmri_t0": 8,
    "stats_fmri_hpf": 128,
    "stats_fmri_cvi": "AR(1)",
    "stats_fmri_hrf": np.array([6, 16, 1, 1, 6, 0, 32], dtype=np.int8),
    "mask_thresh": 0.8,
    "stats_maxmem": 2**30,
    "stats_maxres": 64,
    "stats_resmem": False,
    "stats_fmri_ufp": 0.001,
    "stats_pet_ufp": 0.05,
    "stats_eeg_ufp": 0.05,
    "stats_topoFDR": 1,
    "stats_rft_nonstat": 0,
    "stats_results_volume_distmin": 8,
    "stats_results_volume_nbmax": 3,
    "stats_results_svc_distmin": 4,
    "stats_results_svc_nbmax": 16,
    "stats_results_mipmat": None,  # This should be set to the actual directory
    "slicetiming_prefix": "a",
    "realign_write_prefix": "r",
    "coreg_write_prefix": "r",
    "unwarp_write_prefix": "u",
    "normalise_write_prefix": "w",
    "deformations_modulate_prefix": "m",
    "smooth_prefix": "s",
    "imcalc_prefix": "i",
    "realign_estimate_quality": 0.95,
    "realign_estimate_interp": 2,
    "realign_estimate_wrap": np.array([0, 0, 0]),
    "realign_estimate_sep": 1.5,
    "realign_estimate_fwhm": 1,
    "realign_estimate_rtm": 1,
    "realign_write_mask": 1,
    "realign_write_interp": 4,
    "realign_write_wrap": np.array([0, 0, 0]),
    "realign_write_which": np.array([2, 1]),
    "unwarp_estimate_fwhm": 2,
    "unwarp_estimate_basfcn": np.array([12, 12]),
    "unwarp_estimate_regorder": 1,
    "unwarp_estimate_regwgt": 1e5,
    "unwarp_estimate_foe": np.array([4, 5]),
    "unwarp_estimate_soe": np.array([]),
    "unwarp_estimate_rem": 1,
    "unwarp_estimate_noi": 5,
    "unwarp_estimate_expround": "Average",
    "unwarp_write_jm": 1,
    "coreg_estimate_cost_fun": "nmi",
    "coreg_estimate_sep": np.array([4, 2]),
    "coreg_estimate_tol": np.array(
        [0.02, 0.02, 0.02, 0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001]
    ),
    "coreg_estimate_fwhm": np.array([7, 7]),
    "coreg_write_interp": 4,
    "coreg_write_wrap": np.array([0, 0, 0]),
    "coreg_write_mask": 0,
    "normalise_write_preserve": 0,
    "normalise_write_bb": np.array([[-78, -112, -70], [78, 76, 85]]),
    "normalise_write_vox": np.array([2, 2, 2]),
    "normalise_write_interp": 4,
    "old_normalise_estimate_smosrc": 8,
    "old_normalise_estimate_smoref": 0,
    "old_normalise_estimate_regtype": "mni",
    "old_normalise_estimate_weight": "",
    "old_normalise_estimate_cutoff": 25,
    "old_normalise_estimate_nits": 16,
    "old_normalise_estimate_reg": 1,
    "old_normalise_write_preserve": 0,
    "old_normalise_write_bb": np.array([[-78, -112, -70], [78, 76, 85]]),
    "old_normalise_write_vox": np.array([2, 2, 2]),
    "old_normalise_write_interp": 1,
    "old_normalise_write_wrap": np.array([0, 0, 0]),
    "old_normalise_write_prefix": "w",
    "old_preproc_tpm": None,  # This should be set to the actual directory
    "old_preproc_ngaus": np.array([2, 2, 2, 4]),
    "old_preproc_warpreg": 1,
    "old_preproc_warpco": 25,
    "old_preproc_biasreg": 0.0001,
    "old_preproc_biasfwhm": 60,
    "old_preproc_regtype": "mni",
    "old_preproc_fudge": 5,
    "old_preproc_samp": 3,
    "old_preproc_output_GM": np.array([0, 0, 1]),
    "old_preproc_output_WM": np.array([0, 0, 1]),
    "old_preproc_output_CSF": np.array([0, 0, 0]),
    "old_preproc_output_biascor": 1,
    "old_preproc_output_cleanup": 0,
    "smooth_fwhm": np.array([8, 8, 8]),
    "dcm_verbose": True,
}
