"""Processing pipeline example for resting state fMRI datasets

   This example demonstrates using DiFuMo with 64 components
   and 50 subjects from ABIDE data downloaded using Nilearn.

   Note that this example can be adapted to any dataset and
   plugin more atlases. The pattern will be the same.
"""
import runpy
import os
import numpy as np

from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

from nilearn import datasets
from nilearn.input_data import NiftiMapsMasker
from nilearn import connectome

from confounds import extract_confounds

# DiFuMo atlases: https://parietal-inria.github.io/DiFuMo/
# Script to download DiFuMo atlases:
# https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/fetcher.py

# Load a file not on the path
fetcher = runpy.run_path('../fetcher.py')
fetch_difumo = fetcher['fetch_difumo']

###########################################################################
# Data
# ----
# Grab datasets

abide = datasets.fetch_abide_pcp(n_subjects=2)
func_imgs = abide['func_preproc']
phenotypes = abide.phenotypic

# Fetch grey matter mask from nilearn shipped with ICBM templates
gm_mask = datasets.fetch_icbm152_brain_gm_mask(threshold=0.2)
###########################################################################
# Predefined Atlases
# ------------------
# Fetch the atlas

maps_img = fetch_difumo(dimension=64).maps

##############################################################################
# Extract timeseries
# ------------------
mask_params = {'mask_img': gm_mask,
               'detrend': True, 'standardize': True,
               'high_pass': 0.01, 'low_pass': 0.1, 't_r': 2.53,
               'smoothing_fwhm': 6., 'verbose': 1}

masker = NiftiMapsMasker(maps_img=maps_img, **mask_params)
subjects_timeseries = []
dx_groups = []
for label, func_img in zip(phenotypes['DX_GROUP'], func_imgs):
    confounds = extract_confounds(func_img, mask_img=gm_mask,
                                  n_confounds=10)
    signals = masker.fit_transform(func_img, confounds=confounds)
    subjects_timeseries.append(signals)
    dx_groups.append(label)

##############################################################################
# Functional Connectomes
# ----------------------
connectome_measure = connectome.ConnectivityMeasure(
    cov_estimator=LedoitWolf(assume_centered=True),
    kind='tangent', vectorize=True)

# Vectorized connectomes across subject-specific timeseries
vec = connectome_measure.fit_transform(subjects_timeseries)

##############################################################################
# Linear model
# -------------
# Logistic Regression 'l2'
estimator = LogisticRegression(penalty='l2', random_state=0)
cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25,
                            random_state=0)
scores = cross_val_score(estimator, vec, dx_groups,
                         scoring='roc_auc', cv=cv)
