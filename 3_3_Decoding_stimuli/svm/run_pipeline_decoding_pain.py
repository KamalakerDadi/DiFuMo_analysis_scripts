"""Decoding pain without and with reduction using DiFuMo of 64 components

   Note that this script can be expanded to various multi-scale atlases
   resolutions if one wants.
"""
import runpy
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_val_score, GroupShuffleSplit,
                                     GridSearchCV)
from sklearn.metrics import r2_score
from sklearn.svm import SVC

from nilearn.input_data import MultiNiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import load_img
from nilearn.datasets import fetch_neurovault
from nilearn.input_data import NiftiMapsMasker
from nilearn._utils import check_niimg

# DiFuMo atlases: https://parietal-inria.github.io/DiFuMo/
# Script to download DiFuMo atlases:
# https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/fetcher.py

# Load a file not on the path
fetcher = runpy.run_path('../../fetcher.py')
fetch_difumo = fetcher['fetch_difumo']

####################################################################
# Fetch statistical maps from Neurovault repository
# -------------------------------------------------

collection_terms = {'id': 504}
image_terms = {'not_mni': False}
pain_data = fetch_neurovault(max_images=None, image_terms=image_terms,
                             collection_terms=collection_terms)
n_images = len(pain_data.images)
ref_img = load_img(pain_data.images[0])

input_images = []
y = []
groups = []
for index in range(n_images):
    input_images.append(pain_data.images[index])
    target = pain_data.images_meta[index]['PainLevel']
    subject_id = pain_data.images_meta[index]['SubjectID']
    y.append(target)
    groups.append(subject_id)
y = np.ravel(y)
groups = np.ravel(groups)
######################################################################
# Grid search 'C' for prediction
# ---------------------------------

X_sc = StandardScaler()
estimator = SVC(kernel='linear')
param_grid = {'C': np.logspace(-3., 3., 10)}
# Grid search 'C' in linear SVC
cv = GroupShuffleSplit(n_splits=10, test_size=0.3, random_state=0)

######################################################################
# Predictive model - without reduction
# ------------------------------------

# Standard mask
icbm = fetch_icbm152_2009()
mask = icbm.mask

masker = MultiNiftiMasker(mask_img=mask, target_affine=ref_img.affine,
                          target_shape=ref_img.shape,
                          n_jobs=5, verbose=1)
X = masker.fit_transform(input_images)
X = np.vstack(X)
X = X_sc.fit_transform(X)

grid = GridSearchCV(estimator, param_grid=param_grid,
                    cv=cv.split(X, y, groups),
                    verbose=1, n_jobs=5)
# Prediction without reduction
grid.fit(X, y)
C = grid.best_params_['C']

svc = SVC(kernel='linear', C=C)
model_cv = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
non_reduced_scores = cross_val_score(estimator=svc, X=X, y=y, groups=groups,
                                     cv=model_cv, n_jobs=1, verbose=1)
######################################################################
# Predictive model - reduced with atlases
# ----------------------------------------

atlases = ['difumo']
dimensions = {'difumo': [64]}

model_cv = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
reduced = []
for atlas in atlases:
    this_atlas_dimensions = dimensions[atlas]
    for dim in this_atlas_dimensions:
        this_reports = []
        if atlas == 'difumo':
            maps_img = fetch_difumo(dimension=dim).maps
            masker = NiftiMapsMasker(maps_img=maps_img,
                                     resampling_target='data', verbose=1)
        X_reduced = masker.fit_transform(input_images)
        X_reduced = X_sc.fit_transform(X_reduced)
        # Grid search parameter 'C'
        grid = GridSearchCV(estimator, param_grid=param_grid,
                            cv=cv.split(X_reduced, y, groups),
                            verbose=1, n_jobs=1)
        grid.fit(X_reduced, y)
        C = grid.best_params_['C']
        svc = SVC(kernel='linear', C=C)
        reduced_scores = cross_val_score(estimator=svc, X=X_reduced,
                                         y=y, groups=groups, cv=model_cv,
                                         n_jobs=1, verbose=1)
        this_reduced = pd.DataFrame(reduced_scores, columns=['Scores'])
        this_reduced['Dimension'] = pd.Series([dim] * len(this_reduced),
                                              index=this_reduced.index)
        this_reduced['Atlas'] = pd.Series([atlas] * len(this_reduced),
                                          index=this_reduced.index)
        reduced.append(this_reduced)
reduced_scores = pd.concat(reduced)
