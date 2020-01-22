"""Decoding emotion without and with reduction using DiFuMo of 64 components

   Note that this script can be expanded to various multi-scale atlases
   resolutions if one wants.
"""
import runpy
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

from nilearn.input_data import MultiNiftiMasker
from nilearn.datasets import fetch_icbm152_2009
from nilearn.image import load_img, resample_to_img
from nilearn.input_data import NiftiMapsMasker
from nilearn.datasets import fetch_neurovault
from nilearn._utils.niimg_conversions import _check_same_fov

# DiFuMo atlases: https://parietal-inria.github.io/DiFuMo/
# Script to download DiFuMo atlases:
# https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/fetcher.py

# Load a file not on the path
fetcher = runpy.run_path('../../fetcher.py')
fetch_difumo = fetcher['fetch_difumo']
####################################################################
# Fetch statistical maps from Neurovault repository
# -------------------------------------------------

collection_terms = {'id': 503}
image_terms = {'not_mni': False}
emotion_data = fetch_neurovault(max_images=None, image_terms=image_terms,
                                collection_terms=collection_terms)
n_images = len(emotion_data.images)

input_images = []
y = []
groups = []
ref_img = load_img(emotion_data.images[0])
for index in range(n_images):
    img = emotion_data.images[index]
    if not _check_same_fov(ref_img, load_img(img)):
        img = resample_to_img(source_img=img, target_img=ref_img)
    input_images.append(img)
    target = emotion_data.images_meta[index]['Rating']
    subject_id = emotion_data.images_meta[index]['SubjectID']
    y.append(target)
    groups.append(subject_id)
y = np.ravel(y)
groups = np.ravel(groups)
######################################################################
# Grid search 'alpha' for prediction
# ---------------------------------

X_sc = StandardScaler()
# RidgeCV
ridge = RidgeCV(alphas=np.logspace(-3., 3., 10))
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

model_cv = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
r2_scores = []
for split, (train_index, test_index) in enumerate(model_cv.split(X, y, groups)):
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    r2_scores.append(r2_score(y_test, y_pred))
non_reduced_scores = pd.DataFrame(r2_scores, columns=['R2 score'])
######################################################################
# Predictive model - reduced with atlases
# ---------------------------------------

atlases = ['difumo']
dimensions = {'difumo': [64]}

model_cv = GroupShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
reports = []
for atlas in atlases:
    this_atlas_dimensions = dimensions[atlas]
    for dim in this_atlas_dimensions:
        this_reports = []
        if atlas == 'difumo':
            maps_img = fetch_difumo(dimension=dim).maps
            masker = NiftiMapsMasker(maps_img=maps_img,
                                     resampling_target='data', verbose=1)
        X_reduced = masker.fit_transform(input_images)
        # Standardize data
        X_reduced = X_sc.fit_transform(X_reduced)
        r2_scores = []
        for split, (train_index, test_index) in enumerate(model_cv.split(X_reduced, y, groups)):
            X_train, y_train = X_reduced[train_index], y[train_index]
            X_test, y_test = X_reduced[test_index], y[test_index]
            ridge.fit(X_train, y_train)
            y_pred = ridge.predict(X_test)
            r2_scores.append(r2_score(y_test, y_pred))
        this_reports.append(pd.DataFrame(r2_scores, columns=['R2 score']))
        this_reports = pd.concat(this_reports, axis=1)
        this_reports['Dimension'] = pd.Series([dim] * len(this_reports),
                                              index=this_reports.index)
        this_reports['Atlas'] = pd.Series([atlas] * len(this_reports),
                                          index=this_reports.index)
        reports.append(this_reports)
reduced_scores = pd.concat(reports)
