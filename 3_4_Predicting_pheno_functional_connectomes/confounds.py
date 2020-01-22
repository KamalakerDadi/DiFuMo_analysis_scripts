"""Module for confounds extraction
"""
import collections
import numpy as np

from nilearn import _utils
from nilearn.image import high_variance_confounds, resample_img
from nilearn._utils.compat import _basestring
from nilearn._utils.niimg_conversions import _check_same_fov
from nilearn._utils.numpy_conversions import csv_to_array


def extract_confounds(imgs, mask_img, n_confounds=10):
    """To extract confounds on list of subjects

    See nilearn.image.high_variance_confounds for technical details.

    Parameters
    ----------
    imgs : list of Nifti images, either str or nibabel.Nifti1Image
        Functional images on which confounds should be extracted

    mask_img : str or nibabel.Nifti1Image
        Mask image with binary values. This image is imposed on
        each functional image for confounds extraction..

    n_confounds : int, optional
        By default, 10 high variance confounds are extracted. Otherwise,
        confounds as specified are extracted.

    Returns
    -------
    confounds : list of Numpy arrays.
        Each numpy array is a confound corresponding to imgs provided.
        Each numpy array will have shape (n_timepoints, n_confounds)        
    """
    confounds = []
    if not isinstance(imgs, collections.Iterable) or \
            isinstance(imgs, _basestring):
        imgs = [imgs, ]

    img = _utils.check_niimg_4d(imgs[0])
    shape = img.shape[:3]
    affine = img.affine

    if isinstance(mask_img, _basestring):
        mask_img = _utils.check_niimg_3d(mask_img)

    if not _check_same_fov(img, mask_img):
        mask_img = resample_img(
            mask_img, target_shape=shape, target_affine=affine,
            interpolation='nearest')

    for img in imgs:
        print("[Confounds Extraction] Image selected {0}".format(img))
        img = _utils.check_niimg_4d(img)
        print("Extracting high variance confounds")
        high_variance = high_variance_confounds(img, mask_img=mask_img,
                                                n_confounds=n_confounds)
        confounds.append(high_variance)
    return confounds
