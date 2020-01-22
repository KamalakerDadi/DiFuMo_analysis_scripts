import warnings
import runpy

warnings.filterwarnings('ignore', category=DeprecationWarning)

from nibabel import Nifti1Image
from nilearn._utils import check_niimg

from os.path import expanduser, join

import numpy as np
from joblib import Memory
from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from nistats.design_matrix import make_first_level_design_matrix
from nistats.first_level_model import FirstLevelModel
from nistats.second_level_model import SecondLevelModel
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut

# DiFuMo atlases: https://parietal-inria.github.io/DiFuMo/
# Script to download DiFuMo atlases:
# https://github.com/Parietal-INRIA/DiFuMo/blob/master/notebook/fetcher.py

# Load a file not on the path
fetcher = runpy.run_path('../../fetcher.py')
fetch_difumo = fetcher['fetch_difumo']


def label_to_maps(label_img):
    label_img = check_niimg(label_img, ensure_ndim=3)
    shape = label_img.shape
    data = label_img.get_data()
    affine = label_img.get_affine()
    n_components = int(data.max())
    img_data = np.zeros((*shape, n_components))
    for i in range(1, n_components + 1):
        img_data[:, :, :, i - 1] = data == i
    return Nifti1Image(img_data, affine)


class OurNiftiMasker(NiftiMasker):
    def __init__(self, maps_img=None, label_img=None,
                 ensure_finite=True,
                 reduction=None, mask_img=None, sessions=None,
                 smoothing_fwhm=None,
                 standardize=False, detrend=False,
                 low_pass=None, high_pass=None, t_r=None,
                 target_affine=None, target_shape=None,
                 mask_strategy='background',
                 mask_args=None, sample_mask=None,
                 memory_level=1, memory=Memory(cachedir=None),
                 verbose=0):
        assert reduction in [None, 'maps', 'labels']
        self.reduction = reduction
        self.maps_img = maps_img
        self.label_img = label_img
        self.ensure_finite = ensure_finite
        super().__init__(mask_img=mask_img, sessions=sessions,
                         smoothing_fwhm=smoothing_fwhm,
                         standardize=standardize, detrend=detrend,
                         low_pass=low_pass, high_pass=high_pass, t_r=t_r,
                         target_affine=target_affine,
                         target_shape=target_shape,
                         mask_strategy=mask_strategy,
                         mask_args=mask_args, sample_mask=sample_mask,
                         memory_level=memory_level, memory=memory,
                         verbose=verbose)

    def fit(self, imgs=None, y=None):
        super().fit(imgs, y)
        if self.reduction is not None:
            if self.reduction == 'labels':
                self.maps_img_ = self.memory.cache(label_to_maps)(
                    self.label_img)
            else:
                self.maps_img_ = self.maps_img
            simple_masker = NiftiMasker(mask_img=self.mask_img_).fit()
            self._masked_maps_ = simple_masker.transform_single_imgs(
                self.maps_img_)
            self._masked_maps_inv_ = np.linalg.pinv(self._masked_maps_)
        return self

    def transform_single_imgs(self, imgs, confounds=None, copy=True):
        imgs = check_niimg(imgs)
        data = imgs.get_data()
        data[np.logical_not(np.isfinite(data))] = 0
        imgs = Nifti1Image(data, imgs.affine)
        masked_imgs = super().transform_single_imgs(imgs, confounds, copy)
        if self.reduction is None:
            return masked_imgs
        else:
            return masked_imgs.dot(self._masked_maps_inv_)

    def inverse_transform(self, X):
        if self.reduction is not None:
            X = X.dot(self._masked_maps_)
        return super().inverse_transform(X)


def make_masker(atlas, dim, mask):
    mem = Memory(location=expanduser('cache'))
    masker_params = dict(mask_img=mask,
                         smoothing_fwhm=5, memory=mem, memory_level=1,
                         standardize=True, detrend=True, t_r=2,
                         low_pass=0.01, high_pass=None,
                         verbose=0, )
    if atlas in ['difumo']:
        if atlas == 'difumo':
            maps_img = fetch_difumo(dimension=dim).maps
        masker = OurNiftiMasker(maps_img=maps_img,
                                reduction='maps', **masker_params)
    elif atlas in ['basc']:
        if atlas == 'basc':
            basc = fetch_atlas_basc_multiscale_2015()
            this_labels_img = basc[f'scale{dim:03.0f}']
        masker = OurNiftiMasker(label_img=this_labels_img,
                                reduction='labels', **masker_params)
    elif atlas == 'none':
        masker = OurNiftiMasker(reduction=None, **masker_params)
    else:
        raise ValueError('Wrong atlas argument')
    masker.fit()
    masker_geom = NiftiMasker(mask_img=mask).fit()
    return masker, masker_geom


def fit_first_level(imgs, confounds, dmtxs, mask, atlas, dim):
    mem = Memory(location=expanduser('cache'))
    masker, masker_geom = make_masker(atlas, dim, mask)
    model = FirstLevelModel(mask=masker, t_r=2., memory=mem, memory_level=1,
                            period_cut=128, signal_scaling=False)
    model.fit(imgs, confounds=confounds, design_matrices=dmtxs)
    return model


def fit_second_level(models1, contrasts, atlas, dim, split, write_dir):
    mem = Memory(location=expanduser('cache'))
    model = SecondLevelModel(n_jobs=1, memory=mem, memory_level=1)
    model.fit(models1)

    for contrast in contrasts:
        for output_type in ['z_score', 'effect_size']:
            img = model.compute_contrast(
                first_level_contrast=contrast,
                output_type=output_type)
            img.to_filename(join(write_dir,
                                 f'{contrast}_{output_type}_{atlas}'
                                 f'_{dim}_{split}.nii.gz'))


def loo_encoding(data, mask, atlas, dim, subject, session, write_dir):
    # set up a masker
    masker, masker_geom = make_masker(atlas, dim, mask)
    masker_voxel, _ = make_masker('none', 'none', mask)

    Y = []
    X = []
    beta = []
    imgs = data[0]
    confounds = data[1]
    dmtxs = data[2]
    for i, (img, confound, dmtx) in enumerate(zip(imgs, confounds, dmtxs)):
        this_Y = masker_voxel.transform(img, confounds=confound)
        if atlas != 'none':
            this_Y_red = masker.transform(img, confound)
        else:
            this_Y_red = this_Y
        this_beta = np.linalg.lstsq(dmtx, this_Y_red, rcond=None)[0]
        if atlas != 'none':
            this_beta = masker_geom.transform(
                masker.inverse_transform(this_beta))
        Y.append(this_Y)
        X.append(dmtx)
        beta.append(this_beta)

    print('Making X')
    X = np.concatenate([np.array(this_X)[None, :, :] for this_X in X], axis=0)
    print('Making Y')
    Y = np.concatenate([this_Y[None, :, :] for this_Y in Y], axis=0)
    print('Making beta')
    beta = np.concatenate([this_beta[None, :, :] for this_beta in beta],
                          axis=0)

    print('Computing r2 scores subject %s session %s '
          'atlas %s dim %s' % (subject, session, atlas, dim))
    loo = LeaveOneOut()
    r2 = []
    for train_index, test_index in loo.split(X):
        test_index = test_index[0]
        beta_predicted = beta[train_index].mean(axis=0)
        Y_predicted = np.dot(X[test_index], beta_predicted)
        fold_r2 = r2_score(Y[test_index], Y_predicted,
                           multioutput='raw_values')
        r2.append(fold_r2)
    r2_img = masker_geom.inverse_transform(np.array(r2).mean(0, keepdims=True))
    filename = join(write_dir, 'r2_img_%s_%s_%s_%s.nii.gz' % (
        subject, session, atlas, dim))
    r2_img.to_filename(filename)


def make_dmtx(events, n_scans, t_r=2.):
    from pandas import read_csv
    frame_times = np.arange(n_scans) * t_r
    events = read_csv(events, sep='\t')
    complexs = ['complex_sentence_objclef', 'complex_sentence_objrel',
                'complex_sentence_subjrel']
    simples = ['simple_sentence_adj', 'simple_sentence_coord',
               'simple_sentence_cvp']
    for complex_ in complexs:
        events = events.replace(complex_, 'complex')
    for simple_ in simples:
        events = events.replace(simple_, 'simple')
    dmtx = make_first_level_design_matrix(
        frame_times, events=events, hrf_model='spm',
        drift_model=None)
    dmtx.drop(columns='constant', inplace=True)
    return dmtx  # remove the constant regressor
