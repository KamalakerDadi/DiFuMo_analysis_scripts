"""
This script aims at performing an encoding study
for the RSVP language protocol of IBC.

to get ibc_public: install git@github.com:hbp-brain-charting/public_analysis_code.git

Author: Bertrand Thirion, Arthur Mensch, kamalaker Dadi
"""
import warnings

from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from utils.encoding import fit_second_level, fit_first_level, make_dmtx

warnings.filterwarnings('ignore', category=DeprecationWarning)

import glob
import os
from os.path import expanduser, join

import ibc_public.utils_data
from joblib import Memory, delayed, Parallel
from nibabel import Nifti1Image

import numpy as np


def get_data():
    _package_directory = os.path.dirname(
        os.path.abspath(ibc_public.utils_data.__file__))
    subject_session = ibc_public.utils_data.get_subject_session(
        'rsvp-language')
    ibc = '/storage/store/data/ibc/derivatives/'
    data = []
    for subject, session in subject_session:
        # fetch the data
        data_path = os.path.join(ibc, subject, session, 'func')
        imgs = sorted(
            glob.glob(os.path.join(data_path, 'w*RSVPLanguage*')))
        n_scans = [Nifti1Image.load(img).shape[3] for img in imgs]

        confounds = sorted(
            glob.glob(os.path.join(data_path, 'rp_*RSVPLanguage*')))
        events = sorted(glob.glob(os.path.join(data_path,
                                               '*RSVPLanguage*_events.tsv')))
        dmtxs = [make_dmtx(these_events, n_scans=these_scans, t_r=2)
                 for these_events, these_scans in
                 zip(events, n_scans)]
        data.append((imgs, confounds, dmtxs))
    return data


def perturbate_data(data, n_splits=10, random_state=None):
    random_state = check_random_state(random_state)
    random_states = random_state.randint(0, np.iinfo('uint32').max, n_splits)
    perturbed_data = [[] for _ in range(n_splits)]

    for random_state, (imgs, confounds, dmtxs) in zip(random_states, data):
        cv = ShuffleSplit(n_splits=n_splits, train_size=0.5, test_size=0.5,
                          random_state=random_state)
        for i, (train, test) in enumerate(cv.split(imgs)):
            perturbed_data[i].append(([imgs[j] for j in train],
                                      [confounds[j] for j in train],
                                      [dmtxs[j] for j in train]))
    return perturbed_data


# %% Loading files
n_jobs = 20
write_dir = expanduser('output/ibc/language4')


atlases = ['none', 'difumo']
dimensions = {'none': ['none'], 'difumo': [64]}

mem = Memory(location=expanduser('cache'))

subject_session = ibc_public.utils_data.get_subject_session(
    'rsvp-language')

full_data = get_data()
contrasts = np.array(full_data[0][2][0].columns.values)
perturbed_data = [full_data]
perturbed_data = [full_data] + perturbate_data(full_data, n_splits=10, random_state=0)

if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# %% Computing mask
print('Computing mask')
# obtain a grey matter mask
# _package_directory = os.path.dirname(os.path.abspath(ibc_public.utils_data.__file__))
# mask_gm = join(_package_directory, '../ibc_data', 'gm_mask_1_5mm.nii.gz')
# maps_dir = '/storage/store/derivatives/OpenNeuro/modl/'
# mask_modl = join(maps_dir, 'mask.nii.gz')
# mask_gm = resample_to_img(mask_gm, mask_modl, interpolation='nearest')
# mask = intersect_masks([mask_modl, mask_gm])
# mask.to_filename(join(write_dir, 'mask_ibc.nii.gz'))
mask = join(write_dir, 'mask_ibc.nii.gz')

for split, data in enumerate(perturbed_data):
    # %% Fitting first level models
    print('Fitting first level models')
    models1 = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_first_level)(imgs, confounds, dmtxs, mask, atlas, dim)
        for atlas in atlases
        for dim in dimensions[atlas]
        for imgs, confounds, dmtxs in data)

    # Unroll results
    models1_ = []
    for atlas in atlases:
        for dim in dimensions[atlas]:
            these_models = []
            for _ in data:
                # First level models
                this_model, models1 = models1[0], models1[1:]
                these_models.append(this_model)
            models1_.append((atlas, dim, these_models))

    # %% Fitting second level models

    contrasts = [
        'complex',
        'jabberwocky',
        'word_list',
        'consonant_strings',
        'simple',
        'pseudoword_list',
        'probe',
    ]
    print('Fitting second level model')
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(fit_second_level)(these_models, contrasts, atlas, dim, split,
                                  write_dir)
        for atlas, dim, these_models in models1_)
