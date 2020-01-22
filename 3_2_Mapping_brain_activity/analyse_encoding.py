import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import re
import pandas as pd

import glob
from os.path import expanduser, join

from joblib import delayed, Parallel
from nilearn.input_data import NiftiMasker
from nistats.thresholding import map_threshold, fdr_threshold

import numpy as np


def dice_index(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0

    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1 = im1 != 0
    im2 = im2 != 0

    if im1.sum() == 0 and im2.sum() == 0:
        return 1.

    if im1.shape != im2.shape:
        raise ValueError(
            "Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def threshold_and_save(img, masker):
    data = masker.transform(img)
    thr = fdr_threshold(data, 0.05)
    data[np.abs(data) <= thr] = 0
    thr_img = masker.inverse_transform(data)
    thr_img.to_filename(img.replace('.nii.gz', '_thr.nii.gz'))


def compute_thresholded_maps():
    output_dir = expanduser('output/ibc/test_reductions/language44')

    regex = re.compile(r'(?P<contrast>.*)_z_score_(?P<reduction>.*)_'
                       r'(?P<size>.*)_(?P<split>[1-9]*).nii.gz')

    imgs = sorted(filter(regex.match, os.listdir(output_dir)))

    mask = join(output_dir, 'mask_ibc.nii.gz')
    masker = NiftiMasker(mask_img=mask).fit()

    Parallel(n_jobs=5, backend='multiprocessing', verbose=10)(
        delayed(threshold_and_save)(join(output_dir, img), masker) for img in
        imgs)


def compute_dices():
    output_dir = expanduser('output/ibc/language44')

    mask = join(output_dir, 'mask_ibc.nii.gz')
    masker = NiftiMasker(mask_img=mask).fit()

    imgs = sorted(glob.glob(join(output_dir, '*_thr.nii.gz')))

    regex = re.compile(r'(?P<contrast>.*)_z_score_(?P<reduction>.*)_'
                       r'(?P<size>.*)_(?P<split>[1-9]*)_thr.nii.gz')

    df = []
    for img in imgs:
        _, filename = os.path.split(img)
        match = regex.match(filename)
        if match:
            contrast = match['contrast']
            reduction = match['reduction']
            size = int(match['size']) if match['size'] != 'none' else 'none'
            split = int(match['split'])
            df.append(dict(contrast=contrast, reduction=reduction, size=size,
                           split=split, filename=img))
    df = pd.DataFrame(df)

    intra_dices = []
    ref_df = df.loc[df['reduction'] == 'none']

    for contrast, sub_df in ref_df.groupby(by='contrast'):
        n = sub_df.shape[0]
        components = masker.transform(sub_df['filename']) != 0
        dices = np.array([dice_index(components[i], components[j])
                          for i in range(n - 1) for j in range(i + 1, n)])
        print(contrast, dices)
        intra_dices.append(dict(contrast=contrast, mean_dice=np.mean(dices),
                                std_dice=np.std(dices)))

    intra_dices = pd.DataFrame(intra_dices)
    stop
    intra_dices.to_pickle(join(output_dir, 'intra_dices.pkl'))

    inter_dices = []
    for (contrast, reduction, size), sub_df in df.groupby(by=['contrast',
                                                              'reduction',
                                                              'size']):
        sub_ref_df = ref_df.loc[ref_df['contrast'] == contrast]
        n = sub_df.shape[0]
        components = masker.transform(sub_df['filename']) != 0
        ref_components = masker.transform(sub_ref_df['filename']) != 0
        dices = np.array([dice_index(components[i], ref_components[i])
                          for i in range(n)])
        print(contrast, reduction, size, dices)
        inter_dices.append(
            dict(contrast=contrast, reduction=reduction, size=size,
                 mean_dice=np.mean(dices),
                 std_dice=np.std(dices)))

    inter_dices = pd.DataFrame(inter_dices)
    inter_dices.to_pickle(join(output_dir, 'inter_dices.pkl'))


def show_dices():
    output_dir = expanduser('~/output/ibc/language4')

    intra_dices = pd.read_pickle(join(output_dir, 'intra_dices.pkl'))
    # type: pd.DataFrame
    inter_dices = pd.read_pickle(join(output_dir, 'inter_dices.pkl'))
    # type: pd.DataFrame

    inter_dices.set_index(['reduction', 'size', 'contrast'], inplace=True)
    inter_dices.sort_index(inplace=True)
    intra_dices.set_index(['contrast'], inplace=True)
    intra_dices.sort_index(inplace=True)

    trans_dict = {'mean_dice': np.mean,
                  'std_dice': lambda
                      x: np.sqrt(
                      np.mean(x ** 2))}
    global_intra = intra_dices.agg(trans_dict).iloc[0]
    global_inter = inter_dices.groupby(['reduction', 'size']).agg(trans_dict)

    global_inter.reset_index('size', inplace=True)
    global_inter.drop('none', axis=0, inplace=True)

    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    fig.subplots_adjust(left=0.18, top=0.97, right=.99, bottom=0.15)
    labels = {'craddock': 'Craddock clusters', 'basc': 'BASC clusters',
              'modl': 'SOMF atlases'}

    for reduction, this_inter in global_inter.groupby('reduction'):
        ax.errorbar(this_inter['size'], this_inter['mean_dice'],
                    yerr=this_inter['std_dice'], zorder=10, label=labels[reduction],
                    marker='o', elinewidth=2, linewidth=3, capsize=2
                    )
    ax.hlines(global_intra['mean_dice'], 64,
              1024, zorder=2, color='black', linewidth=3, label='Inter-fold level\nw/out reduction')

    ax.fill_between([64,
                     1024],
                    [global_intra['mean_dice'] + global_intra['std_dice']] * 2,
                    [global_intra['mean_dice'] - global_intra['std_dice']] * 2,
                    alpha=0.2, zorder=1, color='black'
                    )
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlabel('Atlas size')
    ax.set_xticks([64, 150, 256, 512, 1024])
    ax.set_ylabel('Mean dice index \n w.r.t voxel-level maps')
    ax.set_ylim([0, 0.82])
    sns.despine(fig)
    plt.savefig(expanduser('~/output/ibc/glm_dice.pdf'))


# show_dices()
compute_dices()
#compute_thresholded_maps()
