import os
import re
from os.path import expanduser, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nilearn.input_data import MultiNiftiMasker
from nilearn.plotting import plot_stat_map

import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.utils import check_random_state

write_dir = expanduser('output/ibc/language3')
mask_gm = join(write_dir, 'mask_ibc.nii.gz')

regex = r'r2_img_sub-([0-9]+)_ses-([0-9]+)_([a-z]+)_([0-9]+|none)\.nii\.gz'
indices = []
filenames = []
for filename in os.listdir(write_dir):
    match = re.match(regex, filename)
    if match:
        group = match.groups()
        sub, ses, atlas, dim = int(group[0]), int(group[1]), str(group[2]), \
                               group[3]
        if dim != 'none':
            dim = int(dim)
        indices.append((atlas, dim, sub, ses))
        filenames.append(join(write_dir, filename))
records = pd.DataFrame(
    index=pd.MultiIndex.from_tuples(indices, names=['atlas',
                                                    'dimension',
                                                    'subject',
                                                    'session',
                                                    ]),
    data=filenames, columns=['filename'])
records.sort_index(inplace=True)

masker = MultiNiftiMasker(mask_img=mask_gm, n_jobs=1).fit()
r2s = masker.transform(records['filename'].values)
r2s = np.concatenate(r2s, axis=0)
r2s[r2s < 0] = 0
r2s = pd.DataFrame(data=r2s, index=records.index)
r2s.sort_index(inplace=True)
ref = r2s.loc['none', 'none'].values.ravel()[:, None]

imgs = [masker.inverse_transform(r2) for _, r2 in r2s.iterrows()]
records['nifti'] = imgs

subject, session = 4, 3
imgs_to_plot = records['nifti'].loc[pd.IndexSlice[:, :, subject, session]]
cut_coords = -55, -38, 2
vmax = r2s.loc[pd.IndexSlice[:, :, subject, session], :].values.max()

fig, axes = plt.subplots(len(imgs_to_plot), 2, figsize=(12, 28),
                         gridspec_kw={'width_ratios': [1, 2]})
voxels = check_random_state(0).permutation(ref.ravel().shape[0])[:1000]

for i, (index, img) in enumerate(imgs_to_plot.items()):
    r2 = r2s.loc[index].values.ravel()[:, None]
    lr = LinearRegression(fit_intercept=False)
    lr.fit(ref, r2)
    r2_pred = lr.predict(ref)
    score = r2_score(r2, r2_pred)

    ax = axes[i, 0]
    ax.scatter(ref[voxels, 0], r2[voxels, 0], marker='+', zorder=10)
    ax.set_xlim([0, 0.05])
    ax.set_ylim([0, 0.05])
    ax.annotate(('%s\n' + r'$\beta = %.2f$' + '\n' + r'$r^2 = %.2f$')
                % (index, lr.coef_[0, 0], score),
                xy=(0.5, 0.8), xycoords='axes fraction', ha='center',
                va='center')
    ax.plot(np.linspace(0, 0.1, 10), np.linspace(0, 0.1, 10), linestyle='--',
            color='r', zorder=1)
    sns.despine(fig, ax)
    ax = axes[i, 1]
    plot_stat_map(img, figure=fig, axes=ax, vmax=vmax,
                  colorbar=i == 0,
                  cut_coords=cut_coords)
axes[0, 0].set_xlabel('R2 score from voxelwise decoding')
axes[0, 0].xaxis.set_label_coords(0.5, 1.2)
axes[0, 0].set_ylabel('R2 score from map decoding')

plt.savefig(expanduser('~/output/ibc/encoding.pdf'))
