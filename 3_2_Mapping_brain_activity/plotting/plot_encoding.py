import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import copy
import pandas as pd

from os.path import expanduser

import numpy as np

# type: pd.DataFrame
intra_dices = pd.read_pickle('glm_dice_data/intra_dices.pkl')
# type: pd.DataFrame
inter_dices = pd.read_pickle('glm_dice_data/inter_dices.pkl')
inter_dices = inter_dices.replace(to_replace={'reduction': {'shirer': 'find',
                                                            'willard': 'find'}})

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

sns.set_style("whitegrid")

fig, ax = plt.subplots(1, 1, figsize=(3.8, 3.8))
fig.subplots_adjust(left=0.18, top=0.97, right=.99, bottom=0.15)
labels = {'craddock': 'Craddock', 'basc': 'BASC',
          'modl': 'SOMF', 'melodic_ica': 'UKBB ICA',
          'find': 'FIND', 'gordon': 'Gordon', 'schaefer': 'Schaefer'}
markers = {'melodic_ica': 'v',
           'modl': '^',
           'basc': '>',
           'craddock': '<',
           'find': 's',
           'gordon': '8',
           'schaefer': 'p'}
colors = {'melodic_ica': 'c', 'modl': 'm', 'basc': 'y', 'craddock': 'g',
          'find': 'b', 'gordon': 'purple', 'schaefer': 'orange'}


zorder = global_inter.groupby('reduction')['mean_dice'].max().rank()
for reduction, this_inter in global_inter.groupby('reduction'):
    ax.errorbar(this_inter['size'], this_inter['mean_dice'],
                yerr=this_inter['std_dice'], label=labels[reduction],
                marker=markers[reduction], elinewidth=1.5, linewidth=3, capsize=2,
                color=colors[reduction], zorder=zorder.loc[reduction])

ax.hlines(global_intra['mean_dice'], 21,
          1024, zorder=2, color='black', linewidth=3,
          label='Non reduced')

ax.fill_between([21,
                 1024],
                [global_intra['mean_dice'] + global_intra['std_dice']] * 2,
                [global_intra['mean_dice'] - global_intra['std_dice']] * 2,
                alpha=0.2, zorder=1, color='black'
                )
ax.set_xscale('log')
ticks = [32, 64, 128, 256, 512, 1024]
ax.set_xticks(ticks)
labels = copy.copy(ticks)
ax.set_xticklabels(labels)
handles, labels = ax.get_legend_handles_labels()
atlases = ['UKBB ICA', 'DiFuMo', 'BASC', 'Craddock', 'FIND',
           'Gordon', 'Schaefer', 'Voxel-level']
marker1 = plt.Line2D([], [], color='c', marker='v', linestyle='')
marker2 = plt.Line2D([], [], color='m', marker='^', linestyle='')
marker3 = plt.Line2D([], [], color='y', marker='>', linestyle='')
marker4 = plt.Line2D([], [], color='g', marker='<', linestyle='')
marker5 = plt.Line2D([], [], color='b', marker='s', linestyle='')
marker6 = plt.Line2D([], [], color='purple', marker='8', linestyle='')
marker7 = plt.Line2D([], [], color='orange', marker='p', linestyle='')

leg = plt.legend([marker1, marker2, marker3, marker4, marker5,
                  marker6, marker7, handles[0]], atlases,
                 handlelength=0.4,
                 handletextpad=0.6, fontsize=11, columnspacing=0.6,
                 frameon=False, ncol=4, loc='lower left',
                 bbox_transform=fig.transFigure,
                 bbox_to_anchor=(-0.025, -0.01))
ax.set_xlabel('Dimension', size=12)
ax.xaxis.set_label_coords(-0.05, -0.033)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax.set_ylabel('Mean Dice index relative to\nvoxel-level maps', size=12,
              x=0.12, y=0.55)
ax.set_xlim([16, 1100])
ax.set_ylim([-.13, 0.86])
plt.tight_layout()
plt.subplots_adjust(top=0.97, right=0.95, bottom=0.21, left=0.21)
ax.annotate('Across-fold consistency\nat voxel-level',
            xy=(0.4, 0.91), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='->', color='black'),
            xytext=(0.4, 0.76), fontsize=10, ha='center', zorder=1000)
plt.savefig(expanduser('../../figures/glm_dice.pdf'))
