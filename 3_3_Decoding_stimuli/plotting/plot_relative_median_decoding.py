"""Comparison plot with and without standard scaler
"""
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker


def plot_regression(data, meaned, task, axes, atlases, colors):
    """Plot regression results on axes from age and emotion predictions
       with customized colors and order of atlases
    """
    for color, atlas in zip(colors, atlases):
        this_atlas = data[data['Atlas'] == atlas]
        this_atlas_mean = this_atlas.groupby(['Dimension']).mean().sort_values(by='rank')
        this_atlas_mean = this_atlas_mean.reset_index()
        # x = this_atlas_mean.Dimension.unique().tolist()
        x = list(map(lambda x: int(x), list(this_atlas_mean.Dimension)))
        axes.plot(x, this_atlas_mean['demeaned_scores'], 'o', alpha=0.3,
                  color=color, markersize=5., zorder=10)
    return axes


sns.set_style("whitegrid")
plot_data = pd.read_csv('scores.csv')
plot_data = plot_data.drop('Unnamed: 0', axis=1)

dic_dimension = {'21': 1,
                 '55': 2,
                 '64': 3,
                 '90': 4,
                 '100': 5,
                 '122': 6,
                 '128': 7,
                 '197': 8,
                 '200': 9,
                 '256': 10,
                 '264': 11,
                 '300': 12,
                 '325': 13,
                 '400': 14,
                 '444': 15,
                 '499': 16,
                 '500': 17,
                 '512': 18,
                 '600': 19,
                 '800': 20,
                 '1000': 21,
                 '1024': 22,
                 '1500': 23}
plot_data['rank'] = plot_data['Dimension'].map(dic_dimension)
plot_data.sort_values(by=['rank'], inplace=True)

print(plot_data)
df = plot_data.groupby(['task', 'Atlas', 'Dimension']).mean()

idx = pd.IndexSlice

def demean(group):
    # mean = group.loc[idx[:, 'Non \n reduced', 1500]].values[0]
    mean = group.median()
    return group - mean

demeaned_scores = df.groupby(level='task')['Scores'].transform(demean)
df['demeaned_scores'] = demeaned_scores

print(df['demeaned_scores'])
df = df.reset_index()
meaned = df.groupby(['Atlas', 'Dimension'])['demeaned_scores'].median()
meaned = meaned.reset_index()
meaned['rank'] = meaned['Dimension'].map(dic_dimension)
meaned.sort_values(by=['rank'], inplace=True)

# Visualizations
fig, axs = plt.subplots(figsize=(3.8, 3.8))
atlases = ['UKBB ICA', 'SOMF', 'BASC', 'Craddock', 'FIND',
           'Gordon', 'Schaefer', 'Non \n reduced']
colors = ['c', 'm', 'y', 'g', 'b', 'purple', 'orange', 'k']

markers = {'UKBB ICA': 'v',
           'SOMF': '^',
           'BASC': '>',
           'Craddock': '<',
           'FIND': 's',
           'Gordon': '8',
           'Power': 'd',
           'Schaefer': 'p',
           'Non \n reduced': 'X'}

tasks = ['emotion', 'pain', 'face vs place',
         'punish vs reward', 'relational versus match',
         'left vs right button press']

zorder = 100 - df.groupby(by=['Atlas', 'Dimension'])['demeaned_scores'].mean().groupby(by='Atlas').max().argsort()
print(zorder)

for i, task in enumerate(tasks):
    this_data = df[df['task'] == task]
    ax = plot_regression(data=this_data, meaned=meaned, task=task,
                         axes=axs, atlases=atlases, colors=colors)

for color, atlas in zip(colors, atlases):
    atl_mean = meaned[meaned['Atlas'] == atlas]
    xs = list(map(lambda x: int(x), list(atl_mean['Dimension'])))
    ax.plot(xs, atl_mean['demeaned_scores'], marker=markers[atlas],
            color=color, linewidth=3., markersize=10., zorder=zorder.loc[atlas])
ax.set_xscale('log')

ticks = list(map(lambda x: int(x), list(dic_dimension.keys())))

ticks = [32, 64, 128, 256, 512, 1024]
ax.set_xticks(ticks)
labels = copy.copy(ticks)
# labels[-1] = 'Voxels'
ax.set_xticklabels(labels)

ax.set_ylabel('Accuracy gain relative\nto median atlas performance', size=12)
ax.set_xlabel('Dimension', size=12)
ax.xaxis.set_label_coords(-0.05, -0.033)
ax.yaxis.set_label_coords(-0.15, 0.5)
ax.set_ylim([-.1, .05])
ax.set_xlim([16, 1800])

ax.axhline(0, linewidth=2.5, zorder=0, color='0.8')

# Non-reduced marker position
y = meaned[meaned['Atlas'] == 'Non \n reduced'].demeaned_scores.values[0]
ax.axhline(y=y, color=colors[-1], alpha=0.45)

plt.xticks(size=12)
# ticks = ax.get_xticklabels()
# ticks[-1].set_size(10)
# ticks[-1].set_zorder(10000)
# ticks = ax.xaxis.get_major_ticks()
# ticks[-1].set_pad(-15)
plt.yticks(size=10)
percentages = [0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1]
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1 / 2))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))

atlases = ['UKBB ICA', 'DiFuMo', 'BASC', 'Craddock', 'FIND',
           'Gordon', 'Schaefer', 'Voxel-level']
marker1 = plt.Line2D([], [], color='c', marker='v', linestyle='')
marker2 = plt.Line2D([], [], color='m', marker='^', linestyle='')
marker3 = plt.Line2D([], [], color='y', marker='>', linestyle='')
marker4 = plt.Line2D([], [], color='g', marker='<', linestyle='')
marker5 = plt.Line2D([], [], color='b', marker='s', linestyle='')
marker6 = plt.Line2D([], [], color='purple', marker='8', linestyle='')
marker7 = plt.Line2D([], [], color='orange', marker='p', linestyle='')
marker8 = plt.Line2D([], [], color='black', marker='X', linestyle='')

leg = plt.legend([marker1, marker2, marker3, marker4, marker5,
                  marker6, marker7, marker8], atlases,
                 handlelength=0.01, fontsize=11, columnspacing=0.6,
                 frameon=False, ncol=4, loc='lower left',
                 bbox_transform=fig.transFigure,
                 bbox_to_anchor=(-0.01, -0.01))
leg.set_zorder(1000)
# for l in leg.get_lines():
#     l.set_alpha(1)
#     l.set_marker('o')
plt.tight_layout(rect=[-0.01, 0.01, 1, 1], h_pad=-0.1)

ax_yticklabels = []
for i in ax.get_yticklabels():
    print(i.get_text())
    if i.get_text() == '-0%':
        ax_yticklabels.append('0%')
    elif i.get_text().startswith('-'):
        ax_yticklabels.append(i.get_text())
    else:
        ax_yticklabels.append('+' + i.get_text())
ax.set_yticklabels(ax_yticklabels)
plt.subplots_adjust(top=0.97, right=0.95, bottom=0.21, left=0.21)
plt.savefig('../../figures/relative_to_median_decoding.pdf')
