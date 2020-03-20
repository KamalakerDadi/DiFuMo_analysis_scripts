"""DiFuMo extraction
"""
from os.path import join

from modl.decomposition.fmri import fMRIDictFact

from fetcher import fetch_resampled_openneuro


n_components = 1024

batch_size = 200
learning_rate = 0.92
method = 'masked'
reduction = 12
n_epochs = 1
verbose = 15
n_jobs = 1
alpha = 1e-3
smoothing_fwhm = 4
positive = True

data = fetch_resampled_openneuro()

# leave out task mixed gambles
funcs = data[data['task'] != 'task-mixedgamblestask']
func_imgs = funcs.func_resampled.values

for components in [64, 128, 256, 512, 1024]:
    dict_fact = fMRIDictFact(method=method,
                             verbose=verbose,
                             n_epochs=n_epochs,
                             smoothing_fwhm=smoothing_fwhm,
                             n_jobs=n_jobs,
                             random_state=1,
                             n_components=n_components,
                             positive=positive,
                             learning_rate=learning_rate,
                             batch_size=batch_size,
                             mask_strategy='epi',
                             reduction=reduction,
                             memory='cache',
                             change_dtype=True,
                             alpha=alpha,
                             memory_level=2,
                             )
dict_fact.fit(func_imgs)
dict_fact.components_img_.to_filename(join(components,
                                           'maps.nii.gz'))
