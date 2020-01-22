"""Grabber for popular functional atlases

   DiFuMo atlases download function can be adapted
   by downloading them at: https://parietal-inria.github.io/DiFuMo/
"""
import os
import numpy as np
from sklearn.datasets.base import Bunch

from nilearn.datasets.utils import _get_dataset_dir, _fetch_files


def fetch_craddock_adhd_200_parcellations(data_dir=None, verbose=1):
    """These are the parcellations from the Athena Pipeline of the ADHD
    200 preprocessing initiative. 200 and 400 ROI atlases were generated
    using 2-level parcellation of 650 individuals from the ADHD 200 Sample.

    Parameters
    ----------
    data_dir : str
        Directory where the data should be downloaded.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        parcellations_200, parcellations_400
    """
    url = 'http://www.nitrc.org/frs/download.php/5906/ADHD200_parcellations.tar.gz'
    opts = {'uncompress': True}

    dataset_name = 'craddock_ADHD200_parcellations'
    filenames = [("ADHD200_parcellate_200.nii.gz", url, opts),
                 ("ADHD200_parcellate_400.nii.gz", url, opts)]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)
    keys = ("parcellations_200", "parcellations_400")
    params = dict(list(zip(keys, files)))
    return Bunch(**params)


def fetch_atlas_gordon_2014(coordinate_system='MNI', resolution=2,
                            data_dir=None, url=None, resume=True, verbose=1):
    """Download and returns Gordon et al. 2014 atlas

    References
    ----------
    Gordon, E. M., Laumann, T. O., Adeyemo, B., Huckins, J. F., Kelley, W. M., &
    Petersen, S. E., "Generation and evaluation of a cortical area
    parcellation from resting-state correlations", 2014, Cerebral cortex, bhu239.

    See http://www.nil.wustl.edu/labs/petersen/Resources.html for more
    information on this parcellation.
    """
    if url is None:
        url = ("https://sites.wustl.edu/petersenschlaggarlab/files/"
               "2018/06/Parcels-19cwpgu.zip")
    dataset_name = "gordon_2014"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    valid_coordinate_systems = ['MNI', '711-2b']

    if coordinate_system not in valid_coordinate_systems:
        raise ValueError('Unknown coordinate system {0}. '
                         'Valid options are {1}'.format(
                             coordinate_system, valid_coordinate_systems))

    if resolution not in [1, 2, 3]:
        raise ValueError('Invalid resolution {0}. '
                         'Valid options are 1, 2 or 3.'.format(resolution))

    target_file = os.path.join('Parcels', 'Parcels_{0}_{1}.nii'.format(
        coordinate_system, str(resolution) * 3))

    atlas = _fetch_files(data_dir, [(target_file, url, {"uncompress": True})],
                         resume=resume, verbose=verbose)

    return atlas


def fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1,
                              data_dir=None, base_url=None, resume=True,
                              verbose=1):
    """Download and return file names for the Schaefer 2018 parcellation

    References
    ----------
    For more information on this dataset, see
    https://github.com/ThomasYeoLab/CBIG/tree/v0.8.1-Schaefer2018_LocalGlobal/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal

    Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ,
    Eickhoff SB, Yeo BTT. Local-Global parcellation of the human
    cerebral cortex from intrinsic functional connectivity MRI,
    Cerebral Cortex, 29:3095-3114, 2018.

    Yeo BT, Krienen FM, Sepulcre J, Sabuncu MR, Lashkari D, Hollinshead M,
    Roffman JL, Smoller JW, Zollei L., Polimeni JR, Fischl B, Liu H,
    Buckner RL. The organization of the human cerebral cortex estimated by
    intrinsic functional connectivity. J Neurophysiol 106(3):1125-65, 2011.

    Licence: MIT.
    """
    valid_n_rois = [100, 200, 300, 400, 500, 600, 800, 1000]
    valid_yeo_networks = [7, 17]
    valid_resolution_mm = [1, 2]
    if n_rois not in valid_n_rois:
        raise ValueError("Requested n_rois={} not available. Valid "
                         "options: {}".format(n_rois, valid_n_rois))
    if yeo_networks not in valid_yeo_networks:
        raise ValueError("Requested yeo_networks={} not available. Valid "
                         "options: {}".format(yeo_networks, valid_yeo_networks))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm)
                         )

    if base_url is None:
        base_url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/'
                    'v0.8.1-Schaefer2018_LocalGlobal/stable_projects/'
                    'brain_parcellation/Schaefer2018_LocalGlobal/'
                    'Parcellations/MNI/'
                    )

    files = []
    labels_file_template = 'Schaefer2018_{}Parcels_{}Networks_order.txt'
    img_file_template = ('Schaefer2018_{}Parcels_'
                         '{}Networks_order_FSLMNI152_{}mm.nii.gz')
    for f in [labels_file_template.format(n_rois, yeo_networks),
              img_file_template.format(n_rois, yeo_networks, resolution_mm)]:
        files.append((f, base_url + f, {}))

    dataset_name = 'schaefer_2018'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    labels_file, atlas_file = _fetch_files(data_dir, files, resume=resume,
                                           verbose=verbose)

    labels = np.genfromtxt(labels_file, usecols=1, dtype="S", delimiter="\t")

    return Bunch(maps=atlas_file, labels=labels)
