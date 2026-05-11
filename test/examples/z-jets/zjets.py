from __future__ import absolute_import, division, print_function

import json
import os
from typing import Optional

import hashlib
from urllib.request import urlretrieve
from urllib.error import URLError

import numpy as np

import torch
from tensordict import TensorDict

__all__ = ['load']

ZENODO_URL_PATTERN = 'https://zenodo.org/record/{}/files/{}?download=1'
DROPBOX_URL_PATTERN = 'https://www.dropbox.com/s/{}/{}?dl=1'
FILENAME_PREFIX = {
    'pythia21': 'Pythia21_Zjet_pTZ-200GeV',
    'pythia25': 'Pythia25_Zjet_pTZ-200GeV',
    'pythia26': 'Pythia26_Zjet_pTZ-200GeV',
    'herwig':   'Herwig_Zjet_pTZ-200GeV'
}

#EF_DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, 'data')
PROJECT_HOME = "/home/tanmaypani/star-workspace/jet-angularity-study"
JSON_METADATA_DIR = f"{PROJECT_HOME}/runtime_files"
DATASETS = frozenset(FILENAME_PREFIX.keys())
KEYS = {
    # jets
    'jets': 'Jet four vectors, [pt, y, phi, m]',
    'particles': 'Jet constituents, [pt, y, phi, pid_float]',

    # Zs
    'Zs': 'Z momenta, [pt, y, phi]',

    # observables
    'ang2s': 'Jet angularity, beta = 2',
    'lhas': 'Les Houches Angularity, beta = 1/2',
    'mults': 'Jet constituent multiplicity',
    'sdms': 'Soft Dropped jet mass, zcut = 0.1, beta = 0',
    'tau2s': '2-subjettiness, beta = 1',
    'widths': 'Jet widths, beta = 1',
    'zgs': 'Groomed momentum fraction, zcut = 0.1, beta = 0',
}
NUM_FILES = 17
SOURCES = ['zenodo', 'dropbox', "path"]
ZENODO_RECORD = '3548091'

def load(dataset, pad=True, cache_dir='.', source='zenodo', which='all', include_keys=None, exclude_keys=["particles"]):
    # handle selecting keys
    keys = set(include_keys or KEYS.keys()) - set(exclude_keys or [])
    for key in keys:
        if key not in KEYS:
            raise ValueError("Unrecognized key '{}'".format(key))

    # create dictionray to store values to be returned
    levels = ['gen', 'sim'] if which == 'all' else [which.lower()]
    for level in levels:
        if level != 'gen' and level != 'sim':
            raise ValueError("Unrecognized specification '{}' ".format(level) +
                             "for argument 'which', allowed options are 'all', 'gen', 'sim'")

    # check that options are valid
    dataset_low = dataset.lower()
    if dataset_low not in DATASETS:
        raise ValueError("Invalid dataset '{}'".format(dataset))
    if source not in SOURCES:
        raise ValueError("Invalid source '{}'".format(source))

    
    data_dir = os.path.join(cache_dir, 'datasets', 'ZjetsDelphes')
    filepaths = _fetch_filepaths(dataset_low, data_dir, source)
    
    tdicts = {l : [] for l in levels}

    for path in filepaths:
        with np.load(path) as f:
            for lvl in levels:
                val = TensorDict({})
                #vals_to_stack = []
                for key in keys:
                    #if lvl == "sim" and key == "Zs": 
                    if key == "Zs": 
                        continue
                    if key == 'particles' and not pad:
                        val[key] = [np.array(ps[ps[:,0] > 0]) for ps in f[f"{lvl}_{key}"]]
                    elif key == "jets": 
                        val["pts"] = f[f"{lvl}_{key}"][:, 0]
                        val["ys"] = f[f"{lvl}_{key}"][:, 1]
                        val["phis"] = f[f"{lvl}_{key}"][:, 2]
                        val["ms"] = f[f"{lvl}_{key}"][:, 3]
                    else:
                        val[key] = f[f"{lvl}_{key}"]

                tdicts[lvl].append(val.auto_batch_size_())

    return [
        torch.cat(v).memmap_(
            prefix = os.path.join(
                data_dir, 
                FILENAME_PREFIX[dataset_low], 
                l
            )
        )
        for l, v in tdicts.items()
    ]


def _fetch_filepaths(dataset, data_dir, source):
    # load info from JSON file
    with open(os.path.join(JSON_METADATA_DIR, 'ZjetsDelphes.json'), 'r') as f:
        INFO = json.load(f)

    # get filenames
    filenames = [f"{FILENAME_PREFIX[dataset]}_{i}.npz" for i in range(NUM_FILES)]

    # get urls
    if source == 'dropbox':
        db_link_hashes = INFO['dropbox_link_hashes'][dataset]
        urls = [DROPBOX_URL_PATTERN.format(dl, fn) for dl,fn in zip(db_link_hashes, filenames)]
    elif source == 'zenodo':
        urls = [ZENODO_URL_PATTERN.format(ZENODO_RECORD, fn) for fn in filenames]
    else:
        urls = filenames
    
    # get hashes
    hashes = INFO['hashes'][dataset]['sha256']

    return [
        _maybe_download(filename, url, data_dir, file_hash=h)
        for filename, url, h in zip(filenames, urls, hashes)
    ]


def _maybe_download(filename, url, data_dir, file_hash=None):
    """Pulls file from the internet."""

    # handle '~' in path
    datadir = os.path.expanduser(data_dir)
    # ensure that directory exists
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    fpath = os.path.join(datadir, filename)

    # determine if file needs to be downloaded
    download = False
    if os.path.exists(fpath):
        if file_hash is not None and not _validate_file(fpath, file_hash):
            print('Local file hash does not match so we will redownload...')
            download = True
    else:
        download = True

    if download:
        print('Downloading {} from {} to {}'.format(filename, url, datadir))

        error_msg = 'URL fetch failure on {}: {} -- {}'
        try:
            try:
                urlretrieve(url, fpath)
            except URLError as e:
                raise Exception(error_msg.format(url, e.errno, e.reason))
            #except HTTPError as e:
            #    raise Exception(error_msg.format(url, e.errno, e.reason))
        except (Exception, KeyboardInterrupt):
            if os.path.exists(fpath):
                os.remove(fpath)
            raise

        if file_hash is not None:
            assert _validate_file(fpath, file_hash), 'Hash of downloaded file incorrect.'

    return fpath

def _validate_file(fpath, file_hash, algorithm='auto', chunk_size=131071):
    """Validates a file against a sha256 or md5 hash.
    # Arguments
        fpath: path to the file being validated
        file_hash:  The expected hash string of the file.
            The sha256 and md5 hash algorithms are both supported.
        algorithm: Hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        Whether the file is valid
    """
    if ((algorithm == 'sha256') or (algorithm == 'auto' and len(file_hash) == 64)):
        hasher = 'sha256'
    else:
        hasher = 'md5'

    return str(_hash_file(fpath, hasher, chunk_size)) == str(file_hash)

def _hash_file(fpath, algorithm='sha256', chunk_size=131071):
    """Calculates a file sha256 or md5 hash.
    # Example
    ```python
        >>> from keras.data_utils import _hash_file
        >>> _hash_file('/path/to/file.zip')
        'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    ```
    # Arguments
        fpath: path to the file being validated
        algorithm: hash algorithm, one of 'auto', 'sha256', or 'md5'.
            The default 'auto' detects the hash algorithm in use.
        chunk_size: Bytes to read at a time, important for large files.
    # Returns
        The file hash
    """
    if (algorithm == 'sha256') or (algorithm == 'auto'):
        hasher = hashlib.sha256()
    else:
        hasher = hashlib.md5()

    with open(fpath, 'rb') as fpath_file:
        for chunk in iter(lambda: fpath_file.read(chunk_size), b''):
            hasher.update(chunk)

    return hasher.hexdigest()

def _pad_events_axis1(events, axis1_shape):
    """Pads the first axis of the NumPy array `events` with zero subarrays
    such that the first dimension of the results has size `axis1_shape`.
    """

    if events.ndim != 3:
        raise ValueError('events must be a 3d numpy array')

    num_zeros = axis1_shape - events.shape[1]
    if num_zeros > 0:
        zeros = np.zeros((events.shape[0], num_zeros, events.shape[2]))
        return np.concatenate((events, zeros), axis=1)

    return events
