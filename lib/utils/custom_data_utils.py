import os 
import sys 
import numpy as np 
import scipy.io as sio
import h5py

'''
    Helper functions to write .h5 data files for pointnet, etc.
'''

def save_h5_basic(h5_filename, data, data_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.close()


def save_h5(h5_filename, data, label, bbox=None, data_dtype='float32', label_dtype='int', 
            bbox_dtype='float32'):
    h5_fout = h5py.File(h5_filename, 'w')
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype,
    )
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype,
    )
    if isinstance(bbox, np.ndarray):
        h5_fout.create_dataset(
            'bbox', data=bbox,
            compression='gzip', compression_opts=1,
            dtype=bbox_dtype,
        )       
    h5_fout.close()
    

def load_h5_basic(filename): 
    f = h5py.File(filename, 'r')
    data = f['data'][:]
    return data


def load_h5(h5_filename, bbox=False):
    f = h5py.File(h5_filename, 'r')
    # f.keys() should be [u'data', u'label']
    data = f['data'][:]
    label = f['label'][:]
    if bbox: 
        bbox = f['bbox'][:]
        return (data, label, bbox)
    return (data,label)


def get_data_files(data_dir):
    """ Retrieves a list from data_files to train/test with from a txt file.

    Args:
        data_dir (string): the path to the txt file
    """
    data_files = [x.strip() for x in open(data_dir).readlines()]
    return data_files

