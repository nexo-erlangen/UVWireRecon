#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import warnings
import numpy as np
import h5py
import random

#------------- Function used for supplying images to the GPU -------------#
def generate_batches_from_files(files, batchsize, class_type=None, f_size=None, yield_mc_info=False):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string filepath: Full filepath of the input h5 file, e.g. '/path/to/file/file.h5'.
    :param int batchsize: Size of the batches that should be generated.
    :param str class_type: String identifier to specify the exact target variables. i.e. 'energy_and_position'
    :param int/None f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param bool yield_mc_info: Specifies if mc-infos (y_values) should be yielded as well.
                               The mc-infos are used for evaluation after training and testing is finished.
    :param bool/str swap_col: Specifies, if the index of the columns for xs should be swapped. Necessary for 3.5D nets.
                          Currently available: 'yzt-x' -> [3,1,2,0] from [0,1,2,3]
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info if yield_mc_info=True).
    """

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    eventInfo = {}
    while 1:
        random.shuffle(files)  # TODO maybe omit in future? # TODO shuffle events between files
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)
                # warnings.warn( 'f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                #     'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

            lst = np.arange(0, f_size, batchsize)  #  TODO maybe omit in future?
            # random.shuffle(lst)  #  TODO maybe omit in future?

            for i in lst:
                xs = f['wfs'][ i : i + batchsize ]
                xs = np.swapaxes(xs, 0, 1)
                xs = np.swapaxes(xs, 2, 3)
                # filter the labels we don't want for now
                for key in f.keys():
                    if key in ['wfs']: continue
                    eventInfo[key] = np.asarray(f[key][ i : i + batchsize ])
                ys = encode_targets(eventInfo, batchsize, class_type)
                yield (list(xs), ys) if yield_mc_info is False else (xs, ys) + (eventInfo, )
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def encode_targets(y_dict, batchsize, class_type=None):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. energy_and_position
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    if class_type == None:
        train_y = np.zeros(batchsize, dtype='float32')
    elif class_type == 'energy_and_position':
        train_y = np.zeros((batchsize, 4), dtype='float32')
        train_y[:,0] = y_dict['MCEnergy'][:,0]  # energy
        train_y[:,1] = y_dict['MCPosX'][:,0]  # dir_x
        train_y[:,2] = y_dict['MCPosY'][:,0]  # dir_y
        train_y[:,3] = y_dict['MCTime'][:,0]  # time (to calculate dir_z)
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')

    return train_y

#------------- Functions used for supplying images to the GPU -------------#

def getNumEvents(files):
    if isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y,files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['wfs'].shape[0]
        f.close()
    return counter

#------------- Functions for preprocessing -------------#
# def get_array_memsize(array):
#     """
#     Calculates the approximate memory size of an array.
#     :param ndarray array: an array.
#     :return: float memsize: size of the array in bytes.
#     """
#     shape = array.shape
#     n_numbers = reduce(lambda x, y: x*y, shape) # number of entries in an array
#     precision = 8 # Precision of each entry, typically uint8 for xs datasets
#     memsize = (n_numbers * precision) / float(8) # in bytes
#
#     return memsize
#
#------------- Functions for preprocessing -------------#

