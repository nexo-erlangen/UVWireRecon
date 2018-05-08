#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import warnings
import numpy as np
import h5py
import random

#------------- Function used for supplying images to the GPU -------------#
def gen_batches_from_files(files, batchsize, class_type, f_size=None, yield_mc_info=False, swap_col=None):
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
    eventInfo = {}
    while 1:
        random.shuffle(files)  #  TODO maybe omit in future?
        print files
        for filename in files:
            print filename
            f = h5py.File(str(filename), "r")
            if f_size is None:
                f_size = num_events([filename])
                warnings.warn( 'f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                    'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')
            else:
                raise ValueError('The argument "f_size"=' + str(swap_col) + ' may have no effect. Check implementation.')

            lst = np.arange(0, f_size, batchsize)  #  TODO maybe omit in future?
            # random.shuffle(lst)  #  TODO maybe omit in future?

            for i in lst:
                print i, 'of', f_size
                xs = f['wfs'][ i : i + batchsize ]
                if swap_col is not None:
                    raise ValueError('The argument "swap_col"=' + str(swap_col) + ' is not valid.')
                # filter the labels we don't want for now
                for key in f.keys():
                    if key in ['wfs']: continue
                    print key
                    eventInfo[key] = np.asarray(f[key][ i : i + batchsize ])
                ys = encode_targets(eventInfo, batchsize, class_type)

                yield (xs, ys) if yield_mc_info is False else (xs, ys) + (eventInfo, )
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def encode_targets(y_dict, batchsize, class_type):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. energy_and_position
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    if class_type == 'energy_and_position':
        train_y = np.zeros((batchsize, 4), dtype='float32')
        train_y[:,0] = y_dict['MCEnergy'][:,0]  # energy
        train_y[:,1] = y_dict['MCPosX'][:,0]  # dir_x
        train_y[:,2] = y_dict['MCPosY'][:,0]  # dir_y
        train_y[:,3] = y_dict['MCTime'][:,0]  # time (to calculate dir_z)
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')

    return train_y

#------------- Functions used for supplying images to the GPU -------------#

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['wfs'].shape[0]
        f.close()
    return counter

path = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/UniformGamma_ExpWFs_MC_SS/'
file = [path + '0.hdf5', path + '1.hdf5', path + '2.hdf5']
gen = gen_batches_from_files(file, 16, 'energy_and_position', f_size=None, yield_mc_info=False, swap_col=None)
gen.next()


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


#------------- Classes -------------#
#
# class TensorBoardWrapper(ks.callbacks.TensorBoard):
#     """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
#      Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
#      Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.
#
#      This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
#      that the whole validation data is put into a single array by using the generator.
#      Then, the single array is used in the validation steps. This workaround is experimental!"""
#     def __init__(self, batch_gen, nb_steps, **kwargs):
#         super(TensorBoardWrapper, self).__init__(**kwargs)
#         self.batch_gen = batch_gen # The generator.
#         self.nb_steps = nb_steps   # Number of times to call next() on the generator.
#
#     def on_epoch_end(self, epoch, logs):
#         # Fill in the `validation_data` property.
#         # After it's filled in, the regular on_epoch_end method has access to the validation_data.
#         imgs, tags = None, None
#         for s in xrange(self.nb_steps):
#             ib, tb = next(self.batch_gen)
#             if imgs is None and tags is None:
#                 imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
#                 tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
#             imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
#             tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
#         self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
#         return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)
#
#
# class BatchLevelPerformanceLogger(ks.callbacks.Callback):
#     # Gibt loss aus über alle :display batches, gemittelt über die letzten :display batches
#     def __init__(self, display, modelname, steps_per_epoch, epoch):
#         ks.callbacks.Callback.__init__(self)
#         self.seen = 0
#         self.display = display
#         self.averageLoss = 0
#         self.averageAcc = 0
#         self.logfile_train_fname = 'models/trained/perf_plots/log_train_' + modelname + '.txt'
#         self.logfile_train = None
#         self.steps_per_epoch = steps_per_epoch
#         self.epoch = epoch
#         self.loglist = []
#
#     def on_batch_end(self, batch, logs={}):
#         self.seen += 1
#         self.averageLoss += logs.get('loss')
#         self.averageAcc += logs.get("acc")
#         if self.seen % self.display == 0:
#             averaged_loss = self.averageLoss / self.display
#             averaged_acc = self.averageAcc / self.display
#             batchnumber_float = (self.seen - self.display / 2.) / float(self.steps_per_epoch) + self.epoch - 1  # start from zero
#             self.loglist.append('\n{0}\t{1}\t{2}\t{3}'.format(self.seen, batchnumber_float, averaged_loss, averaged_acc))
#             self.averageLoss = 0
#             self.averageAcc = 0
#
#     def on_epoch_end(self, batch, logs={}):
#         self.logfile_train = open(self.logfile_train_fname, 'a+')
#         if os.stat(self.logfile_train_fname).st_size == 0: self.logfile_train.write("#Batch\t#Batch_float\tLoss\tAccuracy")
#
#         for batch_statistics in self.loglist: # only write finished epochs to the .txt
#             self.logfile_train.write(batch_statistics)
#
#         self.logfile_train.flush()
#         os.fsync(self.logfile_train.fileno())
#         self.logfile_train.close()
#
#------------- Classes -------------#