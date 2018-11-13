#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generator used for training a CNN."""

import warnings
import numpy as np
import h5py
import random
import cPickle as pickle
from sklearn.cluster import DBSCAN


# ------------- Function used for supplying images to the GPU ------------- #
def generate_batches_from_files(files, batchsize, inputImages, multiplicity, class_type=None, f_size=None, yield_mc_info=0):
    """
    Generator that returns batches of images ('xs') and labels ('ys') from a h5 file.
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param int batchsize: Size of the batches that should be generated.
    :param str class_type: String identifier to specify the exact target variables. i.e. 'energy_and_position'
    :param int/None f_size: Specifies the filesize (#images) of the .h5 file if not the whole .h5 file
                       but a fraction of it (e.g. 10%) should be used for yielding the xs/ys arrays.
                       This is important if you run fit_generator(epochs>1) with a filesize (and hence # of steps) that is smaller than the .h5 file.
    :param int yield_mc_info: Specifies if mc-infos should be yielded. 0: Only Waveforms, 1: Waveforms+MC Info, 2: Only MC Info
                               The mc-infos are used for evaluation after training and testing is finished.
    :return: tuple output: Yields a tuple which contains a full batch of images and labels (+ mc_info depending on yield_mc_info).
    """
    # print files

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    wireindex = 'null'

    if inputImages == 'U':
        wireindex = [0, 2]
    elif inputImages == 'V':
        wireindex = [1, 3]
    elif inputImages in ['UV', 'U+V']:
        wireindex = [0, 1, 2, 3]
        # slice(4)
    else:
        raise ValueError('passed wire specifier need to be U/V/UV')

    eventInfo = {}
    while 1:
        random.shuffle(files)  # TODO maybe omit in future? # TODO shuffle events between files
        for filename in files:
            f = h5py.File(str(filename), "r")
            if f_size is None: f_size = getNumEvents(filename)
                # warnings.warn( 'f_size=None could produce unexpected results if the f_size used in fit_generator(steps=int(f_size / batchsize)) with epochs > 1 '
                #     'is not equal to the f_size of the true .h5 file. Should be ok if you use the tb_callback.')

            lst = np.arange(0, f_size, batchsize)
            # random.shuffle(lst)  #  TODO maybe omit in future?

            # filter the labels we don't want for now
            for key in f.keys():
                if key in ['wfs']: continue
                eventInfo[key] = np.asarray(f[key])
            ys = encode_targets(eventInfo, f_size, multiplicity, class_type)

            for i in lst:
                if not yield_mc_info == 2:
                    xs_i = f['wfs'][ i : i + batchsize, wireindex]
                    xs_i = np.swapaxes(xs_i, 0, 1)
                    xs_i = np.swapaxes(xs_i, 2, 3)
                ys_i = ys[i: i + batchsize]
                # print ys_i
                # raw_input('')

                if   yield_mc_info == 0:    yield (list(xs_i), ys_i)
                elif yield_mc_info == 1:    yield (list(xs_i), ys_i) + ({ key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() }
                else:   raise ValueError("Wrong argument for yield_mc_info (0/1/2)")
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize


def encode_targets(y_dict, batchsize, multiplicity, class_type=None):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. energy_and_position
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """

    # if multiplicity == 'SS+MS':
    #     if class_type == 'energy_and_UV_position':
    #
    #         numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])
    #
    #         # TODO: in x, y, z clustern; dann in u, v, z umrechnen; PCDs mit Depositchannel < 0 NACH clustern
    #         # TODO: wegwerfen um ihr Energie nicht fuer Berechnung der Cluster Energie zu verwenden
    #         x, y, z, energy, depositChannel = y_dict['MCPosX'][eventnumber][0:numPCDs], y_dict['MCPosY'][eventnumber][0:numPCDs], y_dict['MCPosZ'][eventnumber][0:numPCDs], y_dict['MCEnergy'][eventnumber][0:numPCDs], y_dict['MCDepositChannel'][eventnumber][0:numPCDs]
    #
    #         X = []
    #
    #
    #         for i in range(len(x)):
    #             X.append([x[i], y[i], z[i]])
    #             # if depositChannel[i] == -999:
    #             #     x_noDeposit.append(x[i])
    #             #     y_noDeposit.append(y[i])
    #             #     z_noDeposit.append(z[i])
    #
    #         clustering = DBSCAN(eps=5, min_samples=1).fit(X)
    #         label = clustering.labels_
    #
    #         x_mean, y_mean, z_mean, energy_sum = [], [], [], []
    #         for j in range(max(label) + 1):
    #             mask = label == j
    #             x_temp = x[mask]
    #             y_temp = y[mask]
    #             z_temp = z[mask]
    #             energy_temp = energy[mask]
    #
    #             x_mean.append(np.average(x_temp, weights=energy_temp))
    #             y_mean.append(np.average(y_temp, weights=energy_temp))
    #             z_mean.append(np.average(z_temp, weights=energy_temp))
    #             energy_sum.append(sum(energy_temp))
    #
    #
    #         # x_mean, y_mean to u, v
    #
    #
    #
    #         train_y = np.zeros((batchsize, 20), dtype='float32')
    #
    #         train_y[:, 0] = normalize(y_dict['MCEnergy'][:, 0], 'energy')  # energy
    #         train_y[:, 1] = normalize(y_dict['MCPosU'][:, 0], 'U')  # dir_u
    #         train_y[:, 2] = normalize(y_dict['MCPosV'][:, 0], 'V')  # dir_v
    #         train_y[:, 3] = normalize(y_dict['MCPosZ'][:, 0], 'Z')  # dir_z

    # else:
        # print len(y_dict)

    print '%%%%%%%%'
    print class_type
    print '%%%%%%%%'

    if class_type == None:
        # train_y = np.zeros(batchsize, dtype='float32')
        raise ValueError('Class type: ' + str(class_type))
    elif class_type == 'energy_and_position' or class_type == 'position_and_energy':
        train_y = np.zeros((batchsize, 4), dtype='float32')
        train_y[:, 0] = y_dict['MCEnergy'][:,0]  # energy
        train_y[:, 1] = y_dict['MCPosX'][:,0]  # dir_x
        train_y[:, 2] = y_dict['MCPosY'][:,0]  # dir_y
        train_y[:, 3] = y_dict['MCTime'][:,0]  # time (to calculate dir_z)
    # elif class_type == 'energy_and_UV_position':
    #     train_y = np.zeros((batchsize, 4), dtype='float32')
    #     train_y[:, 0] = y_dict['MCEnergy'][:, 0]  # energy
    #     train_y[:, 1] = y_dict['MCPosU'][:, 0]  # dir_u
    #     train_y[:, 2] = y_dict['MCPosV'][:, 0]  # dir_v
    #     train_y[:, 3] = y_dict['MCTime'][:, 0]  # time (to calculate dir_z)

    elif class_type == 'energy_and_UV_position':
        print '$$$$$$$$$$'
        train_y = np.zeros((batchsize, 4), dtype='float32')
        train_y[:, 0] = normalize(y_dict['MCEnergy'][:, 0], 'energy')  # energy
        train_y[:, 1] = normalize(y_dict['MCPosU'][:, 0], 'U')  # dir_u
        train_y[:, 2] = normalize(y_dict['MCPosV'][:, 0], 'V')  # dir_v
        train_y[:, 3] = normalize(y_dict['MCPosZ'][:, 0], 'Z')  # dir_z
    elif class_type == 'energy':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCEnergy'][:,0]  # energy
    elif class_type == 'U':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCPosU'][:, 0]  # dir_u
    elif class_type == 'V':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCPosV'][:, 0]  # dir_u
    elif class_type == 'position':
        train_y = np.zeros((batchsize, 3), dtype='float32')
        train_y[:, 0] = y_dict['MCPosU'][:,0]  # dir_u
        train_y[:, 1] = y_dict['MCPosV'][:,0]  # dir_v
        train_y[:, 2] = y_dict['MCTime'][:, 0] # time (to calculate dir_z)
    elif class_type == 'time':
        train_y = np.zeros((batchsize, 1), dtype='float32')
        train_y[:, 0] = y_dict['MCTime'][:, 0]  # time (to calculate dir_z)
    else:
        raise ValueError('Class type ' + str(class_type) + ' not supported!')

    return train_y


def predict_events(model, generator):
    X, Y_TRUE, EVENT_INFO = generator.next()
    Y_PRED = np.asarray(model.predict(X, 10))
    return (Y_PRED, Y_TRUE, EVENT_INFO)


def get_events(args, files, model, fOUT):
    try:
        if args.new: raise IOError
        spec = pickle.load(open(fOUT, "rb"))
        if args.events > len(spec['Y_TRUE']): raise IOError
    except IOError:
        if model == None: print 'model not found and not events file found' ; exit()
        events_per_batch = 50
        if args.events % events_per_batch != 0: raise ValueError('choose event number in multiples of %f events'%(events_per_batch))
        iterations = round_down(args.events, events_per_batch) / events_per_batch
        gen = generate_batches_from_files(files, events_per_batch, inputImages=args.inputImages, class_type=args.var_targets, f_size=None, yield_mc_info=1)

        Y_PRED, Y_TRUE = [], []
        for i in xrange(iterations):
            print i * events_per_batch, ' of ', iterations * events_per_batch
            Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp = predict_events(model, gen)
            Y_PRED.extend(Y_PRED_temp)
            Y_TRUE.extend(Y_TRUE_temp)
            if i == 0: EVENT_INFO = EVENT_INFO_temp
            else:
                for key in EVENT_INFO:
                    EVENT_INFO[key] = np.concatenate((EVENT_INFO[key], EVENT_INFO_temp[key]))

        spec = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': EVENT_INFO}
        pickle.dump(spec, open(fOUT, "wb"))
    return spec


def getNumEvents(files):
    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x,y: x+y, files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += f['MCEventNumber'].shape[0]
        f.close()
    return counter


def get_array_memsize(array, unit='KB'):
    """
    Calculates the approximate memory size of an array.
    :param ndarray array: an array.
    :param string unit: output unit of memsize.
    :return: float memsize: size of the array given in unit.
    """
    units = {'B': 0., 'KB': 1., 'MB': 2., 'GB':3.}
    if isinstance(array, list):
        array = np.asarray(array)

    shape = array.shape
    n_numbers = reduce(lambda x, y: x*y, shape)     # number of entries in an array
    precision = array.dtype.itemsize    # Precision of each entry in bytes
    memsize = (n_numbers * precision)   # in bytes
    return memsize/1024**units[unit]


def round_down(num, divisor):
    return num - (num%divisor)


def normalize(data, mode):
    if mode == 'energy':
        data_norm = (data - 550) / 2950     * 1
    elif mode == 'U' or mode == 'V':
        data_norm = (data + 170) / 340      * 1
    elif mode == 'time':
        data_norm = (data - 1030) / 110     * 1
    elif mode == 'Z':
        data_norm = np.abs(data) / 190      * 1
    return data_norm * 100



