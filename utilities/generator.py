#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Generator used for training a CNN."""

import warnings
import numpy as np
import h5py
import random
import cPickle as pickle
from sklearn.cluster import DBSCAN
from keras.utils import to_categorical


# ------------- Function used for supplying images to the GPU ------------- #
def generate_batches_from_files(files, batchsize, inputImages, multiplicity, class_type=None, f_size=None, yield_mc_info=0, mode='mc'):
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
    # print '>>', files

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variabel need to be list/np.array/str/dict[dict]')

    wireindex = 'null'

    histo = []


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


            ys = encode_targets(eventInfo, f_size, multiplicity, class_type, histo=histo, mode=mode)


            for i in lst:
                if not yield_mc_info == 2:

                    xs_i = f['wfs'][ i : i + batchsize, wireindex]

                    #  pad to larger waveforms
                    xs_i = np.pad(xs_i, ((0, 0), (0, 0), (0, 0), (0, 50), (0, 0)), 'constant', constant_values=0)

                    xs_i = np.swapaxes(xs_i, 0, 1)
                    xs_i = np.swapaxes(xs_i, 2, 3)


                # ys_i = [ys[0][i: i + batchsize], ys[1][i: i + batchsize]]  # MS regression and classification
                ys_i = ys[i: i + batchsize] # SS
                # print ys_i
                # raw_input('')

                if   yield_mc_info == 0:    yield (list(xs_i), ys_i)
                elif yield_mc_info == 1:    yield (list(xs_i), ys_i) + ({ key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() },)
                elif yield_mc_info == 2:    yield { key: eventInfo[key][i: i + batchsize] for key in eventInfo.keys() }
                else:   raise ValueError("Wrong argument for yield_mc_info (0/1/2)")
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize


def encode_targets(y_dict, batchsize, multiplicity, class_type=None, histo=[], mode='mc'):
    """
    Encodes the labels (classes) of the images.
    :param dict y_dict: Dictionary that contains ALL event class information for the events of a batch.
    :param str class_type: String identifier to specify the exact output classes. i.e. energy_and_position
    :return: ndarray(ndim=2) train_y: Array that contains the encoded class label information of the input events of a batch.
    """


    # MS events:
    if multiplicity == 'SS+MS':

        number_timesteps = 1

        if class_type == 'energy_and_UV_position':
            # train_y = np.zeros((batchsize, number_timesteps, 4), dtype='float32')
            train_y = np.zeros((batchsize, number_timesteps, 1), dtype='float32')
            # train_y = np.zeros((batchsize, 5), dtype='float32')

            # number_cluster_array = np.zeros((batchsize, number_timesteps+1))
            number_cluster_array = np.zeros((batchsize, 1))

            for eventnumber in range(batchsize):
            # for eventnumber in range(5):

                numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])

                # TODO: in x, y, z clustern; dann in u, v, z umrechnen; PCDs mit Depositchannel < 0 NACH clustern
                # TODO: wegwerfen um ihr Energie nicht fuer Berechnung der Cluster Energie zu verwenden
                x, y, z_temp, energy_temp, depositChannel = y_dict['MCPosX'][eventnumber][0:numPCDs], y_dict['MCPosY'][eventnumber][0:numPCDs], y_dict['MCPosZ'][eventnumber][0:numPCDs], y_dict['MCEnergy'][eventnumber][0:numPCDs], y_dict['MCDepositChannel'][eventnumber][0:numPCDs]
                X = []

                for i in range(len(x)):
                    X.append([x[i], y[i], z_temp[i]])

                cluster_distance = 5    # in mm
                clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(X)
                label = clustering.labels_


                x_mean, y_mean, z_mean, energy_sum = [], [], [], []


                histo.append(max(label)+1)

                for j in range(max(label) + 1):

                    mask = np.logical_and(depositChannel >= 0.0, label == j)

                    x_temp = x[mask]
                    y_temp = y[mask]
                    z_temp = z_temp[mask]
                    energy_temp = energy_temp[mask]


                    if len(energy_temp) != 0:
                        x_mean.append(np.average(x_temp, weights=energy_temp))
                        y_mean.append(np.average(y_temp, weights=energy_temp))
                        z_mean.append(np.average(z_temp, weights=energy_temp))
                        energy_sum.append(np.sum(energy_temp))

                while len(x_mean) < number_timesteps:
                    x_mean.append(0.0)
                    y_mean.append(0.0)
                    z_mean.append(0.0)
                    energy_sum.append(0.0)



                x_mean = np.asarray(x_mean)
                y_mean = np.asarray(y_mean)
                z_mean = np.asarray(z_mean)
                energy_sum = np.asarray(energy_sum)


                # x_mean, y_mean to u, v
                if [z_mean > 0.0]:
                    u_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                    v_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                else:
                    u_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                    v_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean

                dtype = [('energy', float), ('U', float), ('V', float), ('Z', float)]
                values = [(energy_sum[i], u_mean[i], v_mean[i], z_mean[i]) for i in range(number_timesteps)]

                target = np.array(values, dtype=dtype)
                target = np.sort(target, order='energy')

                # if len(x_mean) > number_timesteps:
                #     target = target[:,:]
                #     x_mean = x_mean[:5]
                #     y_mean = y_mean[:5]
                #     z_mean = z_mean[:5]
                #     energy_sum = energy_sum[:5]



                for timestep in range(-1, -number_timesteps -1, -1):
                    #[normalize(target[timestep]['energy'], 'energy')]   #,
                    train_y[eventnumber][-timestep - 1] = [normalize(target[timestep]['U'], 'U')]#,
                                                         # normalize(target[timestep]['V'], 'V'),
                                                         # normalize(target[timestep]['Z'], 'Z')]

                number_cluster = np.count_nonzero(train_y[eventnumber])

                # print number_cluster

                # number_cluster -= 1

                # number_cluster_array[eventnumber] = to_categorical(number_cluster, number_timesteps+1)  # convert to one-hot vectors
                number_cluster_array[eventnumber] = number_cluster

                # print target

                # for timestep in range(-1, -number_timesteps - 1, -1):
                #     train_y[eventnumber][-timestep - 1] = target[timestep]['energy']    #normalize(target[timestep]['energy'], 'energy')
                    # train_y[eventnumber][-timestep - 1 +5] = target[timestep]['U']  #normalize(target[timestep]['U'], 'U')
                    # train_y[eventnumber][-timestep - 1 +10] = target[timestep]['V']  #normalize(target[timestep]['V'], 'V')
                    # train_y[eventnumber][-timestep - 1 +15] = target[timestep]['Z']  #normalize(target[timestep]['Z'], 'Z')

        elif class_type == 'U':
            # train_y = np.zeros((batchsize, number_timesteps, 4), dtype='float32')
            train_y = np.zeros((batchsize, number_timesteps, 1), dtype='float32')
            # train_y = np.zeros((batchsize, 5), dtype='float32')

            # number_cluster_array = np.zeros((batchsize, number_timesteps+1))
            number_cluster_array = np.zeros((batchsize, 1))

            for eventnumber in range(batchsize):
            # for eventnumber in range(5):

                numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])

                # TODO: in x, y, z clustern; dann in u, v, z umrechnen; PCDs mit Depositchannel < 0 NACH clustern
                # TODO: wegwerfen um ihr Energie nicht fuer Berechnung der Cluster Energie zu verwenden
                x, y, z_temp, energy_temp, depositChannel = y_dict['MCPosX'][eventnumber][0:numPCDs], y_dict['MCPosY'][eventnumber][0:numPCDs], y_dict['MCPosZ'][eventnumber][0:numPCDs], y_dict['MCEnergy'][eventnumber][0:numPCDs], y_dict['MCDepositChannel'][eventnumber][0:numPCDs]
                X = []

                for i in range(len(x)):
                    X.append([x[i], y[i], z_temp[i]])

                cluster_distance = 3    # in mm
                clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(X)
                label = clustering.labels_


                x_mean, y_mean, z_mean, energy_sum = [], [], [], []


                histo.append(max(label)+1)

                for j in range(max(label) + 1):

                    mask = np.logical_and(depositChannel >= 0.0, label == j)

                    x_temp = x[mask]
                    y_temp = y[mask]
                    z_temp = z_temp[mask]
                    energy_temp = energy_temp[mask]


                    if len(energy_temp) != 0:
                        x_mean.append(np.average(x_temp, weights=energy_temp))
                        y_mean.append(np.average(y_temp, weights=energy_temp))
                        z_mean.append(np.average(z_temp, weights=energy_temp))
                        energy_sum.append(np.sum(energy_temp))

                while len(x_mean) < number_timesteps:
                    x_mean.append(0.0)
                    y_mean.append(0.0)
                    z_mean.append(0.0)
                    energy_sum.append(0.0)



                x_mean = np.asarray(x_mean)
                y_mean = np.asarray(y_mean)
                z_mean = np.asarray(z_mean)
                energy_sum = np.asarray(energy_sum)


                # x_mean, y_mean to u, v
                if [z_mean > 0.0]:
                    u_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                    v_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                else:
                    u_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                    v_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean

                dtype = [('energy', float), ('U', float), ('V', float), ('Z', float)]
                values = [(energy_sum[i], u_mean[i], v_mean[i], z_mean[i]) for i in range(number_timesteps)]

                target = np.array(values, dtype=dtype)
                target = np.sort(target, order='energy')

                # if len(x_mean) > number_timesteps:
                #     target = target[:,:]
                #     x_mean = x_mean[:5]
                #     y_mean = y_mean[:5]
                #     z_mean = z_mean[:5]
                #     energy_sum = energy_sum[:5]



                for timestep in range(-1, -number_timesteps -1, -1):
                    train_y[eventnumber][-timestep - 1] = [normalize(target[timestep]['U'], 'U')]

                # number_cluster = np.count_nonzero(train_y[eventnumber])
                number_cluster = np.count_nonzero(energy_sum)

                if number_cluster > 1:
                    train_y[eventnumber][0] = normalize(1000, 'U')
                    # print normalize(1000, 'U')

                # print number_cluster

                # number_cluster -= 1

                # number_cluster_array[eventnumber] = to_categorical(number_cluster, number_timesteps+1)  # convert to one-hot vectors
                number_cluster_array[eventnumber] = number_cluster

                # print target

                # for timestep in range(-1, -number_timesteps - 1, -1):
                #     train_y[eventnumber][-timestep - 1] = target[timestep]['energy']    #normalize(target[timestep]['energy'], 'energy')
                    # train_y[eventnumber][-timestep - 1 +5] = target[timestep]['U']  #normalize(target[timestep]['U'], 'U')
                    # train_y[eventnumber][-timestep - 1 +10] = target[timestep]['V']  #normalize(target[timestep]['V'], 'V')
                    # train_y[eventnumber][-timestep - 1 +15] = target[timestep]['Z']  #normalize(target[timestep]['Z'], 'Z')

        else:
            raise ValueError('Class type ', class_type, 'not supported for MS events')


    # SS events
    else:

        # u, v, z, energy = [], [], [], []
        #
        # # print '>>>>>', y_dict.keys()
        #
        # # for eventnumber in range(batchsize):
        # for eventnumber in range(50):
        #
        #     if y_dict['CCIsSS'][eventnumber] == 1:
        #
        #         # print y_dict['MCPosU'][eventnumber]
        #
        #         # numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])
        #         numPCDs = int(y_dict['PCDNumberPCDs'][eventnumber])
        #
        #         print '>>>>>'
        #         # print y_dict.keys()
        #         # print y_dict['MCPosU'][eventnumber].shape
        #         # print y_dict['MCPosU'][eventnumber]
        #         # print
        #
        #         depositChannel = y_dict['PCDDepositChannel'][eventnumber][0:numPCDs]
        #         mask_depositChannel = depositChannel >= 0.0
        #
        #         u_temp, v_temp, z_temp, energy_temp = y_dict['PCDPosU'][eventnumber][0:numPCDs], y_dict['PCDPosV'][eventnumber][0:numPCDs], \
        #                                           y_dict['PCDPosZ'][eventnumber][0:numPCDs], \
        #                                           y_dict['PCDEnergy'][eventnumber][0:numPCDs]
        #
        #         u_temp = u_temp[mask_depositChannel]
        #         v_temp = v_temp[mask_depositChannel]
        #         z_temp = z_temp[mask_depositChannel]
        #         energy_temp = energy_temp[mask_depositChannel]
        #
        #         u_temp = np.average(u_temp, weights=energy_temp)
        #         v_temp = np.average(v_temp, weights=energy_temp)
        #         z_temp = np.average(z_temp, weights=energy_temp)
        #         energy_temp = np.sum(energy_temp)
        #
        #         u.append(u_temp)
        #         v.append(v_temp)
        #         z.append(z_temp)
        #         energy.append(energy_temp)
        #
        #         print '>>>'
        #         print
        #         print u_temp, '\t\t', y_dict['MCPosU'][eventnumber]
        #         print v_temp, '\t\t', y_dict['MCPosV'][eventnumber]
        #         print z_temp, '\t\t', y_dict['MCPosZ'][eventnumber]
        #         print energy_temp, '\t\t', y_dict['MCEnergy'][eventnumber]
        #
        #     else:
        #         print
        #         print '>>>>>>>>>>>>> MS event in SS data'
        #
        #
        # u = normalize(np.asarray(u), 'U')
        # v = normalize(np.asarray(v), 'V')
        # z = normalize(np.asarray(z), 'Z')
        # energy = normalize(np.asarray(energy), 'energy')
        #


        if mode == 'data':
            train_y = np.zeros(batchsize, dtype='float32')

        elif class_type == None:
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
            train_y = np.zeros((batchsize, 4), dtype='float32')
            train_y[:, 0] = normalize(y_dict['MCEnergy'][:, 0], 'energy')  # energy
            train_y[:, 1] = normalize(y_dict['MCPosU'][:, 0], 'U')  # dir_u
            train_y[:, 2] = normalize(y_dict['MCPosV'][:, 0], 'V')  # dir_v
            train_y[:, 3] = normalize(y_dict['MCPosZ'][:, 0], 'Z')  # dir_z

            # train_y[:, 0] = normalize(y_dict['MCEnergy'][:], 'energy')  # energy
            # train_y[:, 1] = normalize(y_dict['MCPosU'][:], 'U')  # dir_u
            # train_y[:, 2] = normalize(y_dict['MCPosV'][:], 'V')  # dir_v
            # train_y[:, 3] = normalize(y_dict['MCPosZ'][:], 'Z')  # dir_z

            # train_y[:, 0] = energy
            # train_y[:, 1] = u
            # train_y[:, 2] = v
            # train_y[:, 3] = z

        elif class_type == 'energy':
            # SS from SS data:
            train_y = np.zeros((batchsize, 1), dtype='float32')
            train_y[:, 0] = normalize(y_dict['MCEnergy'][:, 0], 'energy')
            # train_y[:, 0] = normalize(y_dict['MCEnergy'][:], 'energy')

            # SS from SS+MS data:
            # train_y = np.reshape(energy, (-1, 1))

        elif class_type == 'U':
            train_y = np.zeros((batchsize, 1), dtype='float32')


            # train_y[:, 0] = normalize(y_dict['MCPosU'][:, 0], 'U')
            train_y[:, 0] = normalize(y_dict['MCPosU'][:], 'U')

            # SS from SS+MS data:
            # train_y = np.reshape(u, (-1, 1))

        elif class_type == 'V':
            train_y = np.zeros((batchsize, 1), dtype='float32')
            # train_y[:, 0] = normalize(y_dict['MCPosV'][:, 0], 'V')
            train_y[:, 0] = normalize(y_dict['MCPosV'][:], 'V')

            # SS from SS+MS data:
            # train_y = np.reshape(v, (-1, 1))

        elif class_type == 'Z':
            train_y = np.zeros((batchsize, 1), dtype='float32')
            # train_y[:, 0] = normalize(y_dict['MCPosZ'][:, 0], 'Z')
            train_y[:, 0] = normalize(y_dict['MCPosZ'][:], 'Z')

            # SS from SS+MS data:
            # train_y = np.reshape(z, (-1, 1))

        elif class_type == 'position':
            train_y = np.zeros((batchsize, 3), dtype='float32')
            train_y[:, 0] = normalize(y_dict['MCPosU'][:, 0], 'U')
            train_y[:, 1] = normalize(y_dict['MCPosV'][:, 0], 'V')
            train_y[:, 2] = normalize(y_dict['MCPosZ'][:, 0], 'Z')
        # elif class_type == 'time':
        #     train_y = np.zeros((batchsize, 1), dtype='float32')
        #     train_y[:, 0] = y_dict['MCTime'][:, 0]  # time (to calculate dir_z)
        else:
            raise ValueError('Class type ' + str(class_type) + ' not supported!')


    # print 'Counter: ', counter_5

    return train_y
    # return [number_cluster_array, train_y]
    # return number_cluster_array


#  SS
def predict_events(model, generator):
    wf, Y_TRUE, EVENT_INFO = generator.next()

# ====================================================    U bzw V wire auf null setzen:   ==========================================
    wf_shape = np.asarray(wf).shape
    wf = np.asarray(wf)
    #  U
    # wf[0,:,:,:,:] = np.zeros((wf_shape[1], wf_shape[2], wf_shape[3], wf_shape[4]))
    # wf[2,:,:,:,:] = np.zeros((wf_shape[1], wf_shape[2], wf_shape[3], wf_shape[4]))

    #  V
    # wf[1,:,:,:,:] = np.zeros((wf_shape[1], wf_shape[2], wf_shape[3], wf_shape[4]))
    # wf[3,:,:,:,:] = np.zeros((wf_shape[1], wf_shape[2], wf_shape[3], wf_shape[4]))

# ====================================================   alle U - Nachbarchannel NULLEN  ==========================================

    # wf_shape = np.asarray(wf).shape
    # wf = np.asarray(wf)
    #
    # for eventnumber in range(wf_shape[1]):
    #     wf_temp = np.zeros((wf_shape[0], wf_shape[2], wf_shape[3], wf_shape[4]))
    #     #
    #     # max_wire = np.argmax(wf_np[0, event, :, :, :])
    #
    #     max_channel_1_TPC1 = np.amax(wf[0, eventnumber, :, :, :])
    #     max_channel_1_TPC2 = np.amax(wf[2, eventnumber, :, :, :])
    #     tpc = 0
    #
    #     if max_channel_1_TPC1 > max_channel_1_TPC2:
    #         max_channel = max_channel_1_TPC1
    #         arg_max_1 = np.unravel_index(np.argmax(wf[0, eventnumber, :, :, :]), wf[0, eventnumber, :, :, :].shape)
    #         tpc = 0
    #     else:
    #         max_channel = max_channel_1_TPC2
    #         arg_max_1 = np.unravel_index(np.argmax(wf[2, eventnumber, :, :, :]), wf[2, eventnumber, :, :, :].shape)
    #         tpc = 2
    #
    #     i, j, k = arg_max_1
    #     wf_temp[tpc, :, j, :] = wf[tpc, eventnumber, :,j,:]
    #     wf[:, eventnumber, :, :, :] = wf_temp
    #
    # wf = list(wf)

# =====================================================   maximum von Nachbarchannel  ==========================================
#     wf_shape = np.asarray(wf).shape
#     wf = np.asarray(wf)
#
#     wf_max_neighbour = np.zeros(wf_shape[1])
#
#     for eventnumber in range(wf_shape[1]):
#         max_channel_1_TPC1 = np.amax(wf[0, eventnumber, :, :, :])
#         max_channel_1_TPC2 = np.amax(wf[2, eventnumber, :, :, :])
#         tpc = 0
#
#         if max_channel_1_TPC1 > max_channel_1_TPC2:
#             max_channel = max_channel_1_TPC1
#             arg_max_1 = np.unravel_index(np.argmax(wf[0, eventnumber, :, :, :]), wf[0, eventnumber, :, :, :].shape)
#             tpc = 0
#         else:
#             max_channel = max_channel_1_TPC2
#             arg_max_1 = np.unravel_index(np.argmax(wf[2, eventnumber, :, :, :]), wf[2, eventnumber, :, :, :].shape)
#             tpc = 2
#
#         i, j, k = arg_max_1
#
#         try:
#             channel_1plus = np.amax(wf[tpc, eventnumber, i - 25:i + 25, j + 1, k])
#         except:
#             channel_1plus = 0
#
#         try:
#             channel_1minus = np.amax(wf[tpc, eventnumber, i - 25:i + 25, j - 1, k])
#         except:
#             channel_1minus = 0
#
#         wf_max_neighbour[eventnumber] = max([channel_1plus, channel_1minus])
#
#     EVENT_INFO['wf_max_neighbour'] = wf_max_neighbour
#
# =====================

    wf = list(wf)
    Y_PRED = np.asarray(model.predict(wf, 10))
    return (Y_PRED, Y_TRUE, EVENT_INFO)

    # return (Y_PRED, Y_TRUE, EVENT_INFO, wf)


# MS CLassification and regression network
# def predict_events(model, generator):
#     X, Y_TRUE, EVENT_INFO = generator.next()
#     Y_PRED_1, Y_PRED_2 = np.asarray(model.predict(X, 10)[0]), np.asarray(model.predict(X, 10)[1])
#     Y_TRUE_1, Y_TRUE_2 = Y_TRUE[0], Y_TRUE[1]
#     return (Y_PRED_1, Y_PRED_2, Y_TRUE_1, Y_TRUE_2, EVENT_INFO)



def get_events(args, files, model, fOUT, mode='mc'):
    try:
        if args.new: raise IOError
        spec = pickle.load(open(fOUT, "rb"))
        if args.events > len(spec['Y_TRUE']): raise IOError
    except IOError:
        if model == None: print 'model not found and not events file found' ; exit()
        events_per_batch = 50
        if args.events % events_per_batch != 0: raise ValueError('choose event number in multiples of %f events'%(events_per_batch))
        iterations = round_down(args.events, events_per_batch) / events_per_batch
        gen = generate_batches_from_files(files, events_per_batch, multiplicity=args.multiplicity, inputImages=args.inputImages, class_type=args.var_targets, f_size=None, yield_mc_info=1, mode=mode)

        Y_PRED, Y_TRUE = [], []
        # wf = np.asarray(wf)
        for i in xrange(iterations):
            print i * events_per_batch, ' of ', iterations * events_per_batch
            # Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp, wf_temp = predict_events(model, gen)
            Y_PRED_temp, Y_TRUE_temp, EVENT_INFO_temp = predict_events(model, gen)
            Y_PRED.extend(Y_PRED_temp)
            Y_TRUE.extend(Y_TRUE_temp)


            if i == 0:

                EVENT_INFO = EVENT_INFO_temp

                if args.sources == 'wirecheck':
                    wf = wf_temp
                # print np.asarray(wf).shape
            else:
                if args.sources == 'wirecheck':
                    wf = np.append(wf, wf_temp, axis=1)
                # print wf.shape

                for key in EVENT_INFO:
                    EVENT_INFO[key] = np.concatenate((EVENT_INFO[key], EVENT_INFO_temp[key]))

        spec = {'Y_PRED': np.asarray(Y_PRED), 'Y_TRUE': np.asarray(Y_TRUE), 'EVENT_INFO': EVENT_INFO}

        if args.sources == 'wirecheck':
            spec['wf'] = np.asarray(wf)

        EVENT_INFO['Y_TRUE'] = Y_TRUE
        EVENT_INFO['Y_PRED'] = Y_PRED

        write_dict_to_hdf5_file(data=EVENT_INFO, file=fOUT)
        # pickle.dump(spec, open(fOUT, "wb"))
    return spec


def write_dict_to_hdf5_file(data, file, keys_to_write=['all']):
    """
    Write dict to hdf5 file
    :param dict data: dict containing data.
    :param string file: Full filepath of the output hdf5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param list keys_to_write: Keys that will be written to file
    """
    if not isinstance(data, dict) or not isinstance(file, basestring):
        raise TypeError('passed data/file need to be dict/str. Passed type are: %s/%s'%(type(data),type(file)))
    if 'all' in keys_to_write:
        keys_to_write = data.keys()

    fOUT = h5py.File(file, "w")
    for key in keys_to_write:
        print 'writing', key
        if key not in data.keys():
            print keys_to_write, '\n', data.keys()
            raise ValueError('%s not in dict!'%(str(key)))
        fOUT.create_dataset(key, data=np.asarray(data[key]), dtype=np.float32)
    fOUT.close()
    return





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



def read_EventInfo_from_files(files, maxNumEvents=0):
    """
    Returns EventInfo dict from a single/list h5py file(s).
    :param string files: Full filepath of the input h5 file, e.g. '[/path/to/file/file.hdf5]'.
    :return: dict eventInfo: Yields a dict which contains the stored mc/data info.
    """

    if isinstance(files, list): pass
    elif isinstance(files, basestring): files = [files]
    elif isinstance(files, dict): files = reduce(lambda x, y: x + y, files.values())
    else: raise TypeError('passed variable need to be list/np.array/str/dict[dict]')

    if maxNumEvents < 0: raise ValueError('Maximum number of events should be larger 0 (or zero for all)')

    eventInfo = {}
    for idx, filename in enumerate(files):
        f = h5py.File(str(filename), "r")
        for key in f.keys():
            if key in ['wfs', 'gains']: continue
            if idx == 0:
                eventInfo[key] = np.asarray(f[key])
            else:
                eventInfo[key] = np.concatenate((eventInfo[key], np.asarray(f[key])))
        f.close()
        if maxNumEvents > 0 and len(eventInfo.values()[0]) >= maxNumEvents: break
    if maxNumEvents == 0 or len(eventInfo.values()[0]) <= maxNumEvents:
        return eventInfo
    else:
        return { key: value[ 0 : maxNumEvents ] for key,value in eventInfo.items() }



def round_down(num, divisor):
    return num - (num%divisor)

#  # New:
def normalize(data, mode):
    if mode == 'energy':
        data_norm = (data) / 2950.
    elif mode == 'U' or mode == 'V':
        data_norm = (data + 170) / 340.
    elif mode == 'time':
        data_norm = (data - 1030) / 110.
    elif mode == 'Z':
        # data_norm = np.abs(data) / 190      * 1
        data_norm = (data + 190) / 380.
    return data_norm

def denormalize(data, mode):
    if mode == 'energy':
        data_denorm = (data * 2950.)
    elif mode == 'U' or mode == 'V':
        data_denorm = (data * 340.) - 170
    elif mode == 'time':
        data_denorm = (data * 110.) + 1030
    elif mode == 'Z':
        # data_denorm = (data / 1          * 190)
        data_denorm = (data * 380.) - 190
    return data_denorm


# #  Old:
# def normalize(data, mode):
#     if mode == 'energy':
#         data_norm = (data - 550) / 2950     * 2
#     elif mode == 'U' or mode == 'V':
#         data_norm = (data + 170) / 340      * 1
#     elif mode == 'time':
#         data_norm = (data - 1030) / 110     * 1
#     elif mode == 'Z':
#         data_norm = np.abs(data) / 190      * 2
#     return data_norm * 100



