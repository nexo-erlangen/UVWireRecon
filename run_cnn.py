#!/usr/bin/env python

import numpy as np
import h5py
import time
import os
import cPickle as pickle
import keras as ks
from keras import backend as K

from utilities.input_utilities import *
from utilities.generator import *
from utilities.cnn_utilities import *
from plot_scripts.plot_input_plots import *
from models.shared_conv import *

def main():
    args, files = parseInput()

    frac_train = {'unss': 0.80, 'unms': 0.00}
    frac_val   = {'unss': 0.20, 'unms': 0.00}

    splitted_files = splitFiles(args, files, frac_train=frac_train, frac_val=frac_val)

    plotInputCorrelation(args, splitted_files['train'], add='train')
    plotInputCorrelation(args, splitted_files['val'], add='val')

    executeCNN(args, splitted_files, args.var_targets, args.cnn_arch, args.batchsize, (args.num_weights, args.num_epoch), n_gpu=(args.num_gpu, 'avolkov'), mode='train',
               shuffle=(False, None), tb_logger=False)

    print 'final plots \t start'
    # plot.final_plots(folderOUT=args.folderOUT, obs=pickle.load(open(args.folderOUT + "save.p", "rb")))
    print 'final plots \t end'

    print '===================================== Program finished =============================='


def executeCNN(args, files, var_targets, nn_arch, batchsize, epoch, n_gpu=(1, 'avolkov'), mode='train',
               shuffle=(False, None), tb_logger=False):
    """
    Runs a convolutional neural network.
    :param list(tuple) n_bins: Declares the number of bins for each dimension (x,y,z,t) in the train- and testfiles. Can contain multiple n_bins tuples.
                               Multiple n_bins tuples are currently only used for multi-input models with multiple input files per batch.
    :param (int, str) class_type: Declares the number of output classes and a string identifier to specify the exact output classes.
                                  I.e. (2, 'muon-CC_to_elec-CC')
    :param str nn_arch: Architecture of the neural network. Currently, only 'VGG' or 'WRN' are available.
    :param int batchsize: Batchsize that should be used for the cnn.
    :param int epoch: Declares if a previously trained model or a new model (=0) should be loaded.
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param str mode: Specifies what the function should do - train & test a model or evaluate a 'finished' model?
                     Currently, there are two modes available: 'train' & 'eval'.
    :param None/str swap_4d_channels: For 4D data input (3.5D models). Specifies, if the channels for the 3.5D net should be swapped.
                                      Currently available: None -> XYZ-T ; 'yzt-x' -> YZT-X
    :param bool use_scratch_ssd: Declares if the input files should be copied to the node-local SSD scratch before executing the cnn.
    :param bool zero_center: Declares if the input images ('xs') should be zero-centered before training.
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param bool tb_logger: Declares if a tb_callback should be used during training (takes longer to train due to overhead!).
    :param str str_ident: Optional str identifier that gets appended to the modelname. Useful when training models which would have the same modelname.
                          Also used for defining models and projections!
    :param (str, str) loss_opt: tuple that contains 1) the loss and 2) the metric during training.
    """
    print epoch
    if epoch[0] == 0:
        if nn_arch is 'DCNN':
            model = create_shared_dcnn_network()

        elif nn_arch is 'ResNet':
            raise ValueError('Currently, this is not implemented')
            # model = create_vgg_like_model(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
            #                               n_filters=(64, 64, 64, 64, 64, 64, 128, 128, 128, 128),
            #                               swap_4d_channels=swap_4d_channels)

        elif nn_arch is 'Inception':
            # model = create_convolutional_lstm(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
            #                                   n_filters=(16, 16, 32, 32, 32, 32, 64, 64))
            raise ValueError('Currently, this is not implemented')
        elif nn_arch is 'Conv_LSTM':
            raise ValueError('Currently, this is not implemented')
            # model = create_convolutional_lstm(n_bins, batchsize, nb_classes=class_type[0], dropout=0.1,
            #                                   n_filters=(16, 16, 32, 32, 32, 32, 64, 64))
        else:
            raise ValueError('Currently, only "DCNN" are available as nn_arch')
    else:
        raise ValueError('Check, if loading models is implemented yet')
        # model = load_trained_model(args)
        # model = ks.models.load_model(
        #     'models/trained/trained_' + modelname + '_epoch_' + str(epoch[0]) + '_file_' + str(epoch[1]) + '.h5')

    # plot model, install missing packages with conda install if it throws a module error
    try:
        ks.utils.plot_model(model, to_file=args.folderOUT+'/plot_model.png', show_shapes=True, show_layer_names=True)
    except ImportError:
        print 'could not produce plot_model.png ---- try on CPU'

    adam = ks.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)  # epsilon=1 for deep networks
    optimizer = adam  # Choose optimizer, only used if epoch == 0

    # model, batchsize = parallelize_model_to_n_gpus(model, n_gpu, batchsize)  # TODO compile after restart????
    # if n_gpu[0] > 1: model.compile(loss=loss_opt[0], optimizer=optimizer, metrics=[loss_opt[1]])  # TODO check

    model.summary()

    if epoch[0] == 0:
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error'])

    print "\nFirst Epoch:\t", epoch

    if mode == 'train':
        model.save(args.folderOUT + "models/model_initial.hdf5")
        # lr = None
        for epoch_i in xrange(epoch[0], epoch[1]+1):
            print 'Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch_i)
            # TODO implement lr rate schedule
            # lr, lr_decay = schedule_learning_rate(model, epoch_i, n_gpu, args.splitted_files['train'], lr_initial=0.003, manual_mode=(False, None, 0.0, None))
            # lr, lr_decay = schedule_learning_rate(model, epoch_i, n_gpu, train_files, lr_initial=0.003,
            #                                       manual_mode=(True, 0.0006, 0.07, lr))
            history_train = fit_model(args, model, files, batchsize, var_targets, epoch_i, shuffle, n_events=None, tb_logger=tb_logger)
            history_test = evaluate_model(args, model, files['val'], batchsize, var_targets, epoch_i, n_events=None)

            # save_train_and_test_statistics_to_txt(model, history_train, history_test, lr_decay, epoch_i,
            #                                       files, batchsize, var_targets)
            # plot_train_and_test_statistics(modelname)
            # plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels, modelname, epoch[0], file_no, str_ident)
        model.save(args.folderOUT + "models/model_final.hdf5")
    elif mode == 'eval':
        raise ValueError('Check, if model evaluation is implemented yet')
        # After training is finished, investigate model performance
        # arr_energy_correct = make_performance_array_energy_correct(model, test_files[0][0], n_bins, class_type,
        #                                                            batchsize, xs_mean, swap_4d_channels, str_ident,
        #                                                            samples=None)
        # np.save('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy', arr_energy_correct)
        #
        # arr_energy_correct = np.load('results/plots/saved_predictions/arr_energy_correct_' + modelname + '.npy')
        # # arr_energy_correct = np.load('results/plots/saved_predictions/backup/arr_energy_correct_model_VGG_4d_xyz-t-tight-1-w-geo-fix_and_yzt-x-tight-1-wout-geo-fix_and_4d_xyzt_muon-CC_to_elec-CC_double_input_single_train.npy')
        # make_energy_to_accuracy_plot_multiple_classes(arr_energy_correct,
        #                                               title='Classification for muon-CC_and_elec-CC_3-100GeV',
        #                                               filename='results/plots/PT_' + modelname,
        #                                               compare_pheid=True)  # TODO think about more automatic savenames
        # make_prob_hists(arr_energy_correct[:, ], modelname=modelname, compare_pheid=True)
        # make_property_to_accuracy_plot(arr_energy_correct, 'bjorken-y',
        #                                title='Bjorken-y distribution vs Accuracy, 3-100GeV',
        #                                filename='results/plots/PT_bjorken_y_vs_accuracy' + modelname, e_cut=False,
        #                                compare_pheid=True)
        #
        # make_hist_2d_property_vs_property(arr_energy_correct, modelname,
        #                                   property_types=('bjorken-y', 'probability'), e_cut=(3, 100),
        #                                   compare_pheid=True)
        # calculate_and_plot_correlation(arr_energy_correct, modelname, compare_pheid=True)
    else:
        raise ValueError('chosen mode not available. Choose between train/eval')


# ----------------------------------------------------------
# Define model
# ----------------------------------------------------------
def schedule_learning_rate(model, epoch, lr_initial=0.001, n_gpu=1, manual_mode=(False, None, 0.0, None)):
    """
    Function that schedules a learning rate during training.
    If manual_mode[0] is False, the current lr will be automatically calculated if the training is resumed, based on the epoch variable.
    If manual_mode[0] is True, the final lr during the last training session (manual_mode[1]) and the lr_decay (manual_mode[1])
    have to be set manually.
    :param Model model: Keras nn model instance. Used for setting the lr.
    :param int epoch: The epoch number at which this training session is resumed (last finished epoch).
    :param (int/str) n_gpu: Number of gpu's that the model should be parallelized to [0] and the multi-gpu mode (e.g. 'avolkov').
    :param list train_files: list of tuples that contains the trainfiles and their number of rows (filepath, f_size).
    :param float lr_initial: Initial lr that is used with the automatic mode. Typically 0.01 for SGD and 0.001 for Adam.
    :param (bool, None/float, float, None/float) manual_mode: Tuple that controls the options for the manual mode.
            manual_mode[0] = flag to enable the manual mode, manual_mode[1] = lr value, of which the mode should start off
            manual_mode[2] = lr_decay during epochs, manual_mode[3] = current lr, only used to check if this is the first instance of the while loop
    :return: int epoch: The epoch number of the new epoch (+= 1).
    :return: float lr: Learning rate that has been set for the model and for this epoch.
    :return: float lr_decay: Learning rate decay that has been used to decay the lr rate used for this epoch.
    """

    if manual_mode[0] is True:
        raise ValueError('Currently, no manuel_mode == True implemented')
        # lr = manual_mode[1] if manual_mode[3] is None else K.get_value(model.optimizer.lr)
        # lr_decay = manual_mode[2]
        # K.set_value(model.optimizer.lr, lr)
        #
        # if epoch[0] > 1 and lr_decay > 0:
        #     lr *= 1 - float(lr_decay)
        #     K.set_value(model.optimizer.lr, lr)
        #     print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
        #           ' before epoch ' + str(epoch[0]) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'
    else:
        if epoch == 0:
            lr, lr_decay = lr_initial, 0.00
            #lr, lr_decay = lr_initial * n_gpu[0], 0.00
            K.set_value(model.optimizer.lr, lr)
            print 'Set learning rate to ' + str(K.get_value(model.optimizer.lr)) + ' before epoch ' + str(epoch)
        else:
            lr, lr_decay = get_new_learning_rate(epoch, lr_initial, n_gpu)
            K.set_value(model.optimizer.lr, lr)
            print 'Decayed learning rate to ' + str(K.get_value(model.optimizer.lr)) + \
                  ' before epoch ' + str(epoch) + ' (minus ' + '{:.1%}'.format(lr_decay) + ')'

    return epoch, lr, lr_decay

def get_new_learning_rate(epoch, lr_initial, n_gpu):
    """
    Function that calculates the current learning rate based on the number of already trained epochs.
    Learning rate schedule is as follows: lr_decay = 7% for lr > 0.0003
                                          lr_decay = 4% for 0.0003 >= lr > 0.0001
                                          lr_decay = 2% for 0.0001 >= lr
    :param int epoch: The number of the current epoch which is used to calculate the new learning rate.
    :param float lr_initial: Initial lr for the first epoch. Typically 0.01 for SGD and 0.001 for Adam.
    :param int n_gpu: number of gpu's that are used during the training. Used for scaling the lr.
    :return: float lr_temp: Calculated learning rate for this epoch.
    :return: float lr_decay: Latest learning rate decay used.
    """
    n_lr_decays = epoch - 1
    lr_temp = lr_initial # * n_gpu TODO think about multi gpu lr
    lr_decay = None

    for i in xrange(n_lr_decays):

        if lr_temp > 0.0003:
            lr_decay = 0.07
        elif 0.0003 >= lr_temp > 0.0001:
            lr_decay = 0.04
        else:
            lr_decay = 0.02

        lr_temp = lr_temp * (1 - float(lr_decay))

    return lr_temp, lr_decay

def fit_model(args, model, files, batchsize, var_targets, epoch, shuffle, n_events=None, tb_logger=False):
    """
    Trains a model based on the Keras fit_generator method.
    If a TensorBoard callback is wished, validation data has to be passed to the fit_generator method.
    For this purpose, the first file of the test_files is used.
    :param ks.model.Model/Sequential model: Keras model of a neural network.
    :param str modelname: Name of the model.
    :param str f: full filepath of the file that should be used for training.
    :param int f_size: number of images contained in f.
    :param int file_no: if the full data is split into multiple files, this param indicates the file number.
    :param list test_files: list of tuples that contains the testfiles and their number of rows for the tb_callback.
    :param int batchsize: Batchsize that is used in the fit_generator method.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) in both the train- and test_files. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Epoch of the model if it has been trained before.
    :param (bool, None/int) shuffle: Declares if the training data should be shuffled before the next training epoch.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str str_ident: string identifier for the projection type / model input that is parsed to the image generator. Needed for some specific models.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for training.
    :param bool tb_logger: Declares if a tb_callback during fit_generator should be used (takes long time to save the tb_log!).
    """
    callbacks = []
    if tb_logger is True:
        raise ValueError('Currently, no Tensorboard Logger implemented')
        # boardwrapper = TensorBoardWrapper(generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, str_ident, zero_center_image=xs_mean),
        #                              nb_steps=int(5000 / batchsize), log_dir='models/trained/tb_logs/' + modelname + '_{}'.format(time.time()),
        #                              histogram_freq=1, batch_size=batchsize, write_graph=False, write_grads=True, write_images=True)
        # callbacks.append(boardwrapper)
        # validation_data = generate_batches_from_hdf5_file(test_files[0][0], batchsize, n_bins, class_type, str_ident, swap_col=swap_4d_channels, zero_center_image=xs_mean) #f_size=None is ok here
        # validation_steps = int(min([ getNumEvents(files['val']), 5000 ]) / batchsize)
    else:
        # validation_data, validation_steps, callbacks = None, None, []
        validation_data = generate_batches_from_files(files['val'], batchsize, var_targets)
        validation_steps = int(min([getNumEvents(files['val']), 5000]) / batchsize)

    train_steps_per_epoch = int(getNumEvents(files['train']) / batchsize)

    # logger = BatchLevelPerformanceLogger(display=100, steps_per_epoch=train_steps_per_epoch, epoch=epoch, folderOUT=args.folderOUT)
    # callbacks.append(logger)

    print 'Training in epoch', epoch, 'on events:', train_steps_per_epoch*batchsize

    history = model.fit_generator(
        generate_batches_from_files(files['train'], batchsize, var_targets),
        steps_per_epoch=train_steps_per_epoch,
        epochs=1,
        initial_epoch=epoch,
        verbose=1,
        max_queue_size=10,
        validation_data=validation_data,
        validation_steps=validation_steps)
        # callbacks=callbacks)
    model.save_weights(args.folderOUT + "models/weights_epoch_" + str(epoch) + ".hdf5")

    return history

def evaluate_model(args, model, files, batchsize, var_targets, epoch, n_events=None):
    """
    Evaluates a model with validation data based on the Keras evaluate_generator method.
    :param ks.model.Model/Sequential model: Keras model (trained) of a neural network.
    :param str modelname: Name of the model.
    :param list test_files: list of tuples that contains the testfiles and their number of rows.
    :param int batchsize: Batchsize that is used in the evaluate_generator method.
    :param list(tuple) n_bins: Number of bins for each dimension (x,y,z,t) in the test_files. Can contain multiple n_bins tuples.
    :param (int, str) class_type: Tuple with the number of output classes and a string identifier to specify the output classes.
    :param ndarray xs_mean: mean_image of the x (train-) dataset used for zero-centering the test data.
    :param int epoch: Current epoch of the training.
    :param None/int swap_4d_channels: For 3.5D, param for the gen to specify, if the default channel (t) should be swapped with another dim.
    :param str str_ident: string identifier for the projection type / model input that is parsed to the image generator. Needed for some specific models.
    :param None/int n_events: For testing purposes if not the whole .h5 file should be used for evaluating.
    """

    history = None

    # for i, (f, f_size) in enumerate(test_files):
    #     print 'Testing on file ', i, ',', str(f)
    #
    #     if n_events is not None: f_size = n_events  # for testing
    #
    #     history = model.evaluate_generator(
    #         generate_batches_from_hdf5_file(f, batchsize, n_bins, class_type, str_ident, swap_col=swap_4d_channels, f_size=f_size, zero_center_image=xs_mean),
    #         steps=int(f_size / batchsize), max_queue_size=10)
    #     print 'Test sample results: ' + str(history) + ' (' + str(model.metrics_names) + ')'
    #
    #     logfile_fname = 'models/trained/perf_plots/log_test_' + modelname + '.txt'
    #     logfile = open(logfile_fname, 'a+')
    #     if os.stat(logfile_fname).st_size == 0: logfile.write('#Epoch\tLoss\tAccuracy\n')
    #     logfile.write(str(epoch) + '\t' + str(history[0]) + '\t' + str(history[1]) + '\n')

    return history

def save_train_and_test_statistics_to_txt(model, history_train, history_test, lr, lr_decay,
                                          epoch, files, batchsize, var_targets):
    """
    Function for saving various information during training and testing to a .txt file.
    """

    with open('models/trained/train_logs/log_' + modelname + '.txt', 'a+') as f_out:
        f_out.write('--------------------------------------------------------------------------------------------------------\n')
        f_out.write('\n')
        f_out.write('Current time: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()) + '\n')
        f_out.write('Decayed learning rate to ' + str(lr) + ' before epoch ' + str(epoch) + ' (minus ' + str(lr_decay) + ')\n')
        f_out.write('Trained in epoch ' + str(epoch) + ' on file ' + str(train_file_no) + ', ' + str(train_file) + '\n')
        f_out.write('Tested in epoch ' + str(epoch) + ', file ' + str(train_file_no) + ' on test_files ' + str(test_files) + '\n')
        f_out.write('History for training and testing: \n')
        f_out.write('Train: ' + str(history_train.history) + '\n')
        f_out.write('Test: ' + str(history_test) + ' (' + str(model.metrics_names) + ')' + '\n')
        f_out.write('\n')
        f_out.write('Additional Info:\n')
        f_out.write('Batchsize=' + str(batchsize) + ', n_bins=' + str(n_bins) +
                    ', class_type=' + str(class_type) + '\n' +
                    'swap_4d_channels=' + str(swap_4d_channels) + ', str_ident=' + str_ident + '\n')
        f_out.write('\n')

def load_trained_model(args):
    print "===================================== load Model ==============================================="
    try:
        print "%smodels/(model/weights)-%s.hdf5" % (args.folderMODEL, args.nb_weights)
        print "================================================================================================\n"
        try:
            model = k.models.load_model(args.folderMODEL + "models/model-initial.hdf5")
            model.load_weights(args.folderMODEL + "models/weights-" + args.nb_weights + ".hdf5")
        except:
            model = k.models.load_model(args.folderMODEL + "models/model-" + args.nb_weights + ".hdf5")
        os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
        if args.nb_weights=='final':
            epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
            print epoch_start
        else:
            epoch_start = 1+int(args.nb_weights)
    except:
        print "\t\tMODEL NOT FOUND!\n"
        exit()

# def get_model(args):
    # def def_model():
    #     from keras.models import Sequential
    #     from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
    #     from keras.regularizers import l2, l1, l1l2
    #
    #     init = "glorot_uniform"
    #     activation = "relu"
    #     padding = "same"
    #     regul = l2(1.e-2)
    #     model = Sequential()
    #     # convolution part
    #     model.add(Convolution2D(16, 5, 3, border_mode=padding, init=init, W_regularizer=regul, input_shape=(1024, 76, 1)))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((4, 2), border_mode=padding))
    #
    #     model.add(Convolution2D(32, 5, 3, border_mode=padding, init=init, W_regularizer=regul))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((4, 2), border_mode=padding))
    #
    #     model.add(Convolution2D(64, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((2, 2), border_mode=padding))
    #
    #     model.add(Convolution2D(128, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((2, 2), border_mode=padding))
    #
    #     model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((2, 2), border_mode=padding))
    #
    #     model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
    #     model.add(Activation(activation))
    #     model.add(MaxPooling2D((2, 2), border_mode=padding))
    #
    #     # regression part
    #     model.add(Flatten())
    #     model.add(Dense(32, activation=activation, init=init, W_regularizer=regul))
    #     model.add(Dense(8, activation=activation, init=init, W_regularizer=regul))
    #     model.add(Dense(1 , activation=activation, init=init))
    #     return model

    # if not args.resume:
    #     from keras import optimizers
    #     print "===================================== new Model =====================================\n"
    #     model = def_model()
    #     epoch_start = 0
    #     # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #normal
    #     optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=(1.+1.e-5)) #test1
    #     model.compile(
    #         loss='mean_squared_error',
    #         optimizer=optimizer,
    #         metrics=['mean_absolute_error'])
    # else:
    #
    # print "\nFirst Epoch:\t", epoch_start
    # print model.summary(), "\n"
    # print "\n"
    # return model, epoch_start

# ----------------------------------------------------------
# OLD
# ----------------------------------------------------------
def predict_energy(model, generator):
    E_CNN_wfs, E_True = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:,0])
    return (E_CNN, E_True)

def predict_energy_reconstruction(model, generator):
    E_CNN_wfs, E_True, E_EXO, isSS = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:, 0])
    return (E_CNN, E_True, E_EXO, isSS)

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
