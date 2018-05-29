#!/usr/bin/env python

import numpy as np
import h5py
import time
import os
import cPickle as pickle
from sys import path
path.append('/home/vault/capm/sn0515/PhD/Th_U-Wire/Scripts')
import script_plot as plot

def main():
    args, files = make_organize()
    # frac_train = {'thss': 0.0, 'thms': 0.975, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.0} #th training
    # frac_val   = {'thss': 0.0, 'thms': 0.025, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.0}
    #frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.95} #normal
    #frac_val   = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.05}
    # # frac_train = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.8}  # normal + test
    # # frac_val = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.2}
    frac_train = {'thss': 0.0, 'thms': 1.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 1.00} #for energy plot
    frac_val   = {'thss': 0.0, 'thms': 0.0, 'rass': 0.0, 'rams': 0.0, 'coss': 0.0, 'coms': 0.0, 'gass': 0.0, 'gams': 0.0, 'unss': 0.0, 'unms': 0.00}
    splitted_files = split_data(args, files, frac_train=frac_train, frac_val=frac_val)

    plot.get_energy_spectrum_mixed(args, splitted_files['train'], add='train')
    plot.get_energy_spectrum_mixed(args, splitted_files['val'], add='val')
    plot.get_energy_spectrum_mixed(args, files, add='all')
    exit()

    train_model(args, splitted_files, get_model(args), args.nb_batch * args.nb_GPU)

    print 'final plots \t start'
    plot.final_plots(folderOUT=args.folderOUT, obs=pickle.load(open(args.folderOUT + "save.p", "rb")))
    print 'final plots \t end'

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------

def predict_energy_reconstruction(model, generator):
    E_CNN_wfs, E_True, E_EXO, isSS = generator.next()
    E_CNN = np.asarray(model.predict(E_CNN_wfs, 100)[:, 0])
    return (E_CNN, E_True, E_EXO, isSS)

# ----------------------------------------------------------
# Define model
# ----------------------------------------------------------
def get_model(args):
    def def_model():
        from keras.models import Sequential
        from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
        from keras.regularizers import l2, l1, l1l2

        init = "glorot_uniform"
        activation = "relu"
        padding = "same"
        regul = l2(1.e-2)
        model = Sequential()
        # convolution part
        model.add(Convolution2D(16, 5, 3, border_mode=padding, init=init, W_regularizer=regul, input_shape=(1024, 76, 1)))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))

        model.add(Convolution2D(32, 5, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((4, 2), border_mode=padding))

        model.add(Convolution2D(64, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(128, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        model.add(Convolution2D(256, 3, 3, border_mode=padding, init=init, W_regularizer=regul))
        model.add(Activation(activation))
        model.add(MaxPooling2D((2, 2), border_mode=padding))

        # regression part
        model.add(Flatten())
        model.add(Dense(32, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(8, activation=activation, init=init, W_regularizer=regul))
        model.add(Dense(1 , activation=activation, init=init))
        return model

    if not args.resume:
        from keras import optimizers
        print "===================================== new Model =====================================\n"
        model = def_model()
        epoch_start = 0
        # optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) #normal
        optimizer = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=(1.+1.e-5)) #test1
        model.compile(
            loss='mean_squared_error',
            optimizer=optimizer,
            metrics=['mean_absolute_error'])
    else:
        from keras.models import load_model
        print "===================================== load Model ==============================================="
        try:
            print "%smodels/(model/weights)-%s.hdf5" % (args.folderMODEL, args.nb_weights)
            print "================================================================================================\n"
            try:
                model = load_model(args.folderMODEL + "models/model-initial.hdf5")
                model.load_weights(args.folderMODEL + "models/weights-" + args.nb_weights + ".hdf5")
            except:
                model = load_model(args.folderMODEL + "models/model-" + args.nb_weights + ".hdf5")
            os.system("cp %s %s" % (args.folderMODEL + "history.csv", args.folderOUT + "history.csv"))
            if args.nb_weights=='final':
                epoch_start = 1+int(np.genfromtxt(args.folderOUT+'history.csv', delimiter=',', names=True)['epoch'][-1])
                print epoch_start
            else:
                epoch_start = 1+int(args.nb_weights)
        except:
            print "\t\tMODEL NOT FOUND!\n"
            exit()
    print "\nFirst Epoch:\t", epoch_start
    print model.summary(), "\n"
    print "\n"
    return model, epoch_start

# ----------------------------------------------------------
# Training
# ----------------------------------------------------------
def train_model(args, files, (model, epoch_start), batchSize):
    from keras import callbacks
    start = time.time()

    if args.nb_GPU>1:
        model = make_parallel(model, args.nb_GPU)

    model.save(args.folderOUT + "models/model-initial.hdf5")
    model.save_weights(args.folderOUT + "models/weights-initial.hdf5")
    print 'training los'
    model.fit_generator(
        generate_batch_mixed(gen_train, batchSize, numEvents_train),
        samples_per_epoch=plot.round_down(sum(numEvents_train.values()), batchSize),
        nb_epoch=args.nb_epoch+epoch_start,
        verbose=1,
        validation_data=generate_batch_mixed(gen_val, batchSize, numEvents_val),
        nb_val_samples=plot.round_down(sum(numEvents_val.values())  , batchSize),
        initial_epoch=epoch_start,
        callbacks=[
            callbacks.CSVLogger(args.folderOUT + 'history.csv', append=args.resume),
            callbacks.ModelCheckpoint(args.folderOUT + 'models/weights-{epoch:03d}.hdf5', save_weights_only=True, period=int(args.nb_epoch/100)),
            Histories(args, files)
        ])
    print 'training stop'
    model.save(args.folderOUT+"models/model-final.hdf5")
    model.save_weights(args.folderOUT+"models/weights-final.hdf5")

    end = time.time()
    print "\nElapsed time:\t%.2f minutes\tor rather\t%.2f hours\n" % (((end-start)/60.),((end-start)/60./60.))

    print 'Model performance\tloss\t\tmean_abs_err'
    print '\tTrain:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['train'].values()).tolist()), batchSize), val_samples=128))
    print '\tValid:\t\t%.4f\t%.4f'    % tuple(model.evaluate_generator(generate_batch(generate_event(np.concatenate(files['val'].values()).tolist())  , batchSize), val_samples=128))
    return model

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
