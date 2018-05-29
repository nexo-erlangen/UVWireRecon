#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import keras as ks
import os

class TensorBoardWrapper(ks.callbacks.TensorBoard):
    """Up to now (05.10.17), Keras doesn't accept TensorBoard callbacks with validation data that is fed by a generator.
     Supplying the validation data is needed for the histogram_freq > 1 argument in the TB callback.
     Without a workaround, only scalar values (e.g. loss, accuracy) and the computational graph of the model can be saved.

     This class acts as a Wrapper for the ks.callbacks.TensorBoard class in such a way,
     that the whole validation data is put into a single array by using the generator.
     Then, the single array is used in the validation steps. This workaround is experimental!"""
    def __init__(self, batch_gen, nb_steps, **kwargs):
        super(TensorBoardWrapper, self).__init__(**kwargs)
        self.batch_gen = batch_gen # The generator.
        self.nb_steps = nb_steps   # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs):
        # Fill in the `validation_data` property.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        imgs, tags = None, None
        for s in xrange(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros(((self.nb_steps * ib.shape[0],) + ib.shape[1:]), dtype=np.float32)
                tags = np.zeros(((self.nb_steps * tb.shape[0],) + tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super(TensorBoardWrapper, self).on_epoch_end(epoch, logs)


class BatchLevelPerformanceLogger(ks.callbacks.Callback):
    # Gibt loss aus über alle :display batches, gemittelt über die letzten :display batches
    def __init__(self, display, steps_per_epoch, args):
        ks.callbacks.Callback.__init__(self)
        self.seen = 0
        self.display = display
        self.averageLoss = 0
        self.averageMAE = 0
        self.averageValLoss = 0
        self.averageValMAE = 0
        self.args = args
        self.logfile_train_fname = self.args.folderOUT + 'log_train.txt'
        self.logfile_train = None
        self.steps_per_epoch = steps_per_epoch
        # self.epoch = epoch

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageMAE += logs.get('mean_absolute_error')
        self.averageValLoss += logs.get('val_loss')
        self.averageValMAE += logs.get('val_mean_absolute_error')
        if self.seen % self.display == 0:
            averaged_loss = self.averageLoss / self.display
            averaged_mae = self.averageMAE / self.display
            averaged_val_loss = self.averageValLoss / self.display
            averaged_val_mae = self.averageValMAE / self.display
            batchnumber_float = (self.seen - self.display / 2.) / float(self.steps_per_epoch) # + self.epoch - 1  # start from zero
            self.loglist.append('\n{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(self.seen, batchnumber_float, averaged_loss, averaged_mae, averaged_val_loss, averaged_val_mae))
            self.averageLoss = 0
            self.averageMAE = 0
            self.averageValLoss = 0
            self.averageValMAE = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.loglist = []

    def on_epoch_end(self, epoch, logs={}):
        self.logfile_train = open(self.logfile_train_fname, 'a+')
        if os.stat(self.logfile_train_fname).st_size == 0: self.logfile_train.write("#Batch\t#Batch_float\tLoss\tMAE\tVal-Loss\tVal-MAE")

        for batch_statistics in self.loglist: # only write finished epochs to the .txt
            self.logfile_train.write(batch_statistics)

        self.logfile_train.flush()
        os.fsync(self.logfile_train.fileno())
        self.logfile_train.close()


class EpochLevelPerformanceLogger(ks.callbacks.Callback):
    def __init__(self, args, files):
        ks.callbacks.Callback.__init__(self)
        self.validation_data = None
        self.args = args
        self.files = files
        self.events_val = 10000
        self.events_per_batch = 1000
        self.val_iterations = plot.round_down(self.events_val, self.events_per_batch) / self.events_per_batch
        # self.gen_val = generate_batch_reconstruction(generate_event_reconstruction(np.concatenate(self.files['val'].values()).tolist()), self.events_per_batch)
        self.gen_val = generate_batch_reconstruction(generate_event_reconstruction(self.files['val']['thms']+self.files['test']['thms']), self.events_per_batch)

    def on_train_begin(self, logs={}):
        self.losses = []
        if self.args.resume: os.system("cp %s %s" % (self.args.folderMODEL + "save.p", self.args.folderOUT + "save.p"))
        else: pickle.dump({}, open(self.args.folderOUT + "save.p", "wb"))
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        E_CNN, E_EXO, E_True, isSS = [], [], [], []
        for i in xrange(self.val_iterations):
            E_CNN_temp, E_True_temp, E_EXO_temp, isSS_temp = predict_energy_reconstruction(self.model, self.gen_val)
            E_True.extend(E_True_temp)
            E_CNN.extend(E_CNN_temp)
            E_EXO.extend(E_EXO_temp)
            isSS.extend(isSS_temp)
        dataIn = {'E_CNN': np.asarray(E_CNN), 'E_EXO': np.asarray(E_EXO), 'E_True': np.asarray(E_True), 'isSS': np.asarray(isSS)}
        obs = plot.make_plots(self.args.folderOUT, dataIn=dataIn, epoch=str(epoch), sources='th', position='S5')
        self.dict_out = pickle.load(open(self.args.folderOUT + "save.p", "rb"))
        self.dict_out[str(epoch)] = {'E_CNN': E_CNN, 'E_True': E_True, 'E_EXO': E_EXO,
                                     'peak_pos': obs['peak_pos'],
                                     'peak_sig': obs['peak_sig'],
                                     'resid_pos': obs['resid_pos'],
                                     'resid_sig': obs['resid_sig'],
                                     'loss': logs['loss'], 'mean_absolute_error': logs['mean_absolute_error'],
                                     'val_loss': logs['val_loss'], 'val_mean_absolute_error': logs['val_mean_absolute_error']}
        pickle.dump(self.dict_out, open(self.args.folderOUT + "save.p", "wb"))
        plot.final_plots(folderOUT=self.args.folderOUT, obs=pickle.load(open(self.args.folderOUT + "save.p", "rb")))

        # plot_train_and_test_statistics(modelname)
        # plot_weights_and_activations(test_files[0][0], n_bins, class_type, xs_mean, swap_4d_channels, modelname,
        #                              epoch[0], file_no, str_ident)
        plot_traininghistory(args)

        return