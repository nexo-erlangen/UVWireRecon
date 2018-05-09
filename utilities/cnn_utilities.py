#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Utility functions used for training a CNN."""

import keras as ks

# ------------- Classes -------------#

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
    def __init__(self, display, steps_per_epoch, epoch, folderOUT):
        ks.callbacks.Callback.__init__(self)
        self.seen = 0
        self.display = display
        self.averageLoss = 0
        self.averageAcc = 0
        self.logfile_train_fname = folderOUT + 'log_train.txt'
        self.logfile_train = None
        self.steps_per_epoch = steps_per_epoch
        self.epoch = epoch
        self.loglist = []

    def on_batch_end(self, batch, logs={}):
        self.seen += 1
        self.averageLoss += logs.get('loss')
        self.averageAcc += logs.get("acc")
        if self.seen % self.display == 0:
            averaged_loss = self.averageLoss / self.display
            averaged_acc = self.averageAcc / self.display
            batchnumber_float = (self.seen - self.display / 2.) / float(self.steps_per_epoch) + self.epoch - 1  # start from zero
            self.loglist.append('\n{0}\t{1}\t{2}\t{3}'.format(self.seen, batchnumber_float, averaged_loss, averaged_acc))
            self.averageLoss = 0
            self.averageAcc = 0

    def on_epoch_end(self, batch, logs={}):
        self.logfile_train = open(self.logfile_train_fname, 'a+')
        if os.stat(self.logfile_train_fname).st_size == 0: self.logfile_train.write("#Batch\t#Batch_float\tLoss\tAccuracy")

        for batch_statistics in self.loglist: # only write finished epochs to the .txt
            self.logfile_train.write(batch_statistics)

        self.logfile_train.flush()
        os.fsync(self.logfile_train.fileno())
        self.logfile_train.close()

# ------------- Classes -------------#
