#!/usr/bin/env python

import numpy as np
import h5py
import time
import os
from os import listdir
from os.path import isfile,join

def main():
    args, files = make_organize()

    for f in files:
        print f

    test_model(args, files)

    print '===================================== Program finished =============================='

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
    import argparse
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('-in', dest='folderIN', default='/home/vault/capm/sn0515/PhD/DeepLearning/hdf5-test/files_new_4_1/', help='folderIN Path')
    parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
    args, unknown = parser.parse_known_args()

    files = [os.path.join(args.folderIN, f) for f in listdir(args.folderIN) if isfile(join(args.folderIN, f))]

    return args, files

def generate_event(files):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), 'r')
            X_EXO_i = np.asarray(f.get('reconEnergy'))
            wfs1_i = np.asarray(f.get('wfs1'))
            wfs2_i = np.asarray(f.get('wfs2'))
            f.close()
            lst = range(len(X_EXO_i[:,0]))
            random.shuffle(lst)
            for i in lst:
                X_EXO = X_EXO_i[i]
                wfs1 = wfs1_i[i]
                wfs2 = wfs2_i[i]
                yield (wfs1, wfs2, X_EXO)

def generate_batch(generator, batchSize):
    while 1:
        X, Y, Z = [], [], []
        for i in xrange(batchSize):
            temp = generator.next()
            X.append(temp[0])
            Y.append(temp[1])
            Z.append(temp[2])
        yield (np.asarray(X), np.asarray(Y), np.asarray(Z))

def gen_batches_from_single_file_1(files, batchsize):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), "r")
            f_size = num_events([filename])

            n_entries = 0
            while n_entries <= (f_size - batchsize):
                xs1 = f['wfs1'][n_entries: n_entries + batchsize]
                xs2 = f['wfs2'][n_entries: n_entries + batchsize]
                y_values = f['reconEnergy'][n_entries:n_entries + batchsize]

                yield xs1, xs2, y_values
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def gen_batches_from_single_file_2(files, batchsize):
    import random
    while 1:
        random.shuffle(files)
        for filename in files:
            f = h5py.File(str(filename), "r")
            f_size = num_events([filename])
            lst = np.arange(0, f_size, batchsize)
            random.shuffle(lst)

            for i in lst:
                xs1 = f['wfs1'][i: i + batchsize]
                xs2 = f['wfs2'][i: i + batchsize]
                y_values = f['reconEnergy'][i:i + batchsize]

                yield xs1, xs2, y_values
            f.close()  # this line of code is actually not reached if steps=f_size/batchsize

def num_events(files):
    counter = 0
    for filename in files:
        f = h5py.File(str(filename), 'r')
        counter += np.asarray(f['eventInfo'][:,0]).size
        f.close()
    return counter

# -----------------------------------------------------------
# Training
# ----------------------------------------------------------
def test_model(args, files):
    repeat = 10
    duration = []
    print 'num events:', num_events(files)
    batches = xrange(num_events(files)//args.nb_batch)
    print 'batches:', batches

    print 'start timing test'
    for i in xrange(repeat):
        start = time.time()
        # gen = generate_batch(generate_event(files), args.nb_batch)
        gen = gen_batches_from_single_file_1(files, args.nb_batch)
        # gen = gen_batches_from_single_file_2(files, args.nb_batch)
        for j in batches:
            x = gen.next()
            print x[0].shape, x[1].shape, x[2].shape
            exit()
        end = time.time()
        duration.append(end-start)
        print i, "\tElapsed time:\t%.2f seconds\tor rather\t%.2f minutes" % (((end-start)),((end-start)/60.))

    duration = np.asarray(duration)

    print "\n Mean \tElapsed time:\t%.2f seconds\tor rather\t%.2f minutes" % (np.mean(duration), (np.mean(duration) / 60.))



# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
