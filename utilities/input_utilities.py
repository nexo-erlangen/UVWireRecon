#!/usr/bin/env python

import sys
import os
import stat

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def parseInput():
    """
        Parses the user input for running the CNN.
        There are three available input modes:
        1) Parse the train/test filepaths directly, if you only have a single file for training/testing
        2) Parse a .list file with arg -l that contains the paths to all train/test files, if the whole dataset is split over multiple files
        3) Parse a .list file with arg -m, if you need multiple input files for a single (!) batch during training.
           This is needed, if e.g. the images for a double input model are coming from different .h5 files.
           An example would be a double input model with two inputs: a loose timecut input (e.g. yzt-x) and a tight timecut input (also yzt-x).
           The important thing is that both files contain the same events, just with different cuts/projections!
           Another usecase would be a double input xyz-t + xyz-c model.
        The output (train_files, test_files) is structured as follows:
        1) train/test: [ ( [train/test_filepath]  , n_rows) ]. The outmost list has len 1 as well as the list in the tuple.
        2) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has arbitrary length (depends on number of files), but len 1 for the list in the tuple.
        3) train/test: [ ( [train/test_filepath]  , n_rows), ... ]. The outmost list has len 1, but the list inside the tuple has arbitrary length.
           A combination of 2) + 3) (multiple input files for each batch from 3) AND all events split over multiple files) is not yet supported.
        :param bool use_scratch_ssd: specifies if the input files should be copied to the node-local SSD scratch space.
        :return: list(([train_filepaths], train_filesize)) train_files: list of tuples that contains the list(trainfiles) and their number of rows.
        :return: list(([test_filepaths], test_filesize)) test_files: list of tuples that contains the list(testfiles) and their number of rows.
    """

    import argparse

    parser = argparse.ArgumentParser(description='E.g. < python run_cnn.py ..... > \n'
                                                 'Script that runs a DNN.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('-o', '--out', dest='folderOUT', type=str, default='Dummy', help='folderOUT Path')
    parser.add_argument('-i', '--in', dest='folderIN', type=str, default='/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/', help='folderIN Path')
    parser.add_argument('-r', '--runs', dest='folderRUNS', type=str, default='/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/', help='folderRUNS Path')
    parser.add_argument('-m', '--model', dest='folderMODEL', type=str, default='Dummy', help='folderMODEL Path')
    parser.add_argument('-t', '--targets', type=str, dest='var_targets', default='energy_and_position', choices=['energy_and_position'], help='Targets to train the network against')
    parser.add_argument('-a', '--arch', type=str, dest='cnn_arch', default='DCNN', choices=['DCNN', 'ResNet', 'Inception'], help='Choose network architecture')
    parser.add_argument('-g', '--gpu', type=int, dest='num_gpu', default=1, choices=[1, 2, 3, 4], help='Number of GPUs')
    parser.add_argument('-e', '--epoch', type=int, dest='num_epoch', default=1, help='nb Epochs')
    parser.add_argument('-b', '--batch', type=int, dest='batchsize', default=16, help='Batch Size')
    # parser.add_argument('-multi', dest='multiplicity', default='SS', help='Choose Event Multiplicity (SS / SS+MS)')
    parser.add_argument('-w', '--weights', dest='num_weights', default=0, help='Load weights from Epoch')
    # parser.add_argument('-position', dest='position', default=['S5'], choices=['S2', 'S5', 'S8'], help='sources position')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--test', dest='test', action='store_true', help='Only reduced data')
    args, unknown = parser.parse_known_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    folderIN, files = {}, {}
    args.sources = ["unss", 'unms']
    endings = {
        'unss': "UniformGamma_ExpWFs_MC_SS/",
        'unms': "UniformGamma_ExpWFs_MC_SS+MS/"}

    for source in args.sources:
        folderIN[source] = os.path.join(os.path.join(args.folderIN,''), endings[source])
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
        print 'Input  Folder: (', source, ')\t', folderIN[source]
    args.folderOUT = os.path.join(os.path.join(args.folderRUNS,args.folderOUT),'')
    args.folderMODEL = os.path.join(os.path.join(os.path.join(args.folderRUNS,''),args.folderMODEL),'')
    args.folderIN = folderIN

    adjustPermissions(args.folderOUT)

    if args.resume == True:
        if type(args.num_weights) == int: args.num_weights = str(args.num_weights).zfill(3)
    else:
        args.num_weights = 0
    if not os.path.exists(args.folderOUT+'models'): os.makedirs(args.folderOUT+'models')

    print 'Output Folder:\t\t'  , args.folderOUT
    if args.resume: print 'Model Folder:\t\t', args.folderMODEL
    print 'Number of GPU:\t\t', args.num_gpu
    print 'Number of Epoch:\t', args.num_epoch
    print 'BatchSize:\t\t', args.batchsize, '\n'

    return args, files

def splitFiles(args, files, frac_train, frac_val):
    import cPickle as pickle
    if args.resume:
        os.system("cp %s %s" % (args.folderMODEL + "splitted_files.p", args.folderOUT + "splitted_files.p"))
        print 'load splitted files from %s' % (args.folderMODEL + "splitted_files.p")
        return pickle.load(open(args.folderOUT + "splitted_files.p", "rb"))
    else:
        import random
        splitted_files= {'train': {}, 'val': {}, 'test': {}}
        print "Source\tTotal\tTrain\tValid\tTest"
        for source in args.sources:
            if (frac_train[source] + frac_val[source]) > 1.0 : raise ValueError('check file fractions!')
            num_train = int(round(len(files[source]) * frac_train[source]))
            num_val   = int(round(len(files[source]) * frac_val[source]))
            random.shuffle(files[source])
            if not args.test:
                splitted_files['train'][source] = files[source][0 : num_train]
                splitted_files['val'][source]   = files[source][num_train : num_train + num_val]
                splitted_files['test'][source]  = files[source][num_train + num_val : ]
            else:
                splitted_files['val'][source]   = files[source][0:1]
                splitted_files['test'][source]  = files[source][1:2]
                splitted_files['train'][source] = files[source][2:3]
            print "%s\t%i\t%i\t%i\t%i" % (source, len(files[source]), len(splitted_files['train'][source]), len(splitted_files['val'][source]), len(splitted_files['test'][source]))
        pickle.dump(splitted_files, open(args.folderOUT + "splitted_files.p", "wb"))
        return splitted_files

def adjustPermissions(path):
    # set this folder to read/writeable/exec
    try:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
    except OSError:
        # TODO could copy and replace non-changeable files and apply chmod on new files
        pass

    # step through all the files/folders and change permissions
    for file in os.listdir(path):
        filePath = os.path.join(path, file)

        # if it is a directory, doe recursive call
        if os.path.isdir(filePath):
            adjustPermissions(filePath)
        # for files merely call chmod
        else:
            try:
                # set this file to read/writeable/exec
                os.chmod(filePath, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH)
            except OSError:
                # TODO could copy and replace non-changeable files and apply chmod on new files
                pass