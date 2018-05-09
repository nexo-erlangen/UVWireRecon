#!/usr/bin/env python

# ----------------------------------------------------------
# Program Functions
# ----------------------------------------------------------
def make_organize():
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

    parser.add_argument('-out', dest='folderOUT', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderOUT Path')
    parser.add_argument('-in', dest='folderIN', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/Data_MC', help='folderIN Path')
    parser.add_argument('-model', dest='folderMODEL', default='/home/vault/capm/sn0515/PhD/Th_U-Wire/MonteCarlo/Dummy', help='folderMODEL Path')
    parser.add_argument('-gpu', type=int, dest='nb_GPU', default=1, choices=[1, 2, 3, 4], help='nb of GPU')
    parser.add_argument('-epoch', type=int, dest='nb_epoch', default=1, help='nb Epochs')
    parser.add_argument('-batch', type=int, dest='nb_batch', default=16, help='Batch Size')
    # parser.add_argument('-multi', dest='multiplicity', default='SS', help='Choose Event Multiplicity (SS / SS+MS)')
    parser.add_argument('-weights', dest='nb_weights', default='final', help='Load weights from Epoch')
    # parser.add_argument('-position', dest='position', default=['S5'], choices=['S2', 'S5', 'S8'], help='sources position')
    parser.add_argument('--resume', dest='resume', action='store_true', help='Resume Training')
    parser.add_argument('--test', dest='test', action='store_true', help='Only reduced data')
    args, unknown = parser.parse_known_args()

    folderIN, files = {}, {}
    args.sources = ["thss", "thms", "coss", "coms", "rass", "rams", "gass", "gams", "unss", 'unms']
    # args.sources = ["thss", "thms", "coss", "rass", "gass", "unss", 'unms']
    args.label = {
        'thss': "Th228-SS",
        'rass': "Ra226-SS",
        'rams': "Ra226-SS+MS",
        'coss': "Co60-SS",
        'coms': "Co60-SS+MS",
        'gass': "Gamma-SS",
        'gams': "Gamma-SS+MS",
        'unss': "Uniform-SS",
        'thms': "Th228-SS+MS",
        'unms': "Uniform-SS+MS"}
    endings = {
        'thss': "Th228_Wfs_SS_S5_MC/",
        'thms': "Th228_Wfs_SS+MS_S5_MC/",
        'rass': "Ra226_Wfs_SS_S5_MC/",
        'rams': "Ra226_Wfs_SS+MS_S5_MC/",
        'coss': "Co60_Wfs_SS_S5_MC/",
        'coms': "Co60_Wfs_SS+MS_S5_MC/",
        'gass': "Gamma_Wfs_SS_S5_MC/",
        'gams': "Gamma_Wfs_SS+MS_S5_MC/",
        'unss': "Uniform_Wfs_SS_S5_MC/",
        'unms': "Uniform_Wfs_SS+MS_S5_MC/"}

    for source in args.sources:
        folderIN[source] = os.path.join(args.folderIN, endings[source])
        files[source] = [os.path.join(folderIN[source], f) for f in os.listdir(folderIN[source]) if os.path.isfile(os.path.join(folderIN[source], f))]
        print 'Input  Folder: (', source, ')\t', folderIN[source]
    args.folderOUT = os.path.join(args.folderOUT,'')
    args.folderMODEL = os.path.join(args.folderMODEL,'')
    args.folderIN = folderIN

    if args.nb_weights != 'final': args.nb_weights=str(args.nb_weights).zfill(3)
    if not os.path.exists(args.folderOUT+'models'):
        os.makedirs(args.folderOUT+'models')

    print 'Output Folder:\t\t'  , args.folderOUT
    if args.resume:
        print 'Model Folder:\t\t', args.folderMODEL
    print 'Number of GPU:\t\t', args.nb_GPU
    print 'Number of Epoch:\t', args.nb_epoch
    print 'BatchSize:\t\t', args.nb_batch, '\n'
    return args, files


    parser = argparse.ArgumentParser(description='E.g. < python run_cnn.py train_filepath test_filepath [...] > \n'
                                                 'Script that runs a CNN. \n'
                                                 'The input arguments are either single files for train- and testdata or \n'
                                                 'a .list file that contains the filepaths of the train-/testdata.',
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('train_file', metavar='train_file', type=str, nargs='?', help='the filepath of the traindata file.')
    parser.add_argument('test_file', metavar='test_file', type=str, nargs='?', help='the filepath of the testdata file.')
    parser.add_argument('-l', '--list', dest='listfile_train_and_test', type=str, nargs=2,
                        help='filepath of a .list file that contains all .h5 files that should be used for training/testing.')
    parser.add_argument('-m', '--multiple_files', dest='listfile_multiple', type=str, nargs=1,
                        help='filepath of a .list file that which contains multiple input files '
                             ' that should be used for training/testing in double/triple/... input models that need multiple input files per batch. \n'
                             'The structure of the .list file should be as follows: \n '
                             'f_1_train \n, f_1_test \n, f_2_train \n, f_2_test \n, ..')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    multiple_inputs = False

    if args.listfile_train_and_test:
        train_files, test_files = [], []

        for line in open(args.listfile_train_and_test[0]):
            line = line.rstrip('\n')
            train_files.append(([line], h5_get_number_of_rows(line)))

        for line in open(args.listfile_train_and_test[1]):
            line = line.rstrip('\n')
            test_files.append(([line], h5_get_number_of_rows(line)))

    elif args.listfile_multiple:
        multiple_inputs = True

        train_files, test_files = [], []
        train_files_temp, test_files_temp = [], []

        for i, line in enumerate(open(args.listfile_multiple[0])):
            line = line.rstrip('\n')
            if i % 2 == 0: # even, train_file
                train_files_temp.append(line)
            else: # odd, test_file
                test_files_temp.append(line)

        n_rows_train, n_rows_test = h5_get_number_of_rows(train_files_temp[0]), h5_get_number_of_rows(test_files_temp[0])
        train_files.append((train_files_temp, n_rows_train))
        test_files.append((test_files_temp, n_rows_test))

    else:
        train_files = [([args.train_file], h5_get_number_of_rows(args.train_file))]
        test_files = [([args.test_file], h5_get_number_of_rows(args.test_file))]

    if use_scratch_ssd is True:
        train_files, test_files = use_node_local_ssd_for_input(train_files, test_files, multiple_inputs=multiple_inputs)

    return train_files, test_files