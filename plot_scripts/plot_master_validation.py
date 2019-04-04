import numpy as np
import h5py
import sys

from sys import path
path.append('/home/hpc/capm/sn0515/UVWireRecon')
from utilities.input_utilities import *
from utilities.generator import *
from utilities.cnn_utilities import *
from plot_scripts.plot_input_plots import *
from plot_scripts.plot_traininghistory import *
from plot_scripts.plot_validation import *
from scipy.stats import pearsonr
import matplotlib.colors as colors

def main():


    # =================
    # sources = 'Th228'
    # position = 'S5'
    # folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    # folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position + '_noCorrection/'
    #
    # print folderOUT
    #
    # plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=False)
    # =================



    sources = 'Th228'
    position = 'S5'
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position +'/'

    print folderOUT

    # plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=True)


    sources = 'Th228'
    position = 'S5'
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position +'_real_data/'

    # plotting_real_data(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=True)



    sources = 'bb0n'
    position = 'Uni'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position + '/'
    print folderOUT

    plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=False)



def plotting_real_data(folderIN, folderOUT, sources, position, purity_correction):
    data_U = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190404-1040-15/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_V = read_hdf5_file_to_dict(folderIN + '190206-1314-19/190207-0000-00/190208-0000-00/190209-0000-00/190404-1044-43/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_Z = read_hdf5_file_to_dict(folderIN + '190206-1314-24/190207-0000-00/190208-0000-00/190209-0000-00/190404-1125-55/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_E = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190404-1028-18/0physics-data/060-Th228-S5/events_060_Th228-S5.p')



    data_U['Y_PRED'][:, 0] = denormalize(data_U['Y_PRED'][:, 0], 'U')
    data_V['Y_PRED'][:, 0] = denormalize(data_V['Y_PRED'][:, 0], 'V')
    data_Z['Y_PRED'][:, 0] = denormalize(data_Z['Y_PRED'][:, 0], 'Z')
    data_E['Y_PRED'][:, 0] = denormalize(data_E['Y_PRED'][:, 0], 'energy')



    if purity_correction:
        # =======================
        #  Purity Correction
        # =======================
        print
        print sources, '\t ==> \t purity correction'
        drift_velocity = 0.00171  # drift velocity
        CV = 0.00174  # collection velocity
        collectionTime = 2940.0  # collection time
        lifetime = 4500000.0  # lifetime
        cathode_apdface_distance = 204.4065      # a
        apdplane_uplane_distance = 6.0     # b
        uplane_vplane_distance = 6.0       # c


        index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
        # print index_E
        for key in data_E.keys():
            data_E[key] = data_E[key][index_E]


        index_Z = np.lexsort((data_Z['MCEventNumber'], data_Z['MCRunNumber']))
        for key in data_Z.keys():
            data_Z[key] = data_Z[key][index_Z]


        # for i in range(160000):
        #     if data_E['MCPosU'][i] != data_Z['MCPosU'][i]:
        #         print data_E['MCEventNumber'][i], '\t\t', data_Z['MCEventNumber'][i]
        #         print data_E['MCPosU'][i], '\t', data_Z['MCPosU'][i]
        #         print data_E['MCRunNumber'][i], '\t\t', data_Z['MCRunNumber'][i]
        #         print

        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z['Y_PRED'])) / drift_velocity + collectionTime
        # drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - data_Z['Y_PRED']) / drift_velocity + collectionTime

        # corrected_energy = data_E['Y_PRED'] * data_E['CCPurityCorrectedEnergy'] / data_E['CCCorrectedEnergy']
        # fPurityCorrectedEnergy = fCorrectedEnergy * (energy_scale) * exp((fDriftTime) / lifetime)


        data_E['Y_PRED'] = data_E['Y_PRED'] * np.exp(drifttime / lifetime)
        # corrected_energy = data_E['Y_PRED'] * np.exp(drifttime / lifetime)

    else:
        print sources, '\t--> \t NO purity correction'



    # =======================
    #  Masking
    # =======================
    if sources == 'Th228':
        peak = 2614
    if sources == 'bb0n':
        peak = 2458



    mask_U_SS = data_U['CCIsFiducial'] == 1
    mask_U_SS = np.logical_and(mask_U_SS, np.sum(data_U['CCIs3DCluster'], axis=1) == data_U['CCNumberClusters'])
    mask_U_SS = np.logical_and(mask_U_SS, data_U['CCIsSS'] == 1)
    # mask_U_SS_peak = np.logical_and(mask_U_SS, data_U['MCEnergy'] > peak-10)
    # mask_U_SS_peak = np.logical_and(mask_U_SS_peak, data_U['MCEnergy'] < peak+10)
    # mask_num_U_is_1 = data_U['CCNumberUWires'] == 1
    # mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, mask_U_SS_peak)
    # mask_num_U_is_not_1 = data_U['CCNumberUWires'] > 1
    # mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, mask_U_SS_peak)


    mask_V_SS = data_V['CCIsFiducial'] == 1
    mask_V_SS = np.logical_and(mask_V_SS, np.sum(data_V['CCIs3DCluster'], axis=1) == data_V['CCNumberClusters'])
    mask_V_SS = np.logical_and(mask_V_SS, data_V['CCIsSS'] == 1)
    # mask_V_SS_peak = np.logical_and(mask_V_SS, data_V['MCEnergy'] > peak-10)
    # mask_V_SS_peak = np.logical_and(mask_V_SS_peak, data_V['MCEnergy'] < peak+10)

    mask_Z_SS = data_Z['CCIsFiducial'] == 1
    mask_Z_SS = np.logical_and(mask_Z_SS, np.sum(data_Z['CCIs3DCluster'], axis=1) == data_Z['CCNumberClusters'])
    mask_Z_SS = np.logical_and(mask_Z_SS, data_Z['CCIsSS'] == 1)
    # mask_Z_SS_peak = np.logical_and(mask_Z_SS, data_Z['MCEnergy'] > peak-10)
    # mask_Z_SS_peak = np.logical_and(mask_Z_SS_peak, data_Z['MCEnergy'] < peak+10)

    mask_E_SS = data_E['CCIsFiducial'] == 1
    mask_E_SS = np.logical_and(mask_E_SS, np.sum(data_E['CCIs3DCluster'], axis=1) == data_E['CCNumberClusters'])
    mask_E_SS = np.logical_and(mask_E_SS, data_E['CCIsSS'] == 1)
    # mask_E_SS_peak = np.logical_and(mask_E_SS, data_E['MCEnergy'] > peak-10)
    # mask_E_SS_peak = np.logical_and(mask_E_SS_peak, data_E['MCEnergy'] < peak+10)




    print '\nDataset properties:'
    print 'SS+MS: \t\t\t', data_U['Y_TRUE'].shape[0]
    print 'SS: \t\t\t', np.count_nonzero(mask_U_SS)
    # print 'SS peak: \t\t', np.count_nonzero(mask_U_SS_peak)
    # print 'SS peak onewire:\t', np.count_nonzero(mask_num_U_is_1)
    # print 'SS peak multiwire:\t', np.count_nonzero(mask_num_U_is_not_1)



    print '\n Start plotting'


    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    name_True = 'True'

    # =======================
    #  U Plots
    # =======================

    plot_diagonal(x=data_U['CCPosU'][:, 0][mask_U_SS], y=data_U['Y_PRED'][:, 0][mask_U_SS], xlabel=name_True, ylabel=name_DNN, mode='U',
                  fOUT=(folderOUT + sources + '_' + position + '_U_scatter' + '.pdf'))

    # plot_diagonal(x=data_U['Y_TRUE'][:, 0][mask_U_SS], y=data_U['CCPosU'][:, 0][mask_U_SS], xlabel=name_True,
    #               ylabel=name_EXO, mode='U',
    #               fOUT=(folderOUT + sources + '_' + position + '_U_scatter_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_U['Y_PRED'][:, 0][mask_U_SS], dEXO=data_U['CCPosU'][:, 0][mask_U_SS], dTrue=None,
                  mode='U', fOUT=(folderOUT + sources + '_' + position + '_U_spectrum' + '.pdf'))

    # =======================
    #  V Plots
    # =======================
    plot_diagonal(x=data_V['CCPosV'][:, 0][mask_V_SS], y=data_V['Y_PRED'][:, 0][mask_V_SS], xlabel=name_True, ylabel=name_DNN, mode='V',
                  fOUT=(folderOUT + sources + '_' + position + '_V_scatter' + '.pdf'))

    # plot_diagonal(x=data_V['Y_TRUE'][:, 0][mask_V_SS], y=data_V['CCPosV'][:, 0][mask_V_SS], xlabel=name_True,
    #           ylabel=name_EXO, mode='V',
    #           fOUT=(folderOUT + sources + '_' + position + '_V_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_V['Y_PRED'][:, 0][mask_V_SS], dEXO=data_V['CCPosV'][:, 0][mask_V_SS], dTrue=None,
                  mode='V', fOUT=(folderOUT + sources + '_' + position + '_V_spectrum' + '.pdf'))

    # =======================
    #  Z Plots
    # =======================


    plot_diagonal(x=data_Z['CCPosZ'][:, 0][mask_Z_SS], y=data_Z['Y_PRED'][:, 0][mask_Z_SS], xlabel=name_True, ylabel=name_DNN, mode='Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_scatter' + '.pdf'))

    # plot_diagonal(x=data_Z['Y_TRUE'][:, 0][mask_Z_SS], y=data_Z['CCPosZ'][:, 0][mask_Z_SS], xlabel=name_True,
    #               ylabel=name_EXO, mode='Z',
    #               fOUT=(folderOUT + sources + '_' + position + '_Z_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                  dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS],
                  dTrue=None,
                  mode='Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_spectrum' + '.pdf'))



    # =======================
    #  E Plots
    # =======================

    plot_diagonal(x=data_E['Y_PRED'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_DNN,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_Energy_scatter' + '.pdf'))

    plot_spectrum(dCNN=data_E['Y_PRED'][:, 0][mask_E_SS], dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
              dTrue=None,
              mode='Energy',
              fOUT=(folderOUT + sources + '_' + position + '_Energy_spectrum' + '.pdf'))

    # =======================
    #  Additional Plots
    # =======================



    print 'Finished plotting'



def plotting(folderIN, folderOUT, sources, position, purity_correction):

    print '\nReading files:'

    if sources == 'Th228':
        data_U = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190314-1613-46/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_U_zeroed = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190314-1623-59/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_V = read_hdf5_file_to_dict(folderIN + '190206-1314-19/190207-0000-00/190208-0000-00/190209-0000-00/190315-1019-52/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_Z = read_hdf5_file_to_dict(folderIN + '190206-1314-24/190207-0000-00/190208-0000-00/190209-0000-00/190315-1018-03/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_E = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190315-1021-10/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        # data_E_Uonly =
        # data_E_Vonly =

    if sources == 'bb0n':
        data_U = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190330-1017-31/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_U_zeroed = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190330-1119-32/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_V = read_hdf5_file_to_dict(folderIN + '190206-1314-19/190207-0000-00/190208-0000-00/190209-0000-00/190330-1018-24/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_Z = read_hdf5_file_to_dict(folderIN + '190206-1314-24/190207-0000-00/190208-0000-00/190209-0000-00/190330-1019-07/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
        data_E = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190330-1019-52/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')





    data_U['Y_TRUE'][:, 0] = denormalize(data_U['Y_TRUE'][:, 0], 'U')
    data_U['Y_PRED'][:, 0] = denormalize(data_U['Y_PRED'][:, 0], 'U')
    data_U_zeroed['Y_TRUE'][:, 0] = denormalize(data_U_zeroed['Y_TRUE'][:, 0], 'U')
    data_U_zeroed['Y_PRED'][:, 0] = denormalize(data_U_zeroed['Y_PRED'][:, 0], 'U')

    data_V['Y_TRUE'][:, 0] = denormalize(data_V['Y_TRUE'][:, 0], 'V')
    data_V['Y_PRED'][:, 0] = denormalize(data_V['Y_PRED'][:, 0], 'V')

    data_Z['Y_TRUE'][:, 0] = denormalize(data_Z['Y_TRUE'][:, 0], 'Z')
    data_Z['Y_PRED'][:, 0] = denormalize(data_Z['Y_PRED'][:, 0], 'Z')

    data_E['Y_TRUE'][:, 0] = denormalize(data_E['Y_TRUE'][:, 0], 'energy')
    data_E['Y_PRED'][:, 0] = denormalize(data_E['Y_PRED'][:, 0], 'energy')



    if purity_correction:
        # =======================
        #  Purity Correction
        # =======================
        print
        print sources, '\t ==> \t purity correction'
        drift_velocity = 0.00171  # drift velocity
        CV = 0.00174  # collection velocity
        collectionTime = 2940.0  # collection time
        lifetime = 4500000.0  # lifetime
        cathode_apdface_distance = 204.4065      # a
        apdplane_uplane_distance = 6.0     # b
        uplane_vplane_distance = 6.0       # c


        index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
        # print index_E
        for key in data_E.keys():
            data_E[key] = data_E[key][index_E]


        index_Z = np.lexsort((data_Z['MCEventNumber'], data_Z['MCRunNumber']))
        for key in data_Z.keys():
            data_Z[key] = data_Z[key][index_Z]


        for i in range(160000):
            if data_E['MCPosU'][i] != data_Z['MCPosU'][i]:
                print data_E['MCEventNumber'][i], '\t\t', data_Z['MCEventNumber'][i]
                print data_E['MCPosU'][i], '\t', data_Z['MCPosU'][i]
                print data_E['MCRunNumber'][i], '\t\t', data_Z['MCRunNumber'][i]
                print

        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z['Y_PRED'])) / drift_velocity + collectionTime
        # drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - data_Z['Y_PRED']) / drift_velocity + collectionTime

        # corrected_energy = data_E['Y_PRED'] * data_E['CCPurityCorrectedEnergy'] / data_E['CCCorrectedEnergy']
        # fPurityCorrectedEnergy = fCorrectedEnergy * (energy_scale) * exp((fDriftTime) / lifetime)


        data_E['Y_PRED'] = data_E['Y_PRED'] * np.exp(drifttime / lifetime)
        # corrected_energy = data_E['Y_PRED'] * np.exp(drifttime / lifetime)

    else:
        print sources, '\t--> \t NO purity correction'



    # =======================
    #  Masking
    # =======================
    if sources == 'Th228':
        peak = 2614
    if sources == 'bb0n':
        peak = 2458

    mask_U_SS = np.sum(data_U['CCIsFiducial'], axis=1) == data_U['CCNumberClusters']
    mask_U_SS = np.logical_and(mask_U_SS, np.sum(data_U['CCIs3DCluster'], axis=1) == data_U['CCNumberClusters'])
    mask_U_SS = np.logical_and(mask_U_SS, data_U['CCIsSS'] == 1)
    mask_U_SS_peak = np.logical_and(mask_U_SS, data_U['MCEnergy'] > peak-10)
    mask_U_SS_peak = np.logical_and(mask_U_SS_peak, data_U['MCEnergy'] < peak+10)
    mask_num_U_is_1 = data_U['CCNumberUWires'] == 1
    mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, mask_U_SS_peak)
    mask_num_U_is_not_1 = data_U['CCNumberUWires'] > 1
    mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, mask_U_SS_peak)

    mask_U_SS_zeroed = np.sum(data_U_zeroed['CCIsFiducial'], axis=1) == data_U_zeroed['CCNumberClusters']
    mask_U_SS_zeroed = np.logical_and(mask_U_SS_zeroed, np.sum(data_U_zeroed['CCIs3DCluster'], axis=1) == data_U_zeroed['CCNumberClusters'])
    mask_U_SS_zeroed = np.logical_and(mask_U_SS_zeroed, data_U_zeroed['CCIsSS'] == 1)
    mask_U_SS_peak_zeroed = np.logical_and(mask_U_SS_zeroed, data_U_zeroed['MCEnergy'] > peak-10)
    mask_U_SS_peak_zeroed = np.logical_and(mask_U_SS_peak_zeroed, data_U_zeroed['MCEnergy'] < peak+10)
    mask_num_U_is_1_zeroed = data_U_zeroed['CCNumberUWires'] == 1
    mask_num_U_is_1_zeroed = np.logical_and(mask_num_U_is_1_zeroed, mask_U_SS_peak_zeroed)
    mask_num_U_is_not_1_zeroed = data_U_zeroed['CCNumberUWires'] > 1
    mask_num_U_is_not_1_zeroed = np.logical_and(mask_num_U_is_not_1_zeroed, mask_U_SS_peak_zeroed)

    mask_V_SS = np.sum(data_V['CCIsFiducial'], axis=1) == data_V['CCNumberClusters']
    mask_V_SS = np.logical_and(mask_V_SS, np.sum(data_V['CCIs3DCluster'], axis=1) == data_V['CCNumberClusters'])
    mask_V_SS = np.logical_and(mask_V_SS, data_V['CCIsSS'] == 1)
    mask_V_SS_peak = np.logical_and(mask_V_SS, data_V['MCEnergy'] > peak-10)
    mask_V_SS_peak = np.logical_and(mask_V_SS_peak, data_V['MCEnergy'] < peak+10)

    mask_Z_SS = np.sum(data_Z['CCIsFiducial'], axis=1) == data_Z['CCNumberClusters']
    mask_Z_SS = np.logical_and(mask_Z_SS, np.sum(data_Z['CCIs3DCluster'], axis=1) == data_Z['CCNumberClusters'])
    mask_Z_SS = np.logical_and(mask_Z_SS, data_Z['CCIsSS'] == 1)
    mask_Z_SS_peak = np.logical_and(mask_Z_SS, data_Z['MCEnergy'] > peak-10)
    mask_Z_SS_peak = np.logical_and(mask_Z_SS_peak, data_Z['MCEnergy'] < peak+10)

    mask_E_SS = np.sum(data_E['CCIsFiducial'], axis=1) == data_E['CCNumberClusters']
    mask_E_SS = np.logical_and(mask_E_SS, np.sum(data_E['CCIs3DCluster'], axis=1) == data_E['CCNumberClusters'])
    mask_E_SS = np.logical_and(mask_E_SS, data_E['CCIsSS'] == 1)
    mask_E_SS_peak = np.logical_and(mask_E_SS, data_E['MCEnergy'] > peak-10)
    mask_E_SS_peak = np.logical_and(mask_E_SS_peak, data_E['MCEnergy'] < peak+10)




    print '\nDataset properties:'
    print 'SS+MS: \t\t\t', data_U['Y_TRUE'].shape[0]
    print 'SS: \t\t\t', np.count_nonzero(mask_U_SS)
    print 'SS peak: \t\t', np.count_nonzero(mask_U_SS_peak)
    print 'SS peak onewire:\t', np.count_nonzero(mask_num_U_is_1)
    print 'SS peak multiwire:\t', np.count_nonzero(mask_num_U_is_not_1)



    print '\n Start plotting'


    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    name_True = 'True'

    # =======================
    #  U Plots
    # =======================

    plot_diagonal(x=data_U['Y_TRUE'][:, 0][mask_U_SS], y=data_U['Y_PRED'][:, 0][mask_U_SS], xlabel=name_True, ylabel=name_DNN, mode='U',
                  fOUT=(folderOUT + sources + '_' + position + '_U_scatter_DNN' + '.pdf'))

    plot_diagonal(x=data_U['Y_TRUE'][:, 0][mask_U_SS], y=data_U['CCPosU'][:, 0][mask_U_SS], xlabel=name_True,
                  ylabel=name_EXO, mode='U',
                  fOUT=(folderOUT + sources + '_' + position + '_U_scatter_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_U['Y_PRED'][:, 0][mask_U_SS], dEXO=data_U['CCPosU'][:, 0][mask_U_SS], dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS],
                  mode='U', fOUT=(folderOUT + sources + '_' + position + '_U_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                        dDNN=data_U['Y_PRED'][:, 0][mask_num_U_is_1],
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_1], limit=10,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_onewire' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_num_U_is_not_1],
                        dDNN=data_U['Y_PRED'][:, 0][mask_num_U_is_not_1],
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_not_1], limit=10,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_multiwire' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS],
                        dDNN=data_U['Y_PRED'][:, 0][mask_U_SS],
                        dEXO=data_U['CCPosU'][:, 0][mask_U_SS], limit=10,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                        dDNN=data_U['Y_PRED'][:, 0][mask_num_U_is_1],
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_1], limit=100,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_onewire_100' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_num_U_is_not_1],
                        dDNN=data_U['Y_PRED'][:, 0][mask_num_U_is_not_1],
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_not_1], limit=100,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_multiwire_100' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS],
                        dDNN=data_U['Y_PRED'][:, 0][mask_U_SS],
                        dEXO=data_U['CCPosU'][:, 0][mask_U_SS], limit=100,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_100' + '.pdf')

    plot_residual_correlation(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS],
                              dDNN=data_U['Y_PRED'][:, 0][mask_U_SS],
                              dEXO=data_U['CCPosU'][:, 0][mask_U_SS], limit=6,
                              mode='U',
                              fOUT=folderOUT + sources + '_' + position + '_U_SS_correlation' + '.pdf')


    plot_boxplot(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS_peak],
                 dTrue_masked=data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                 dEXO=data_U['CCPosU'][:, 0][mask_U_SS_peak],
                 dEXO_masked=data_U['CCPosU'][:, 0][mask_num_U_is_1],
                 dDNN=data_U['Y_PRED'][:, 0][mask_U_SS_peak],
                 dDNN_masked=data_U['Y_PRED'][:, 0][mask_num_U_is_1],
                 dTrue_masked_zeroed=data_U_zeroed['Y_TRUE'][:, 0][mask_num_U_is_1_zeroed],
                 dEXO_masked_zeroed=data_U_zeroed['CCPosU'][:, 0][mask_num_U_is_1_zeroed],
                 dDNN_masked_zeroed=data_U_zeroed['Y_PRED'][:, 0][mask_num_U_is_1_zeroed],
                 title='Boxplot', name_DNN_masked='DNN onewire', name_EXO_masked='EXO onewire',
                 fOUT=folderOUT + sources + '_' + position + '_U_boxplot' + '.pdf')





    #  Plot of residual as funciton of max of neighbouring channel:
    # mask_channel_1 = data['EVENT_INFO']['CCNumberUWires'] == 1
    # mask_channel_1 = np.logical_and(mask_channel_1, data['EVENT_INFO']['CCIsSS'] == 1)
    # mask_channel_1 = np.logical_and(mask_channel_1, data['Y_TRUE'][:, a] - data['Y_PRED'][:, a] < limit)
    # mask_channel_1 = np.logical_and(mask_channel_1, data['Y_TRUE'][:, a] - data['EVENT_INFO']['CCPosU'][:, a] < limit)
    # mask_channel_1 = np.logical_and(mask_channel_1, mask_SS_peak)
    #
    # plot_induction_scatter(...)


    # =======================
    #  V Plots
    # =======================

    plot_diagonal(x=data_V['Y_TRUE'][:, 0][mask_V_SS], y=data_V['Y_PRED'][:, 0][mask_V_SS], xlabel=name_True, ylabel=name_DNN, mode='V',
                  fOUT=(folderOUT + sources + '_' + position + '_V_DNN' + '.pdf'))

    plot_diagonal(x=data_V['Y_TRUE'][:, 0][mask_V_SS], y=data_V['CCPosV'][:, 0][mask_V_SS], xlabel=name_True,
              ylabel=name_EXO, mode='V',
              fOUT=(folderOUT + sources + '_' + position + '_V_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_V['Y_PRED'][:, 0][mask_V_SS], dEXO=data_V['CCPosV'][:, 0][mask_V_SS], dTrue=data_V['Y_TRUE'][:, 0][mask_V_SS],
                  mode='V', fOUT=(folderOUT + sources + '_' + position + '_V_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=data_V['Y_TRUE'][:, 0][mask_V_SS], dDNN=data_V['Y_PRED'][:, 0][mask_V_SS],
                        dEXO=data_V['CCPosV'][:, 0][mask_V_SS],
                        title='V', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=5,
                        fOUT=folderOUT + sources + '_' + position + '_V_histo' + '.pdf')

    plot_residual_correlation(dTrue=data_V['Y_TRUE'][:, 0][mask_V_SS],
                              dDNN=data_V['Y_PRED'][:, 0][mask_V_SS],
                              dEXO=data_V['CCPosV'][:, 0][mask_V_SS],
                              limit=6, mode='V',
                              fOUT=folderOUT + sources + '_' + position + '_V_SS_correlation' + '.pdf')


    # =======================
    #  Z Plots
    # =======================


    plot_diagonal(x=data_Z['Y_TRUE'][:, 0][mask_Z_SS], y=data_Z['Y_PRED'][:, 0][mask_Z_SS], xlabel=name_True, ylabel=name_DNN, mode='Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_DNN' + '.pdf'))

    plot_diagonal(x=data_Z['Y_TRUE'][:, 0][mask_Z_SS], y=data_Z['CCPosZ'][:, 0][mask_Z_SS], xlabel=name_True,
                  ylabel=name_EXO, mode='Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                  dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS],
                  dTrue=data_Z['Y_TRUE'][:, 0][mask_Z_SS], mode='Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=data_Z['Y_TRUE'][:, 0][mask_Z_SS], dDNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                        dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS], limit=5,
                        # dEXO=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]), limit=5,
                        title='Z', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_Z_histo' + '.pdf')

    plot_residual_correlation(dTrue=data_Z['Y_TRUE'][:, 0][mask_Z_SS],
                              dDNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                              dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS], limit=6,
                              mode='Z',
                              fOUT=folderOUT + sources + '_' + position + '_Z_SS_correlation' + '.pdf')

    # =======================
    #  E Plots
    # =======================

    plot_diagonal(x=data_E['Y_TRUE'][:, 0][mask_E_SS], y=data_E['Y_PRED'][:, 0][mask_E_SS],
                  xlabel=name_True, ylabel=name_DNN, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_Energy_DNN' + '.pdf'))

    plot_diagonal(x=data_E['Y_TRUE'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_True,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_Energy_EXO' + '.pdf'))

    plot_diagonal(x=data_E['Y_PRED'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_DNN,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_Energy_Both' + '.pdf'))

    plot_spectrum(dCNN=data_E['Y_PRED'][:, 0][mask_E_SS], dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
              dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS],
              mode='Energy',
              fOUT=(folderOUT + sources + '_' + position + '_Energy_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS], dDNN=data_E['Y_PRED'][:, 0][mask_E_SS],
                        dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
                        title='Energy', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=150,
                        fOUT=folderOUT + sources + '_' + position + '_Energy_histo' + '.pdf')

    plot_residual_histo(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS_peak], dDNN=data_E['Y_PRED'][:, 0][mask_E_SS_peak],
                        dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS_peak],
                        title='Energy', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=150,
                        fOUT=folderOUT + sources + '_' + position + '_Energy_histo_peak' + '.pdf')

    plot_hexbin(data_E['MCPosX'][:][mask_E_SS], data_E['MCPosY'][:][mask_E_SS],
                data_E['Y_TRUE'][:, 0][mask_E_SS] - data_E['Y_PRED'][:, 0][mask_E_SS],
                data_E['Y_TRUE'][:, 0][mask_E_SS] - data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
                'X [mm]', 'Y[mm]',
                fOUT=folderOUT + sources + '_' + position + '_Energy_DNN_hexbin' + '.pdf')


    plot_residual_correlation(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS],
                              dDNN=data_E['Y_PRED'][:, 0][mask_E_SS],
                              dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], limit=200,
                              mode='energy',
                              fOUT=folderOUT + sources + '_' + position + '_E_SS_correlation' + '.pdf')


    # plot_hexbin(data['EVENT_INFO']['MCPosX'][:, 0], data['EVENT_INFO']['MCPosY'][:, 0], np.absolute(data['Y_TRUE'][:, 0] - data['EVENT_INFO']['CCCorrectedEnergy'][:, 0]), 'X [mm]', 'Y[mm]',
    #             fOUT=folderOUT + dir_residual + sources + '_' + position + '_Energy_EXO_hexbin_' + epoch + '.pdf')


    # =======================
    #  Additional Plots
    # =======================



    print 'Finished plotting'



    return



def plot_induction_scatter(limit=10):
    from matplotlib.ticker import NullFormatter

    channel_1 = data['EVENT_INFO']['wf_max_neighbour'][:][mask_channel_1], \
                data['Y_TRUE'][:, a][mask_channel_1] - data['Y_PRED'][:, a][mask_channel_1], \
                data['Y_TRUE'][:, a][mask_channel_1] - data['EVENT_INFO']['CCPosU'][:, a][mask_channel_1]


    channel_1_sorted = zip(*sorted(zip(channel_1[0], channel_1[1], channel_1[2])))

    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.8
    bottom, height = 0.1, 0.8
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.5, height]
    rect_legend = [left_h, bottom_h, 0.5, 0.2]

    # start with a rectangular Figure
    plt.figure(1, figsize=(10, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axLegend = plt.axes(rect_legend)

    axScatter.set_xlabel('max value neighbouring channel')
    axScatter.set_ylabel(r'Residual $r = U_{pred} - U_{true}$')
    axHistx.set_ylabel('mean $\mu(|r|)$ ')
    axHisty.set_xlabel('Counts')

    axLegend.text(0.1, 0.6, 'DNN: \n$\mu=%.1f,$ $\sigma=%.1f$' % (np.mean(channel_1[1]), np.std(channel_1[1])),
                  color='navy', alpha=0.5)
    axLegend.text(0.1, 0.2, 'EXO: \n$\mu=%.1f,$ $\sigma=%.1f$' % (np.mean(channel_1[2]), np.std(channel_1[2])),
                  color='red', alpha=0.5)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    axLegend.xaxis.set_major_formatter(nullfmt)
    axLegend.yaxis.set_major_formatter(nullfmt)
    plt.xticks([])
    plt.yticks([])

    # now determine nice limits by hand:
    binwidth = 0.15
    xymax = np.max([np.max(np.fabs(channel_1[0])), np.max(np.fabs(channel_1[1]))])
    lim = (int(xymax / binwidth) + 1) * binwidth
    # lim = 4

    bins_hex = int(lim * 2. / binwidth)
    # the scatter plot:
    # axScatter.hexbin(x, y, cmap=color, gridsize=bins_hex)
    axScatter.scatter(channel_1[0], channel_1[1], alpha=0.2, marker='.', color='navy')
    axScatter.scatter(channel_1[0], channel_1[2], alpha=0.2, color='red', marker='.')

    # H, xedges, yedges = np.histogram2d(x, y, bins=1000)
    # axScatter.imshow(H, cmap='PuBu')
    # axScatter.contour(x, y, cmap='PuBu')

    # now determine nice limits by hand:

    # xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    # lim = (int(xymax / binwidth) + 1) * binwidth
    # lim = 4

    axScatter.set_xlim((0, lim))
    axScatter.set_ylim((-limit, limit))

    bins = np.arange(-lim, lim + binwidth, binwidth)

    # n_x, bins_x, patches_x = axHistx.hist(x, bins=bins)
    axHisty.hist(channel_1[1], bins=bins, orientation='horizontal', alpha=0.5, color='navy')
    axHisty.hist(channel_1[2], bins=bins, orientation='horizontal', alpha=0.5, color='red')

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    # col_x = n_x / max(n_x)
    # bin_centers_x = 0.5 * (bins_x[:-1] + bins_x[1:])
    # for c, p in zip(col_x, patches_x):
    #     plt.setp(p, 'facecolor', cm(c))
    #     plt.setp(p, 'edgecolor', cm(c), alpha=1)
    # axHistx.plot(bin_centers_x, n_x, color=cm(max(n_x)), )
    axHistx.plot(running_mean(channel_1_sorted[0], 100), running_mean(np.absolute(channel_1_sorted[1]), 100), alpha=0.5,
                 color='navy')
    axHistx.plot(running_mean(channel_1_sorted[0], 100), running_mean(np.absolute(channel_1_sorted[2]), 100),
                 color='red', alpha=0.5)

    # col_y = n_y / max(n_y)
    # bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
    # for c, p in zip(col_y, patches_y):
    #     plt.setp(p, 'facecolor', cm(c))
    #     plt.setp(p, 'edgecolor', cm(c), alpha=1)
    # axHisty.plot(n_y, bin_centers_y, color=cm(max(n_y)))

    plt.savefig(folderOUT + sources + '_' + position + '_U_wirecheck_' + epoch + '.pdf',
                bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def read_hdf5_file_to_dict(file, keys_to_read=['all']):
    """
    Write dict to hdf5 file
    :param string file: Full filepath of the output hdf5 file, e.g. '[/path/to/file/file.hdf5]'.
    :param list keys_to_write: Keys that will be written to file
    :return dict data: dict containing data.
    """
    data = {}
    fIN = h5py.File(file, "r")
    if 'all' in keys_to_read:
        keys_to_read = fIN.keys()
    for key in keys_to_read:
        if key not in fIN.keys():
            print keys_to_read, '\n', fIN.keys()
            raise ValueError('%s not in file!' % (str(key)))
        data[key] = np.asarray(fIN.get(key))
    fIN.close()
    print '   ', file
    return data



if __name__ == '__main__':
    main()
