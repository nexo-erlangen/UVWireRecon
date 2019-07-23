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
    # plt.rcParams['figure.figsize'] = (9, 5)  # Make the figures a bit bigger
    plt.rc('font', size=16, family='serif')

    # =================
    sources = 'Th228'
    position = 'S5'
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position + '_noCorrection/'

    # plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=False)
    # =================



    sources = 'Th228'
    position = 'S5'
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position +'/'

    # plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=True)


    sources = 'Th228'
    position = 'S5'
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position +'_real_data/'

    # plotting_real_data(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=True)



    sources = 'bb0n'
    position = 'Uni'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position + '/'

    plotting(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=False)



    sources = 'bb0n'
    position = 'Uni'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/Validation/190206-1314/' + sources + '-' + position + '/'

    # plotting_wireplane_input(folderIN=folderIN, folderOUT=folderOUT, sources=sources, position=position, purity_correction=False)


def plotting_wireplane_input(folderIN, folderOUT, sources, position, purity_correction):

    print '\nReading files:'



    data_E = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190330-1019-52/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
    data_E_Uonly = read_hdf5_file_to_dict(folderIN + '190404-0924-29/190405-0000-00/190406-0000-00/190409-1646-58/1validation-data/' + sources + '-' + position + '/events_036_' + sources + '-' + position + '.p')
    data_E_Vonly = read_hdf5_file_to_dict(folderIN + '190404-0926-39/190405-0000-00/190406-0000-00/190409-1648-54/1validation-data/' + sources + '-' + position + '/events_036_' + sources + '-' + position + '.p')




    data_E['Y_TRUE'][:, 0] = denormalize(data_E['Y_TRUE'][:, 0], 'energy')
    data_E['Y_PRED'][:, 0] = denormalize(data_E['Y_PRED'][:, 0], 'energy')
    data_E['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E['Y_PRED'][:, 0] = calibrate_energy(data_E['Y_PRED'][:, 0], source=sources)

    data_E_Uonly['Y_TRUE'][:, 0] = denormalize(data_E_Uonly['Y_TRUE'][:, 0], 'energy')
    data_E_Uonly['Y_PRED'][:, 0] = denormalize(data_E_Uonly['Y_PRED'][:, 0], 'energy')
    data_E_Uonly['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E_Uonly['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E_Uonly['Y_PRED'][:, 0] = calibrate_energy(data_E_Uonly['Y_PRED'][:, 0], source=sources)


    data_E_Vonly['Y_TRUE'][:, 0] = denormalize(data_E_Vonly['Y_TRUE'][:, 0], 'energy')
    data_E_Vonly['Y_PRED'][:, 0] = denormalize(data_E_Vonly['Y_PRED'][:, 0], 'energy')
    data_E_Vonly['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E_Vonly['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E_Vonly['Y_PRED'][:, 0] = calibrate_energy(data_E_Vonly['Y_PRED'][:, 0], source=sources)


    # index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
    # for key in data_E.keys():
    #     data_E[key] = data_E[key][index_E]
    #
    # index_U = np.lexsort((data_U['MCEventNumber'], data_U['MCRunNumber']))
    # for key in data_U.keys():
    #     data_U[key] = data_U[key][index_U]
    #
    # index_U_zeroed = np.lexsort((data_U_zeroed['MCEventNumber'], data_U_zeroed['MCRunNumber']))
    # for key in data_U_zeroed.keys():
    #     data_U_zeroed[key] = data_U_zeroed[key][index_U_zeroed]
    #
    # index_V = np.lexsort((data_V['MCEventNumber'], data_V['MCRunNumber']))
    # for key in data_V.keys():
    #     data_V[key] = data_V[key][index_V]
    #
    # index_Z = np.lexsort((data_Z['MCEventNumber'], data_Z['MCRunNumber']))
    # for key in data_Z.keys():
    #     data_Z[key] = data_Z[key][index_Z]

    if purity_correction:
        # =======================
        #  Purity Correction
        # =======================
        print '\nSIMULATED DATA: ', sources, '\t ==> \t purity correction'
        drift_velocity = 0.00171  # drift velocity
        CV = 0.00174  # collection velocity
        collectionTime = 2940.0  # collection time
        lifetime = 4500000.0  # lifetime
        cathode_apdface_distance = 204.4065      # a
        apdplane_uplane_distance = 6.0     # b
        uplane_vplane_distance = 6.0       # c


        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z['Y_PRED'])) / drift_velocity + collectionTime

        # print
        # print '>>>>>>>>'
        # for i in range(20):
        #     if data_E['CCIsSS'][i] == 1:
        #         print
        #         print data_E['Y_PRED'][i]/(data_E['Y_PRED'][i] * np.exp(drifttime[i] / lifetime)), '\t', data_E['CCCorrectedEnergy'][i, 0]/data_E['CCPurityCorrectedEnergy'][i, 0]
        #         print data_E['Y_PRED'][i], '\t', data_E['CCCorrectedEnergy'][i, 0], '\t -----> \t', data_E['Y_PRED'][i] * np.exp(drifttime[i] / lifetime), '\t', data_E['CCPurityCorrectedEnergy'][i, 0], '\t -----> \t', data_E['Y_TRUE'][i]
        # print '>>>>>>>>'
        # print

        data_E['Y_PRED'] = data_E['Y_PRED'] * np.exp(drifttime / lifetime)
        # corrected_energy = data_E['Y_PRED'] * np.exp(drifttime / lifetime)



    else:
        print '\n', 'SIMULATED DATA: ',  sources, '\t--> \t NO purity correction'


    index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
    for key in data_E.keys():
        data_E[key] = data_E[key][index_E]

    index_E_Uonly = np.lexsort((data_E_Uonly['MCEventNumber'], data_E_Uonly['MCRunNumber']))
    for key in data_E_Uonly.keys():
        data_E_Uonly[key] = data_E_Uonly[key][index_E_Uonly]

    index_E_Vonly = np.lexsort((data_E_Vonly['MCEventNumber'], data_E_Vonly['MCRunNumber']))
    for key in data_E_Vonly.keys():
        data_E_Vonly[key] = data_E_Vonly[key][index_E_Vonly]


    # =======================
    #  Masking
    # =======================
    if sources == 'Th228':
        peak = 2614
    if sources == 'bb0n':
        peak = 2458


    mask_E_SS = np.sum(data_E['CCIsFiducial'], axis=1) == data_E['CCNumberClusters']
    mask_E_SS = np.logical_and(mask_E_SS, np.sum(data_E['CCIs3DCluster'], axis=1) == data_E['CCNumberClusters'])
    mask_E_SS = np.logical_and(mask_E_SS, data_E['CCIsSS'] == 1)
    mask_E_SS_peak = np.logical_and(mask_E_SS, data_E['MCEnergy'] > peak-10)
    mask_E_SS_peak = np.logical_and(mask_E_SS_peak, data_E['MCEnergy'] < peak+10)

    mask_E_SS_Uonly = np.sum(data_E_Uonly['CCIsFiducial'], axis=1) == data_E_Uonly['CCNumberClusters']
    mask_E_SS_Uonly = np.logical_and(mask_E_SS_Uonly, np.sum(data_E_Uonly['CCIs3DCluster'], axis=1) == data_E_Uonly['CCNumberClusters'])
    mask_E_SS_Uonly = np.logical_and(mask_E_SS_Uonly, data_E_Uonly['CCIsSS'] == 1)
    mask_E_SS_peak_Uonly = np.logical_and(mask_E_SS_Uonly, data_E_Uonly['MCEnergy'] > peak-10)
    mask_E_SS_peak_Uonly = np.logical_and(mask_E_SS_peak_Uonly, data_E_Uonly['MCEnergy'] < peak+10)

    mask_E_SS_Vonly = np.sum(data_E_Vonly['CCIsFiducial'], axis=1) == data_E_Vonly['CCNumberClusters']
    mask_E_SS_Vonly = np.logical_and(mask_E_SS_Vonly, np.sum(data_E_Vonly['CCIs3DCluster'], axis=1) == data_E_Vonly['CCNumberClusters'])
    mask_E_SS_Vonly = np.logical_and(mask_E_SS_Vonly, data_E_Vonly['CCIsSS'] == 1)
    mask_E_SS_peak_Vonly = np.logical_and(mask_E_SS_Vonly, data_E_Vonly['MCEnergy'] > peak-10)
    mask_E_SS_peak_Vonly = np.logical_and(mask_E_SS_peak_Vonly, data_E_Vonly['MCEnergy'] < peak+10)

    # print np.count_nonzero(np.logical_and(mask_E_SS_peak, mask_E_SS_peak_Uonly))
    # print np.count_nonzero(np.logical_and(mask_E_SS_peak, mask_E_SS_peak_Vonly))

    print '\n Start plotting'
    plt.rc('font', size=11)

    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    name_True = 'True'

    limit = 150

    residual_UV = data_E['Y_PRED'][:, 0][mask_E_SS_peak] - data_E['Y_TRUE'][:, 0][mask_E_SS_peak]
    residual_U = data_E_Uonly['Y_PRED'][:, 0][mask_E_SS_peak] - data_E_Uonly['Y_TRUE'][:, 0][mask_E_SS_peak]
    residual_V = data_E_Vonly['Y_PRED'][:, 0][mask_E_SS_peak] - data_E_Vonly['Y_TRUE'][:, 0][mask_E_SS_peak]

    mask_limit_UV = np.absolute(residual_UV) < limit
    mask_limit_U = np.absolute(residual_U) < limit
    mask_limit_V = np.absolute(residual_V) < limit

    residual_UV = residual_UV[mask_limit_UV]
    residual_U = residual_U[mask_limit_U]
    residual_V = residual_V[mask_limit_V]

    bins = 100

    # plt.hist(residual_UV, bins=bins, histtype='stepfilled', density=True, color=(0, 0.2, 0.4), alpha=0.2)
    # plt.hist(residual_U, bins=bins, histtype='stepfilled', density=True, color=(0.3, 0.69, 0.29), alpha=0.2)
    # plt.hist(residual_V, bins=bins, histtype='stepfilled', density=True, color=(0.6, 0.31, 0.64), alpha=0.2)

    plt.hist(residual_UV, bins=bins, histtype='step', density=True, color=(0, 0.2, 0.4), linewidth=1.5,
                                      # alpha=0.5,
                                      label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('UV:', np.mean(residual_UV), np.std(residual_UV)))
    plt.hist(residual_U, bins=bins, histtype='step', density=True, linewidth=1.5,
                                      color=(0.3, 0.69, 0.29),
                                      # alpha=0.5,
                                      label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('U:', np.mean(residual_U), np.std(residual_U)))
    plt.hist(residual_V, bins=bins, histtype='step', density=True, linewidth=1.5,
                                      color=(0.6, 0.31, 0.64),
                                      # alpha=0.5,
                                      label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('V:', np.mean(residual_V), np.std(residual_V)))
    plt.rc('font', size=11)
    plt.xlabel('Residual [keV]')
    plt.ylabel('Probability density')

    plt.legend(loc="best")
    plt.xlim(-limit, limit)

    plt.savefig(folderOUT + 'UVvsUvsV.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()


    residual_UV = data_E['Y_PRED'][:, 0][mask_E_SS_peak] - data_E['Y_TRUE'][:, 0][mask_E_SS_peak]
    residual_U = data_E_Uonly['Y_PRED'][:, 0][mask_E_SS_peak] - data_E_Uonly['Y_TRUE'][:, 0][mask_E_SS_peak]
    residual_V = data_E_Vonly['Y_PRED'][:, 0][mask_E_SS_peak] - data_E_Vonly['Y_TRUE'][:, 0][mask_E_SS_peak]
    mask_limit_UV = np.absolute(residual_UV) < limit
    mask_limit_U = np.absolute(residual_U) < limit
    mask_limit_V = np.absolute(residual_V) < limit

    plot_residual_correlation(dTrue=0,
                              dDNN=residual_U[np.logical_and(mask_limit_U, mask_limit_V)],
                              dEXO=residual_V[np.logical_and(mask_limit_U, mask_limit_V)], limit=limit,
                              mode='UvsV',
                              fOUT=folderOUT + 'UvsV_correlation' + '.pdf')
    plot_residual_correlation(dTrue=0,
                              dDNN=residual_UV[np.logical_and(mask_limit_UV, mask_limit_U)],
                              dEXO=residual_U[np.logical_and(mask_limit_UV, mask_limit_U)], limit=limit,
                              mode='UVvsU',
                              fOUT=folderOUT + 'UVvsU_correlation' + '.pdf')
    plot_residual_correlation(dTrue=0,
                              dDNN=residual_UV[np.logical_and(mask_limit_UV, mask_limit_V)],
                              dEXO=residual_V[np.logical_and(mask_limit_UV, mask_limit_V)], limit=limit,
                              mode='UVvsV',
                              fOUT=folderOUT + 'UVvsV_correlation' + '.pdf')

    return




def plotting_real_data(folderIN, folderOUT, sources, position, purity_correction):
    print '\n\n', sources
    print 'Reading files:'

    data_U = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190404-1528-19/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_V = read_hdf5_file_to_dict(folderIN + '190206-1314-19/190207-0000-00/190208-0000-00/190209-0000-00/190404-1531-29/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_Z = read_hdf5_file_to_dict(folderIN + '190206-1314-24/190207-0000-00/190208-0000-00/190209-0000-00/190404-1517-07/0physics-data/060-Th228-S5/events_060_Th228-S5.p')
    data_E = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190404-1518-09/0physics-data/060-Th228-S5/events_060_Th228-S5.p')

    data_U_MC = read_hdf5_file_to_dict(folderIN + '190206-1314-08/190207-0000-00/190208-0000-00/190209-0000-00/190314-1613-46/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
    data_V_MC = read_hdf5_file_to_dict(folderIN + '190206-1314-19/190207-0000-00/190208-0000-00/190209-0000-00/190315-1019-52/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
    data_Z_MC = read_hdf5_file_to_dict(folderIN + '190206-1314-24/190207-0000-00/190208-0000-00/190209-0000-00/190315-1018-03/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')
    data_E_MC = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190315-1021-10/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')


    data_U['Y_PRED'][:, 0] = denormalize(data_U['Y_PRED'][:, 0], 'U')
    data_V['Y_PRED'][:, 0] = denormalize(data_V['Y_PRED'][:, 0], 'V')
    data_Z['Y_PRED'][:, 0] = denormalize(data_Z['Y_PRED'][:, 0], 'Z')
    data_E['Y_PRED'][:, 0] = denormalize(data_E['Y_PRED'][:, 0], 'energy')

    data_U_MC['Y_TRUE'][:, 0] = denormalize(data_U_MC['Y_TRUE'][:, 0], 'U')
    data_U_MC['Y_PRED'][:, 0] = denormalize(data_U_MC['Y_PRED'][:, 0], 'U')
    data_V_MC['Y_TRUE'][:, 0] = denormalize(data_V_MC['Y_TRUE'][:, 0], 'V')
    data_V_MC['Y_PRED'][:, 0] = denormalize(data_V_MC['Y_PRED'][:, 0], 'V')
    data_Z_MC['Y_TRUE'][:, 0] = denormalize(data_Z_MC['Y_TRUE'][:, 0], 'Z')
    data_Z_MC['Y_PRED'][:, 0] = denormalize(data_Z_MC['Y_PRED'][:, 0], 'Z')
    data_E_MC['Y_TRUE'][:, 0] = denormalize(data_E_MC['Y_TRUE'][:, 0], 'energy')
    data_E_MC['Y_PRED'][:, 0] = denormalize(data_E_MC['Y_PRED'][:, 0], 'energy')


    index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
    for key in data_E.keys():
        data_E[key] = data_E[key][index_E]

    index_U = np.lexsort((data_U['MCEventNumber'], data_U['MCRunNumber']))
    for key in data_U.keys():
        data_U[key] = data_U[key][index_U]

    index_V = np.lexsort((data_V['MCEventNumber'], data_V['MCRunNumber']))
    for key in data_V.keys():
        data_V[key] = data_V[key][index_V]

    index_Z = np.lexsort((data_Z['MCEventNumber'], data_Z['MCRunNumber']))
    for key in data_Z.keys():
        data_Z[key] = data_Z[key][index_Z]


    index_E_MC = np.lexsort((data_E_MC['MCEventNumber'], data_E_MC['MCRunNumber']))
    for key in data_E_MC.keys():
        data_E_MC[key] = data_E_MC[key][index_E_MC]

    index_Z_MC = np.lexsort((data_Z_MC['MCEventNumber'], data_Z_MC['MCRunNumber']))
    for key in data_Z_MC.keys():
        data_Z_MC[key] = data_Z_MC[key][index_Z_MC]


    # print '\n\n\n>>>>>>>>'
    # print set(data_E['MCRunNumber'])


    if purity_correction:
        # =======================
        #  Purity Correction
        # =======================
        print
        print 'REAL DATA: ', sources, '\t ==> \t purity correction'
        drift_velocity = 0.00171
        drift_velocity = 0.171 #   cm / microsecond
        CV = 0.00174  # collection velocity
        collectionTime = 2940.0  #
        collectionTime = 2.9400  # microsecond
        lifetime = 4500000.0  # lifetime
        lifetime_MC = 4500 # microsecond
        cathode_apdface_distance = 204.4065      # a
        apdplane_uplane_distance = 6.0     # b
        uplane_vplane_distance = 6.0       # c

        runnumber = [4070, 4064, 5036, 5042]
        lifetime_run = [4462.2, 4412, 4096, 3436]
        lifetime = np.zeros(data_E['MCRunNumber'].shape)
        for i in range(4):
            lifetime_temp = data_E['MCRunNumber'] == runnumber[i]
            lifetime_temp = lifetime_temp * lifetime_run[i]
            lifetime += lifetime_temp

        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z['Y_PRED'])) *0.1 / drift_velocity + collectionTime
        lifetime = np.reshape(lifetime, (100000, 1))

        data_E['Y_PRED'] = data_E['Y_PRED'] * np.exp(drifttime / lifetime)


        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z_MC['Y_PRED']))*0.1 / drift_velocity + collectionTime
        data_E_MC['Y_PRED'] = data_E_MC['Y_PRED'] * np.exp(drifttime / lifetime_MC)

    else:
        print sources, '\t--> \t NO purity correction'


    data_E['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E['Y_PRED'][:, 0] = calibrate_energy(data_E['Y_PRED'][:, 0], source=sources)

    data_E_MC['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E_MC['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E_MC['Y_PRED'][:, 0] = calibrate_energy(data_E_MC['Y_PRED'][:, 0], source=sources)



    # =======================
    #  Masking
    # =======================
    if True:
        if sources == 'Th228':
            peak = 2614
        if sources == 'bb0n':
            peak = 2458



        mask_U_SS = np.sum(data_U['CCIsFiducial'], axis=1) == data_U['CCNumberClusters']
        mask_U_SS = np.logical_and(mask_U_SS, np.sum(data_U['CCIs3DCluster'], axis=1) == data_U['CCNumberClusters'])
        mask_U_SS = np.logical_and(mask_U_SS, data_U['CCIsSS'] == 1)
        mask_U_SS_peak = np.logical_and(mask_U_SS, data_U['CCPurityCorrectedEnergy'][:, 0] > peak-15)
        mask_U_SS_peak = np.logical_and(mask_U_SS_peak, data_U['CCPurityCorrectedEnergy'][:, 0] < peak+15)
        mask_num_U_is_1 = data_U['CCNumberUWires'] == 1
        mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, mask_U_SS_peak)
        mask_num_U_is_not_1 = data_U['CCNumberUWires'] > 1
        mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, mask_U_SS_peak)


        mask_V_SS = np.sum(data_V['CCIsFiducial'], axis=1) == data_V['CCNumberClusters']
        mask_V_SS = np.logical_and(mask_V_SS, np.sum(data_V['CCIs3DCluster'], axis=1) == data_V['CCNumberClusters'])
        mask_V_SS = np.logical_and(mask_V_SS, data_V['CCIsSS'] == 1)
        # mask_V_SS_peak = np.logical_and(mask_V_SS, data_V['MCEnergy'] > peak-10)
        # mask_V_SS_peak = np.logical_and(mask_V_SS_peak, data_V['MCEnergy'] < peak+10)

        mask_Z_SS = np.sum(data_Z['CCIsFiducial'], axis=1) == data_Z['CCNumberClusters']
        mask_Z_SS = np.logical_and(mask_Z_SS, np.sum(data_Z['CCIs3DCluster'], axis=1) == data_Z['CCNumberClusters'])
        mask_Z_SS = np.logical_and(mask_Z_SS, data_Z['CCIsSS'] == 1)
        # mask_Z_SS_peak = np.logical_and(mask_Z_SS, data_Z['MCEnergy'] > peak-10)
        # mask_Z_SS_peak = np.logical_and(mask_Z_SS_peak, data_Z['MCEnergy'] < peak+10)

        mask_E_SS = np.sum(data_E['CCIsFiducial'], axis=1) == data_E['CCNumberClusters']
        mask_E_SS = np.logical_and(mask_E_SS, np.sum(data_E['CCIs3DCluster'], axis=1) == data_E['CCNumberClusters'])
        mask_E_SS = np.logical_and(mask_E_SS, data_E['CCIsSS'] == 1)
        # mask_E_SS_peak = np.logical_and(mask_E_SS, data_E['MCEnergy'] > peak-10)
        # mask_E_SS_peak = np.logical_and(mask_E_SS_peak, data_E['MCEnergy'] < peak+10)




        mask_U_SS_MC = np.sum(data_U_MC['CCIsFiducial'], axis=1) == data_U_MC['CCNumberClusters']
        mask_U_SS_MC = np.logical_and(mask_U_SS_MC, np.sum(data_U_MC['CCIs3DCluster'], axis=1) == data_U_MC['CCNumberClusters'])
        mask_U_SS_MC = np.logical_and(mask_U_SS_MC, data_U_MC['CCIsSS'] == 1)
        mask_U_SS_peak_MC = np.logical_and(mask_U_SS_MC, data_U_MC['MCEnergy'] > peak - 15)
        mask_U_SS_peak_MC = np.logical_and(mask_U_SS_peak_MC, data_U_MC['MCEnergy'] < peak + 15)
        mask_num_U_is_1_MC = data_U_MC['CCNumberUWires'] == 1
        mask_num_U_is_1_MC = np.logical_and(mask_num_U_is_1_MC, mask_U_SS_peak_MC)

        mask_V_SS_MC = np.sum(data_V_MC['CCIsFiducial'], axis=1) == data_V_MC['CCNumberClusters']
        mask_V_SS_MC = np.logical_and(mask_V_SS_MC, np.sum(data_V_MC['CCIs3DCluster'], axis=1) == data_V_MC['CCNumberClusters'])
        mask_V_SS_MC = np.logical_and(mask_V_SS_MC, data_V_MC['CCIsSS'] == 1)

        mask_Z_SS_MC = np.sum(data_Z_MC['CCIsFiducial'], axis=1) == data_Z_MC['CCNumberClusters']
        mask_Z_SS_MC = np.logical_and(mask_Z_SS_MC, np.sum(data_Z_MC['CCIs3DCluster'], axis=1) == data_Z_MC['CCNumberClusters'])
        mask_Z_SS_MC = np.logical_and(mask_Z_SS_MC, data_Z_MC['CCIsSS'] == 1)

        mask_E_SS_MC = np.sum(data_E_MC['CCIsFiducial'], axis=1) == data_E_MC['CCNumberClusters']
        mask_E_SS_MC = np.logical_and(mask_E_SS_MC, np.sum(data_E_MC['CCIs3DCluster'], axis=1) == data_E_MC['CCNumberClusters'])
        mask_E_SS_MC = np.logical_and(mask_E_SS_MC, data_E_MC['CCIsSS'] == 1)



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

    plot_diagonal(x=data_U['CCPosU'][:, 0][mask_U_SS], y=data_U['Y_PRED'][:, 0][mask_U_SS], xlabel=name_True, ylabel=name_DNN, mode='U', mode2='real',
                  fOUT=(folderOUT + sources + '_' + position + '_U_scatter' + '.pdf'))

    # plot_diagonal(x=data_U['Y_TRUE'][:, 0][mask_U_SS], y=data_U['CCPosU'][:, 0][mask_U_SS], xlabel=name_True,
    #               ylabel=name_EXO, mode='U',
    #               fOUT=(folderOUT + sources + '_' + position + '_U_scatter_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_U['Y_PRED'][:, 0][mask_U_SS], dEXO=data_U['CCPosU'][:, 0][mask_U_SS], dTrue=data_U_MC['Y_TRUE'][:, 0][mask_U_SS_MC], mode2='real',
                  mode='U', fOUT=(folderOUT + sources + '_' + position + '_U_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=None,
                        dDNN=data_U['Y_PRED'][:, 0][mask_U_SS],
                        dEXO=data_U['CCPosU'][:, 0][mask_U_SS],
                        # dMC_DNN=data_U_MC['Y_PRED'][:, 0][mask_num_U_is_1_MC],
                        # dMC_EXO=data_U_MC['CCPosU'][:, 0][mask_num_U_is_1_MC],
                        dMC_DNN=data_U_MC['Y_PRED'][:, 0][mask_U_SS_MC],
                        dMC_EXO=data_U_MC['CCPosU'][:, 0][mask_U_SS_MC],
                        limit=7, mode='real_V',
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_histo' + '.pdf')

    plot_residual_real(DNN_one=data_U['Y_PRED'][:, 0][mask_num_U_is_1], EXO_one=data_U['CCPosU'][:, 0][mask_num_U_is_1],
                       DNN_two=data_U['Y_PRED'][:, 0][mask_num_U_is_not_1], EXO_two=data_U['CCPosU'][:, 0][mask_num_U_is_not_1],
                       fOUT=folderOUT + sources + '_' + position + '_U_histo_onetwo' + '.pdf', limit=10)


    # =======================
    #  V Plots
    # =======================
    plot_diagonal(x=data_V['CCPosV'][:, 0][mask_V_SS], y=data_V['Y_PRED'][:, 0][mask_V_SS], xlabel=name_True, ylabel=name_DNN, mode='V', mode2='real',
                  fOUT=(folderOUT + sources + '_' + position + '_V_scatter' + '.pdf'))

    # plot_diagonal(x=data_V['Y_TRUE'][:, 0][mask_V_SS], y=data_V['CCPosV'][:, 0][mask_V_SS], xlabel=name_True,
    #           ylabel=name_EXO, mode='V',
    #           fOUT=(folderOUT + sources + '_' + position + '_V_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_V['Y_PRED'][:, 0][mask_V_SS], dEXO=data_V['CCPosV'][:, 0][mask_V_SS], dTrue=data_V_MC['Y_TRUE'][:, 0][mask_V_SS_MC], mode2='real',
                  mode='V', fOUT=(folderOUT + sources + '_' + position + '_V_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=None,
                        dDNN=data_V['Y_PRED'][:, 0][mask_V_SS],
                        dEXO=data_V['CCPosV'][:, 0][mask_V_SS],
                        dMC_DNN=data_V_MC['Y_PRED'][:, 0][mask_V_SS_MC],
                        dMC_EXO=data_V_MC['CCPosV'][:, 0][mask_V_SS_MC],
                        limit=5, mode='real_V',
                        title='V', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_V_histo' + '.pdf')



    # =======================
    #  Z Plots
    # =======================


    plot_diagonal(x=data_Z['CCPosZ'][:, 0][mask_Z_SS], y=data_Z['Y_PRED'][:, 0][mask_Z_SS], xlabel=name_True, ylabel=name_DNN, mode='Z', mode2='real',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_scatter' + '.pdf'))

    # plot_diagonal(x=data_Z['Y_TRUE'][:, 0][mask_Z_SS], y=data_Z['CCPosZ'][:, 0][mask_Z_SS], xlabel=name_True,
    #               ylabel=name_EXO, mode='Z',
    #               fOUT=(folderOUT + sources + '_' + position + '_Z_EXO' + '.pdf'))

    plot_spectrum(dCNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                  dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS],
                  dTrue=data_Z_MC['Y_TRUE'][:, 0][mask_Z_SS_MC],
                  mode='Z', mode2='real_Z',
                  fOUT=(folderOUT + sources + '_' + position + '_Z_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=None,
                        dDNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                        dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS],
                        dMC_DNN=data_Z_MC['Y_PRED'][:, 0][mask_Z_SS_MC],
                        dMC_EXO=data_Z_MC['CCPosZ'][:, 0][mask_Z_SS_MC],
                        limit=5, mode='real',
                        title='Z', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_Z_histo' + '.pdf')

    # =======================
    #  E Plots
    # =======================

    plot_diagonal(x=data_E['Y_PRED'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_DNN, mode2='real',
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_E_scatter' + '.pdf'))

    plot_spectrum(dCNN=data_E['Y_PRED'][:, 0][mask_E_SS], dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
              dTrue=data_E_MC['Y_TRUE'][:, 0][mask_E_SS_MC],
              mode='Energy', mode2='real',
              fOUT=(folderOUT + sources + '_' + position + '_E_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=None,
                        dDNN=data_E['Y_PRED'][:, 0][mask_E_SS],
                        dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
                        dMC_DNN=data_E_MC['Y_PRED'][:, 0][mask_E_SS_MC],
                        dMC_EXO=data_E_MC['CCPurityCorrectedEnergy'][:, 0][mask_E_SS_MC],
                        limit=150, mode='real',
                        title='E', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_E_histo' + '.pdf')

    # =======================
    #  Additional Plots
    # =======================

    mc_DNN_test = np.random.normal(scale=0.5, size=10000)
    mc_EXO_test = np.random.normal(scale=1.1, size=10000)
    real_DNN_test = np.random.normal(scale=0.5, size=10000)
    real_EXO_test = np.random.normal(scale=1.1, size=10000)

    plot_residual_histo(dTrue=None, dDNN=real_DNN_test, dEXO=real_EXO_test,
                        dMC_DNN=mc_DNN_test, dMC_EXO=mc_EXO_test,
                        limit=10, mode='real',
                        title='test same', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_test_histo_same' + '.pdf')

    mc_DNN_test = np.random.normal(scale=0.5, size=10000)
    mc_EXO_test = np.random.normal(scale=1.1, size=10000)
    real_DNN_test = np.random.normal(scale=2.75, size=10000)
    real_EXO_test = np.random.normal(scale=1.55, size=10000, loc=0)

    plot_residual_histo(dTrue=None, dDNN=real_DNN_test, dEXO=real_EXO_test,
                        dMC_DNN=mc_DNN_test, dMC_EXO=mc_EXO_test,
                        limit=10, mode='real',
                        title='test same', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_test_histo_different' + '.pdf')

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
        data_E_old = read_hdf5_file_to_dict(folderIN + '190206-1314-28/190207-0000-00/190208-0000-00/190209-0000-00/190414-1152-55/1validation-data/' + sources + '-' + position + '/events_060_' + sources + '-' + position + '.p')

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

    if sources == 'Th228':
        data_E_old['Y_TRUE'][:, 0] = denormalize(data_E_old['Y_TRUE'][:, 0], 'energy')
        data_E_old['Y_PRED'][:, 0] = denormalize(data_E_old['Y_PRED'][:, 0], 'energy')

    data_E['CCPurityCorrectedEnergy'][:, 0] = calibrate_energy(data_E['CCPurityCorrectedEnergy'][:, 0], source=sources)
    data_E['Y_PRED'][:, 0] = calibrate_energy(data_E['Y_PRED'][:, 0], source=sources)



    index_E = np.lexsort((data_E['MCEventNumber'], data_E['MCRunNumber']))
    for key in data_E.keys():
        data_E[key] = data_E[key][index_E]

    index_U = np.lexsort((data_U['MCEventNumber'], data_U['MCRunNumber']))
    for key in data_U.keys():
        data_U[key] = data_U[key][index_U]

    index_U_zeroed = np.lexsort((data_U_zeroed['MCEventNumber'], data_U_zeroed['MCRunNumber']))
    for key in data_U_zeroed.keys():
        data_U_zeroed[key] = data_U_zeroed[key][index_U_zeroed]

    index_V = np.lexsort((data_V['MCEventNumber'], data_V['MCRunNumber']))
    for key in data_V.keys():
        data_V[key] = data_V[key][index_V]

    index_Z = np.lexsort((data_Z['MCEventNumber'], data_Z['MCRunNumber']))
    for key in data_Z.keys():
        data_Z[key] = data_Z[key][index_Z]

    if purity_correction:
        # =======================
        #  Purity Correction
        # =======================
        print '\nSIMULATED DATA: ', sources, '\t ==> \t purity correction'
        drift_velocity = 0.00171
        drift_velocity = 0.171  # cm / microsecond
        CV = 0.00174  # collection velocity
        collectionTime = 2940.0  #
        collectionTime = 2.9400  # microsecond
        lifetime = 4500000.0  # lifetime
        lifetime = 4500.  # microsecond
        cathode_apdface_distance = 204.4065  # a
        apdplane_uplane_distance = 6.0  # b
        uplane_vplane_distance = 6.0  # c


        drifttime = (cathode_apdface_distance - apdplane_uplane_distance - uplane_vplane_distance - np.absolute(data_Z['Y_PRED'])) *0.1 / drift_velocity + collectionTime

        # print '\n\n\n>>>>>>>>>>>'
        # print data_Z['Y_PRED'][0:10]
        # print drifttime[0:10]

        data_E['Y_PRED'] = data_E['Y_PRED'] * np.exp(drifttime / lifetime)


    else:
        print '\n', 'SIMULATED DATA: ',  sources, '\t--> \t NO purity correction'



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
    mask_U_SS_peak = np.logical_and(mask_U_SS, data_U['MCEnergy'] > peak-15)
    mask_U_SS_peak = np.logical_and(mask_U_SS_peak, data_U['MCEnergy'] < peak+15)
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
    mask_E_SS_peak = np.logical_and(mask_E_SS, data_E['MCEnergy'] > peak-15)
    mask_E_SS_peak = np.logical_and(mask_E_SS_peak, data_E['MCEnergy'] < peak+15)

    if sources == 'Th228':
        mask_E_SS_old = data_E_old['MCIsFiducial'] == data_E_old['CCNumberClusters']
        mask_E_SS_old = np.logical_and(mask_E_SS_old, np.sum(data_E_old['CCIs3DCluster'], axis=1) == data_E_old['CCNumberClusters'])
        mask_E_SS_old = np.logical_and(mask_E_SS_old, data_E_old['CCIsSS'] == 1)
        mask_E_SS_peak_old = np.logical_and(mask_E_SS_old, data_E_old['MCEnergy'][:, 0] > peak-15)
        mask_E_SS_peak_old = np.logical_and(mask_E_SS_peak_old, data_E_old['MCEnergy'][:, 0] < peak+15)


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
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_1], limit=15,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_onewire' + '.pdf')

    plot_residual_histo(dTrue=data_U_zeroed['Y_TRUE'][:, 0][mask_num_U_is_1_zeroed],
                        dDNN=data_U_zeroed['Y_PRED'][:, 0][mask_num_U_is_1_zeroed],
                        dEXO=data_U_zeroed['CCPosU'][:, 0][mask_num_U_is_1_zeroed], limit=15,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_onewire_noInduction' + '.pdf')


    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_num_U_is_not_1],
                        dDNN=data_U['Y_PRED'][:, 0][mask_num_U_is_not_1],
                        dEXO=data_U['CCPosU'][:, 0][mask_num_U_is_not_1], limit=10,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_multiwire' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS],
                        dDNN=data_U['Y_PRED'][:, 0][mask_U_SS],
                        dEXO=data_U['CCPosU'][:, 0][mask_U_SS], limit=4,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS' + '.pdf')

    plot_residual_histo(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS_peak],
                        dDNN=data_U['Y_PRED'][:, 0][mask_U_SS_peak],
                        dEXO=data_U['CCPosU'][:, 0][mask_U_SS_peak], limit=10,
                        title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_U_SS_peak' + '.pdf')

    print '\n\n >>>>>>>>>>>>>>>>>'
    print 'Pearson coefficient of residual for simulated data Th228 at S5'
    print pearsonr(data_U['Y_PRED'][:, 0][mask_num_U_is_1]-data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                   data_U['CCPosU'][:, 0][mask_num_U_is_1]-data_U['Y_TRUE'][:, 0][mask_num_U_is_1])


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

    if sources == 'bb0n':
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
                     fOUT=folderOUT + sources + '_' + position + '_U_boxplot_wirecheck' + '.pdf',
                     mode='wirecheck')

        plot_boxplot(dTrue=data_U['Y_TRUE'][:, 0][mask_U_SS_peak],
                     dTrue_masked=data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                     dEXO=data_U['CCPosU'][:, 0][mask_U_SS_peak],
                     dEXO_masked=data_U['CCPosU'][:, 0][mask_num_U_is_1],
                     dDNN=data_U['Y_PRED'][:, 0][mask_U_SS_peak],
                     dDNN_masked=data_U['Y_PRED'][:, 0][mask_num_U_is_1],
                     dTrue_masked_zeroed=data_U['Y_TRUE'][:, 0][mask_num_U_is_not_1],
                     dEXO_masked_zeroed=data_U['CCPosU'][:, 0][mask_num_U_is_not_1],
                     dDNN_masked_zeroed=data_U['Y_PRED'][:, 0][mask_num_U_is_not_1],
                     title='Boxplot', name_DNN_masked='DNN onewire', name_EXO_masked='EXO onewire',
                     fOUT=folderOUT + sources + '_' + position + '_U_boxplot_SS' + '.pdf',
                     mode='SS')



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
                        title='V', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=2,
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
                        dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS], limit=1.,
                        # dEXO=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]), limit=5,
                        title='Z', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                        fOUT=folderOUT + sources + '_' + position + '_Z_histo' + '.pdf')

    plot_residual_correlation(dTrue=data_Z['Y_TRUE'][:, 0][mask_Z_SS],
                              dDNN=data_Z['Y_PRED'][:, 0][mask_Z_SS],
                              dEXO=data_Z['CCPosZ'][:, 0][mask_Z_SS], limit=3,
                              mode='Z',
                              fOUT=folderOUT + sources + '_' + position + '_Z_SS_correlation' + '.pdf')

    # =======================
    #  E Plots
    # =======================

    plot_diagonal(x=data_E['Y_TRUE'][:, 0][mask_E_SS], y=data_E['Y_PRED'][:, 0][mask_E_SS],
                  xlabel=name_True, ylabel=name_DNN, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_E_DNN' + '.pdf'))

    plot_diagonal(x=data_E['Y_TRUE'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_True,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_E_EXO' + '.pdf'))

    plot_diagonal(x=data_E['Y_PRED'][:, 0][mask_E_SS], y=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], xlabel=name_DNN,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + sources + '_' + position + '_E_Both' + '.pdf'))

    plot_spectrum(dCNN=data_E['Y_PRED'][:, 0][mask_E_SS], dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
              dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS],
              mode='Energy',
              fOUT=(folderOUT + sources + '_' + position + '_E_spectrum' + '.pdf'))

    plot_residual_histo(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS], dDNN=data_E['Y_PRED'][:, 0][mask_E_SS],
                        dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
                        title='Energy', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=100,
                        fOUT=folderOUT + sources + '_' + position + '_E_histo' + '.pdf')

    plot_residual_histo(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS_peak], dDNN=data_E['Y_PRED'][:, 0][mask_E_SS_peak],
                        dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS_peak],
                        title='Energy', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=150,
                        fOUT=folderOUT + sources + '_' + position + '_E_histo_peak' + '.pdf')

    plot_hexbin(data_E['MCPosX'][:][mask_E_SS], data_E['MCPosY'][:][mask_E_SS], data_E['MCPosZ'][:][mask_E_SS],
                data_E['Y_TRUE'][:, 0][mask_E_SS] - data_E['Y_PRED'][:, 0][mask_E_SS],
                data_E['Y_TRUE'][:, 0][mask_E_SS] - data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS],
                'X [mm]', 'Y [mm]',
                fOUT=folderOUT + sources + '_' + position + '_E_hexbin_xy' + '.pdf')


    plot_residual_correlation(dTrue=data_E['Y_TRUE'][:, 0][mask_E_SS],
                              dDNN=data_E['Y_PRED'][:, 0][mask_E_SS],
                              dEXO=data_E['CCPurityCorrectedEnergy'][:, 0][mask_E_SS], limit=200,
                              mode='energy',
                              fOUT=folderOUT + sources + '_' + position + '_E_SS_correlation' + '.pdf')


    if sources == 'Th228':
        plot_residual_real(DNN_one=data_E_old['Y_PRED'][:, 0][mask_E_SS_old], EXO_one=data_E_old['Y_TRUE'][:, 0][mask_E_SS_old],
                           DNN_two=data_E['Y_PRED'][:, 0][mask_E_SS], EXO_two=data_E['Y_TRUE'][:, 0][mask_E_SS],
                           fOUT=folderOUT + sources + '_' + position + '_E_electron_lifetime' + '.pdf', limit=150, mode='mc')


    # plot_hexbin(data['EVENT_INFO']['MCPosX'][:, 0], data['EVENT_INFO']['MCPosY'][:, 0], np.absolute(data['Y_TRUE'][:, 0] - data['EVENT_INFO']['CCCorrectedEnergy'][:, 0]), 'X [mm]', 'Y[mm]',
    #             fOUT=folderOUT + dir_residual + sources + '_' + position + '_Energy_EXO_hexbin_' + epoch + '.pdf')


    # =======================
    #  Additional Plots
    # =======================
    # print '>>>>>'
    # print data_E['MCPosU'][mask_E_SS][10], '\t', data_U['Y_TRUE'][mask_E_SS][10], '\t', data_E['CCPosU'][mask_E_SS][10,0]
    # print data_E['MCPosU'][mask_E_SS][20], '\t', data_U['Y_TRUE'][mask_E_SS][20], '\t', data_E['CCPosU'][mask_E_SS][20,0]
    # print data_E['MCPosU'][mask_E_SS][30], '\t', data_U['Y_TRUE'][mask_E_SS][30], '\t', data_E['CCPosU'][mask_E_SS][30,0]
    # print data_E['MCPosU'][mask_E_SS][40], '\t', data_U['Y_TRUE'][mask_E_SS][40], '\t', data_E['CCPosU'][mask_E_SS][40,0]
    # print data_E['MCPosU'][mask_E_SS][50], '\t', data_U['Y_TRUE'][mask_E_SS][50], '\t', data_E['CCPosU'][mask_E_SS][50,0]
    # print
    # print np.count_nonzero(np.logical_and(mask_U_SS, mask_E_SS))
    # print
    # print '>>>>>'

    x, y, z = uvz_to_xyz(data_U['Y_PRED'][:, 0][mask_E_SS], data_V['Y_PRED'][:, 0][mask_E_SS], data_Z['Y_PRED'][:, 0][mask_E_SS])
    plot_hexbin_3D(data_E['MCPosX'][:][mask_E_SS], data_E['MCPosY'][:][mask_E_SS], data_E['MCPosZ'][:][mask_E_SS],
                x, y, z,
                data_E['CCPosX'][:, 0][mask_E_SS], data_E['CCPosY'][:, 0][mask_E_SS], data_E['CCPosZ'][:, 0][mask_E_SS],
                name_x='X [mm]', name_y='Y [mm]',
                fOUT=folderOUT + sources + '_' + position + '_position_3d_hexbin' + '.pdf')

    plot_residual_correlation_3D(data_E['MCPosX'][:][mask_E_SS], data_E['MCPosY'][:][mask_E_SS], data_E['MCPosZ'][:][mask_E_SS],
                x, y, z,
                data_E['CCPosX'][:, 0][mask_E_SS], data_E['CCPosY'][:, 0][mask_E_SS], data_E['CCPosZ'][:, 0][mask_E_SS],
                name_x='X [mm]', name_y='Y [mm]',
                fOUT=folderOUT + sources + '_' + position + '_position_3d_correlation' + '.pdf')

    plot_induction(data_U['wf_max_neighbour'][mask_num_U_is_1],
                   data_U['Y_PRED'][:, 0][mask_num_U_is_1] - data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                   data_U['CCPosU'][:, 0][mask_num_U_is_1] - data_U['Y_TRUE'][:, 0][mask_num_U_is_1],
                   fOUT=folderOUT + sources + '_' + position + '_U_induction' + '.pdf', limit=4)




    print 'Finished plotting'

    return

def plot_residual_correlation_3D(X_true, Y_true, Z_true, X_DNN, Y_DNN, Z_DNN, X_EXO, Y_EXO, Z_EXO, name_x, name_y, fOUT):
    from matplotlib.ticker import NullFormatter
    from matplotlib.font_manager import FontProperties
    from scipy.stats import pearsonr
    plt.rc('font', size=13)

    limit = 6

    residual_DNN = np.sqrt((X_DNN - X_true) ** 2 + (Y_DNN - Y_true) ** 2 + (Z_DNN - Z_true) ** 2)
    residual_EXO = np.sqrt((X_EXO - X_true) ** 2 + (Y_EXO - Y_true) ** 2 + (Z_EXO - Z_true) ** 2)


    mask_limit_EXO = np.absolute(residual_EXO) < limit
    mask_limit_DNN = np.absolute(residual_DNN) < limit
    mask_limit = np.logical_and(mask_limit_DNN, mask_limit_EXO)


    bins = 200

    # plt.text(5.5, 4000,
    #              'EXO:\n$\mu=%.1f$\n$\sigma=%.2f$' % (np.mean(residual_EXO[mask_limit_EXO]), np.std(residual_EXO[mask_limit_EXO])),
    #              bbox=dict(fc="none"), horizontalalignment='left',
    #              verticalalignment='top')
    # plt.text(5.5, 2500,
    #              'DNN:\n$\mu=%.1f$\n$\sigma=%.2f$' % (np.mean(residual_DNN[mask_limit_DNN]), np.std(residual_DNN[mask_limit_DNN])),
    #              bbox=dict(fc="none"), horizontalalignment='left',
    #              verticalalignment='top')

    n_x, bins_x, patches_x = plt.hist(residual_EXO[mask_limit_EXO], bins=bins, histtype='stepfilled', color=(1., 0.49803922, 0.05490196), alpha=0.4,
                                      label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('EXO:', np.mean(residual_EXO[mask_limit_EXO]), np.std(residual_EXO[mask_limit_EXO])))
    n_y, bins_y, patches_y = plt.hist(residual_DNN[mask_limit_DNN], bins=bins, histtype='stepfilled', color=(0, 0.2, 0.4), alpha=0.5,
                                      label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('DNN:', np.mean(residual_DNN[mask_limit_DNN]), np.std(residual_DNN[mask_limit_DNN])))


    plt.xlabel('Distance from MC position [mm]')
    plt.ylabel('Counts')

    plt.xlim(0, limit)
    plt.legend(loc="best")

    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return



def plot_hexbin_3D(X_true, Y_true, Z_true, X_DNN, Y_DNN, Z_DNN, X_EXO, Y_EXO, Z_EXO, name_x, name_y, fOUT):
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()

    plt.rc('font', size=35)

    xmin = X_true.min() - 10
    xmax = X_true.max() + 10
    ymin = Y_true.min() - 10
    ymax = Y_true.max() + 10
    zmin = Z_true.min() - 10
    zmax = Z_true.max() + 10

    residual_DNN = np.sqrt((X_DNN - X_true) ** 2 + (Y_DNN - Y_true) ** 2 + (Z_DNN - Z_true) ** 2)
    residual_EXO = np.sqrt((X_EXO - X_true) ** 2 + (Y_EXO - Y_true) ** 2 + (Z_EXO - Z_true) ** 2)


    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(25, 25))#, gridspec_kw={'height_ratios': [3, 1]})
    axes[0, 0].set(aspect='auto')#, adjustable='box-forced')
    axes[1, 0].set(aspect='auto')#, adjustable='box-forced')
    axes[0, 1].set(aspect='auto')#, adjustable='box-forced')
    axes[1, 1].set(aspect='auto')#, adjustable='box-forced')

    axes[0, 0].axis([xmin, xmax, zmin, zmax])
    axes[0, 1].axis([xmin, xmax, zmin, zmax])
    axes[1, 0].axis([xmin, xmax, ymin, ymax])
    axes[1, 1].axis([xmin, xmax, ymin, ymax])

    num_bins = 50
    vmax = 1.5

    ax = axes[0, 0]
    ax.set_title('DNN')
    im = ax.hexbin(X_true, Z_true, C=residual_DNN, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[0, 1]
    ax.set_title('EXO')
    im = ax.hexbin(X_true, Z_true, C=residual_EXO, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[1, 0]
    # ax.set_title('DNN')
    im = ax.hexbin(X_true, Y_true, C=residual_DNN, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[1, 1]
    # ax.set_title('EXO')
    im = ax.hexbin(X_true, Y_true, C=residual_EXO, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, vmin=0., vmax=vmax)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('Spatial resolution')


    axes[1, 0].set_xlabel('%s' % (name_x))
    axes[1, 1].set_xlabel('%s' % (name_x))
    axes[0, 0].set_ylabel('Z [mm]')
    axes[1, 0].set_ylabel('Y [mm]')

    plt.subplots_adjust(hspace=0.02)
    plt.subplots_adjust(wspace=0.02)

    axes[0, 1].yaxis.set_major_formatter(nullfmt)
    axes[1, 1].yaxis.set_major_formatter(nullfmt)

    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def uvz_to_xyz(u, v, z):
    y = (u+v)/np.sqrt(3.)
    x = v-u

    mask_zneg = z < 0
    x[mask_zneg] = -x[mask_zneg]
    return x, y, z


def plot_induction(induction, DNN_residual, EXO_residual, fOUT, limit=10):
    plt.rc('font', size=32)
    xmin = 0
    xmax = 55
    # ymin = E_y.min()-10
    # ymax = E_y.max()+10

    mask_limit = np.logical_and(np.absolute(DNN_residual) < limit, np.absolute(EXO_residual) < limit)

    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(20, 10))  # , gridspec_kw={'height_ratios': [3, 1]})
    axes[0].set(aspect='auto', adjustable='box-forced')
    axes[1].set(aspect='auto', adjustable='box-forced')
    # axes[0].axis([xmin, xmax, -10, 10])

    num_bins = 80


    # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
    ax = axes[0]
    ax.set_title('DNN')
    # im = ax.hexbin(E_x, E_y, C=DNN, cmap='RdBu_r', linewidths=0.1, gridsize=num_bins, norm=MidpointNormalize(midpoint=0.))
    im = ax.hexbin(induction, DNN_residual, cmap='viridis', linewidths=0.1, gridsize=num_bins, bins='log', mincnt=1, extent=(xmin, xmax, -limit, limit))
    ax.axhline(y=0, xmin=xmin, xmax=xmax, color='black', alpha=0.3)
    # (num_bins, int(num_bins/((xmax-xmin)/(limit * 2.))))

    ax = axes[1]
    ax.set_title('EXO')
    plt.setp(ax.get_yticklabels(), visible=False)
    # im = ax.hexbin(E_x, E_y, C=EXO, cmap='RdBu_r', linewidths=0.1, gridsize=num_bins, norm=MidpointNormalize(midpoint=0.))
    im = ax.hexbin(induction, EXO_residual, cmap='viridis', linewidths=0.1, gridsize=num_bins, bins='log', mincnt=1, extent=(xmin, xmax, -limit, limit))    #(num_bins, int(num_bins/((xmax-xmin)/(limit * 2.))))
    ax.axhline(y=0, xmin=xmin, xmax=xmax, color='black', alpha=0.3)


    fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = fig.add_axes([0.91, 0.11, 0.02, 0.78])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('Counts')

    axes[0].set_xlabel('Induction [a.u.]')
    axes[1].set_xlabel('Induction [a.u.]')
    axes[0].set_ylabel('Residual [mm]')

    axes[0].set_xlim(xmin=xmin, xmax=xmax)
    axes[1].set_xlim(xmin=xmin, xmax=xmax)

    plt.subplots_adjust(wspace=0.025)

    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def calibrate_energy(energy, source):
    from scipy.stats import norm
    from scipy.optimize import curve_fit


    # if source == 'Th228':
    #     peak = 2614.53
    # elif source == 'bb0n':
    #     peak = 2457.83
    # else:
    #     print
    #     print
    #     raise ValueError('>>>>>>>>>>  Peak position of that source not known for energy calibration  <<<<<<<<<<')
    #
    # # range = 10
    # # mask_peak = [energy > peak - range]
    # # mask_peak = np.logical_and(mask_peak, [energy < peak + range])
    # # mask_peak = np.reshape(mask_peak, (-1,))
    # # mu, std = norm.fit(energy[mask_peak])
    # # energy = energy * peak / mu
    # # # test
    # # mask_peak = [energy > peak - range]
    # # mask_peak = np.logical_and(mask_peak, [energy < peak + range])
    # # mask_peak = np.reshape(mask_peak, (-1,))
    # # mu2, std2 = norm.fit(energy[mask_peak])
    #
    #
    # fit_range = 20
    #
    # bins = 2000
    # range = (1000, 3000)
    # hist, bin_edges = np.histogram(energy, bins=bins, range=range)
    # bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    # # peak_pos = find_peak(hist=hist, bin_centres=bin_centres, peakpos=peak, peakfinder='max')
    #
    # hist2 = hist[1500:1700]
    # peak_pos = np.argmax(hist2) + 1500
    #
    # coeff = [hist[peak_pos], bin_centres[peak_pos], 50., -0.005]
    # # low = np.digitize(coeff[1] - (5.5 * abs(coeff[2])), bin_centres)
    # # up = np.digitize(coeff[1] + (3.0 * abs(coeff[2])), bin_centres)
    #
    #
    #
    # coeff, var_matrix= curve_fit(gaussErf, bin_centres[peak_pos-fit_range:peak_pos + fit_range], hist[peak_pos-fit_range:peak_pos + fit_range], p0=coeff)
    #
    # mu = coeff[1]
    #
    #
    # energy = energy * peak / mu
    #
    # bins = 2000
    # range = (1000, 3000)
    # hist, bin_edges = np.histogram(energy, bins=bins, range=range)
    # bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    # # peak_pos = find_peak(hist=hist, bin_centres=bin_centres, peakpos=peak, peakfinder='max')
    #
    # hist2 = hist[1500:1700]
    # peak_pos = np.argmax(hist2) + 1500
    #
    # coeff = [hist[peak_pos], bin_centres[peak_pos], 50., -0.005]
    # coeff, var_matrix = curve_fit(gaussErf, bin_centres[peak_pos - fit_range:peak_pos + fit_range], hist[peak_pos - fit_range:peak_pos + fit_range], p0=coeff)
    #
    # mu2 = coeff[1]
    #
    #
    # print 'Energy calibration of ', source
    # print 'peak at: ', bin_centres[peak_pos]
    # print mu, ' ==> ', mu2

    return energy


def plot_residual_real(DNN_one, EXO_one, DNN_two, EXO_two, fOUT, limit=10, mode='real'):
    plt.rc('font', size=12)

    if mode == 'mc':

        res_one = DNN_one - EXO_one
        res_two = DNN_two - EXO_two

        mask_range_one = [np.absolute(res_one) < limit]
        res_one = res_one[mask_range_one]

        mask_range_two = [np.absolute(res_two) < limit]
        res_two = res_two[mask_range_two]

        bins = 200

        color = [(0.6, 0.31, 0.64), (0.3, 0.69, 0.29)]


        plt.hist(res_one, bins=bins, range=(-limit, limit), density=True, alpha=0.6,
                 histtype='stepfilled', color=color[0],
                 label='e$^-$lifetime $ =\infty$\n$\mu=%.1f$\n$ \sigma=%.1f$' % (np.mean(res_one), np.std(res_one)))

        plt.hist(res_two, bins=bins, range=(-limit, limit), density=True, alpha=0.6,
                 histtype='stepfilled', color=color[1],
                 label='e$^-$lifetime $ <\infty$\n$\mu=%.1f$\n$ \sigma=%.1f$' % (np.mean(res_two), np.std(res_two)))

        plt.xlabel('Residual energy [keV]')
        plt.ylabel('Probability density')
        plt.legend(loc="best")
        plt.xlim(xmin=-limit, xmax=limit)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()

    else:
        res_one = DNN_one - EXO_one
        res_two = DNN_two - EXO_two

        mask_range_one = [np.absolute(res_one) < limit]
        res_one = res_one[mask_range_one]

        mask_range_two = [np.absolute(res_two) < limit]
        res_two = res_two[mask_range_two]


        bins = 50

        # if np.mean(res_MC) > -0.05 and np.mean(res_MC) < 0.0:
        #     plt.hist(res_MC, bins=bins, range=(-limit, limit), density=True, color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
        #                                          label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('MC:', 0.0, np.std(res_MC)))
        # else:
        #     plt.hist(res_MC, bins=bins, range=(-limit, limit), density=True,
        #                                          color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
        #                                          label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % (
        #                                          'MC:', np.mean(res_MC), np.std(res_MC)))
        color = [(0.6, 0.31, 0.64), (0.3, 0.69, 0.29)]

        # if np.mean(res_real) > -0.05 and np.mean(res_real) < 0.0:
        #     plt.hist([res_one, res_two], bins=bins, range=(-limit, limit), #density=True, color=(0, 0.2, 0.4), alpha=0.5,
        #              histtype='barstacked', color=color,
        #                                          # label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', 0.00, np.std(res_real)))
        #                                          label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Real:', 0.0, np.std(res_real)))
        # else:

        plt.hist(np.concatenate((res_one, res_two)), bins=bins, range=(-limit, limit),
            color=(0, 0.2, 0.4), #alpha=0.5,
                 histtype='stepfilled',# color='black',
                 # label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', np.mean(res_real), np.std(res_real)))
                 label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % (
                 'SS:', np.mean(np.concatenate((res_one, res_two))), np.std(np.concatenate((res_one, res_two)))))


        # plt.hist([res_one, res_two], bins=bins, range=(-limit, limit),
        #              histtype='barstacked', color=color,
        #              label = ['%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('One-wire:', np.mean(res_one), np.std(res_one)),
        #                       '%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Two-wire:', np.mean(res_two), np.std(res_two))])

        if np.mean(res_one) > -0.05 and np.mean(res_one) < 0.0:
            plt.hist(res_one, bins=bins, range=(-limit, limit), histtype='step', color=color[1],
                 label='%s\n$\mu=%.1f$\n$ \sigma=%.2f$' % ('One-wire:', 0.0, np.std(res_one)))
        else:
            plt.hist(res_one, bins=bins, range=(-limit, limit), histtype='step', color=color[1],
                 label='%s\n$\mu=%.1f$\n$ \sigma=%.2f$' % ('One-wire:', np.mean(res_one), np.std(res_one)))



        plt.xlabel('DNN U - EXO U [mm]')
        plt.ylabel('Counts')
        plt.legend(loc="best")
        plt.xlim(xmin=-limit, xmax=limit)
        plt.savefig(fOUT, bbox_inches='tight')
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
