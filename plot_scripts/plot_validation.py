#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
from plot_scripts.plot_traininghistory import *
mpl.use('PDF')
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import os
from sys import path
path.append('/home/hpc/capm/mppi053h/UVWireRecon')
from math import atan2,degrees
from plot_traininghistory import *
import matplotlib.colors as colors

# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def on_epoch_end_plots(self, folderOUT, epoch, data, var_targets):
    # print data.keys()
    # plot_traininghistory(self, folderOUT)



    if var_targets == 'energy':

        # plot_scatter(np.abs(data['Y_TRUE'][:, 0] - data['Y_PRED'][:, 0]), data['Y_PRED'][:, 1], '|True Energy - DNN Energy| [keV]', 'Sigma [keV]', folderOUT + 'prediction_sigma_' + str(epoch) + '.png')
        # plot_scatter(data['Y_TRUE'][:,0], data['Y_PRED'][:, 0], 'True Energy [keV]', 'DNN Energy [keV]', folderOUT+'prediction_energy_' + str(epoch)+'.png')

        plot_scatter(denormalize(data['Y_TRUE'][:, 0], 'energy'), denormalize(data['Y_PRED'][:, 0], 'energy'), 'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy_' + str(epoch) + '.png')



    if var_targets == 'U':
        plot_scatter(denormalize(data['Y_TRUE'][:, 0], 'U'), denormalize(data['Y_PRED'][:, 0], 'U'), 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U_' + str(epoch) + '.png')
    if var_targets == 'V':
        plot_scatter(denormalize(data['Y_TRUE'][:, 0], 'V'), denormalize(data['Y_PRED'][:, 0], 'V'), 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V_' + str(epoch) + '.png')
    if var_targets == 'Z':
        plot_scatter(denormalize(data['Y_TRUE'][:, 0], 'Z'), denormalize(data['Y_PRED'][:, 0], 'Z'), 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z_' + str(epoch) + '.png')

    # elif var_targets == 'energy_and_UV_position':
    #     plot_scatter(data['Y_TRUE'][:, 0], data['Y_PRED'][:, 0], 'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy_' + str(epoch) + '.png')
    #     plot_scatter(data['Y_TRUE'][:, 2], data['Y_PRED'][:, 2], 'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy3_' + str(epoch) + '.png')
        # plot_scatter(data['Y_TRUE'][:, 1], data['Y_PRED'][:, 1], 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U_'+str(epoch)+'.png')
        # plot_scatter(data['Y_TRUE'][:, 2], data['Y_PRED'][:, 2], 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V_'+str(epoch)+'.png')
        # plot_scatter(data['Y_TRUE'][:, 3], data['Y_PRED'][:, 3], 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z_' + str(epoch) + '.png')

    # for normalized Output values
    # if var_targets == 'energy_and_UV_position':
        # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 0], 'energy'), denormalize(data['Y_PRED'][:, 0, 0], 'energy'), 'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy_' + str(epoch) + '.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 1], 'U'), denormalize(data['Y_PRED'][:, 0, 1], 'U'), 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U_'+str(epoch)+'.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 2], 'V'), denormalize(data['Y_PRED'][:, 0, 2], 'V'), 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V_'+str(epoch)+'.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 3], 'Z'), denormalize(data['Y_PRED'][:, 0, 3], 'Z'), 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z_' + str(epoch) + '.png')
        # #
        # #
        # # # plot_scatter(np.sqrt(4 / 3 * (data['Y_TRUE'][:, 1] * data['Y_TRUE'][:, 1] + data['Y_TRUE'][:, 2] * data['Y_TRUE'][:, 2] - data['Y_TRUE'][:, 1] * data['Y_TRUE'][:, 2])),
        # # #              np.sqrt(4 / 3 * (data['Y_PRED'][:, 1] * data['Y_PRED'][:, 1] + data['Y_PRED'][:, 2] * data['Y_PRED'][:, 2] - data['Y_PRED'][:, 1] * data['Y_PRED'][:, 2])),
        # # #              'True R [% R_max]', 'DNN R [% R_max]', folderOUT + 'prediction_R_' + str(epoch) + '.png')
        # #
        # #
        # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 0], 'energy'), denormalize(data['Y_PRED'][:, 2, 0], 'energy'), 'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy3_' + str(epoch) + '.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 1], 'U'), denormalize(data['Y_PRED'][:, 2, 1], 'U'), 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U3_' + str(epoch) + '.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 2], 'V'), denormalize(data['Y_PRED'][:, 2, 2], 'V'), 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V3_' + str(epoch) + '.png')
        # # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 3], 'Z'), denormalize(data['Y_PRED'][:, 2, 3], 'Z'), 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z3_' + str(epoch) + '.png')





    #   MS Classification and regression
    if var_targets == 'energy_and_UV_position':


        # print data.keys()
        # print data['Y_TRUE_1'].shape
        # print data['Y_TRUE_2'].shape
        # # ['Y_TRUE'].keys()
        # print data['Y_PRED_1'].shape
        # print data['Y_PRED_2'].shape
        #
        # print data['Y_PRED_1'][0, :]
        # print data['Y_TRUE_1'][0, :]
        # print np.argmax(data['Y_PRED_1'], axis=1)




        plot_scatter(denormalize(data['Y_TRUE_2'][:, 0, 0], 'energy'), denormalize(data['Y_PRED_2'][:, 0, 0], 'energy'),
                     'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy_' + str(epoch) + '.png')

        plot_scatter(denormalize(data['Y_TRUE_2'][:, 2, 0], 'energy'), denormalize(data['Y_PRED_2'][:, 2, 0], 'energy'),
                     'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy3_' + str(epoch) + '.png')
        # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 1], 'U'), denormalize(data['Y_PRED'][:, 0, 1], 'U'), 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U_'+str(epoch)+'.png')
        # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 2], 'V'), denormalize(data['Y_PRED'][:, 0, 2], 'V'), 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V_'+str(epoch)+'.png')
        # plot_scatter(denormalize(data['Y_TRUE'][:, 0, 3], 'Z'), denormalize(data['Y_PRED'][:, 0, 3], 'Z'), 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z_' + str(epoch) + '.png')
        #
        #

        # plot_scatter(denormalize(data['Y_TRUE_2'][:, 2, 0], 'energy'), denormalize(data['Y_PRED_2'][:, 2, 0], 'energy'),
        #              'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_energy3_' + str(epoch) + '.png')



        plot_scatter_MS_CR(data['Y_TRUE_1'], denormalize(data['Y_TRUE_2'][:, 0, 0], 'energy'),
                           np.rint(data['Y_PRED_1']), denormalize(data['Y_PRED_2'][:, 0, 0], 'energy'), 1,
                           'True Energy [keV]', 'DNN Energy [keV]',
                           folderOUT + 'prediction_multi_energy1_' + str(epoch) + '.png')
        plot_scatter_MS_CR(data['Y_TRUE_1'], denormalize(data['Y_TRUE_2'][:, 1, 0], 'energy'),
                           np.rint(data['Y_PRED_1']), denormalize(data['Y_PRED_2'][:, 1, 0], 'energy'), 2,
                           'True Energy [keV]', 'DNN Energy [keV]',
                           folderOUT + 'prediction_multi_energy2_' + str(epoch) + '.png')

        plot_scatter_MS_CR(data['Y_TRUE_1'], denormalize(data['Y_TRUE_2'][:, 2, 0], 'energy'),
                           np.rint(data['Y_PRED_1']), denormalize(data['Y_PRED_2'][:, 2, 0], 'energy'), 3,
                           'True Energy [keV]', 'DNN Energy [keV]', folderOUT + 'prediction_multi_energy3_' + str(epoch) + '.png')

        plot_scatter_MS_CR(data['Y_TRUE_1'], denormalize(data['Y_TRUE_2'][:, 3, 0], 'energy'),
                           np.rint(data['Y_PRED_1']), denormalize(data['Y_PRED_2'][:, 3, 0], 'energy'), 4,
                           'True Energy [keV]', 'DNN Energy [keV]',
                           folderOUT + 'prediction_multi_energy4_' + str(epoch) + '.png')
        plot_scatter_MS_CR(data['Y_TRUE_1'], denormalize(data['Y_TRUE_2'][:, 4, 0], 'energy'),
                           np.rint(data['Y_PRED_1']), denormalize(data['Y_PRED_2'][:, 4, 0], 'energy'), 5,
                           'True Energy [keV]', 'DNN Energy [keV]',
                           folderOUT + 'prediction_multi_energy5_' + str(epoch) + '.png')


        plot_scatter(data['Y_TRUE_1'], data['Y_PRED_1'],
                     'True Number Cluster', 'DNN Number Cluster [keV]', folderOUT + 'prediction_numCluster_' + str(epoch) + '.png', alpha=0.3)


        # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 1], 'U'), denormalize(data['Y_PRED'][:, 2, 1], 'U'), 'True U [mm]', 'DNN U [mm]', folderOUT + 'prediction_U3_' + str(epoch) + '.png')
        # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 2], 'V'), denormalize(data['Y_PRED'][:, 2, 2], 'V'), 'True V [mm]', 'DNN V [mm]', folderOUT + 'prediction_V3_' + str(epoch) + '.png')
        # plot_scatter(denormalize(data['Y_TRUE'][:, 2, 3], 'Z'), denormalize(data['Y_PRED'][:, 2, 3], 'Z'), 'True Z [mm]', 'DNN Z [mm]', folderOUT + 'prediction_Z3_' + str(epoch) + '.png')



    return


def validation_mc_plots(folderOUT, data, epoch, sources, position, target):

    dir_spectrum = "/2prediction-spectrum/"
    dir_scatter = "/3prediction-scatter/"
    dir_residual = "/4residual-histo/"
    for dir in [dir_spectrum, dir_scatter, dir_residual]:
        os.system("mkdir -m 770 -p %s " % (folderOUT + dir))
    # os.system("mkdir -m 770 -p %s " % (folderOUT + "/5residual-mean/"      ))
    # os.system("mkdir -m 770 -p %s " % (folderOUT + "/7residual-sigma/"))

    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    name_True = 'True'
    peakpos = 2614.5

    if target == 'energy_and_UV_position':
        a, b, c = 1, 2, 3
    if target == 'position':
        a, b, c = 0, 1, 2
    if target == 'Z':
        c = 0
    if target == 'U':
        a = 0
    if target == 'V':
        b = 0



    print data.keys()
    print data['EVENT_INFO'].keys()


    # mask1 = data['EVENT_INFO']['CCCorrectedEnergy'] > 2609
    # mask2 = data['EVENT_INFO']['CCCorrectedEnergy'] < 2619
    # mask = np.logical_and(mask1, mask2)
    # mask_energy = (data['EVENT_INFO']['MCEnergy'] > 2609) & (data['EVENT_INFO']['MCEnergy'] < 2619)
    # print '<<<<'

    # mask_energy = mask_energy[:,0]
    # mask_energy = np.reshape(mask_energy, (-1))
    # print mask_energy
    # print np.count_nonzero(mask_energy)




    # mask = np.logical_and(data['EVENT_INFO']['CCNumberClusters']==1, data['EVENT_INFO']['CCNumberUWires']==1)

    if target == 'energy' or target == 'energy_and_UV_position':
        data['Y_TRUE'][:, 0] = denormalize(data['Y_TRUE'][:, 0], 'energy')
        data['Y_PRED'][:, 0] = denormalize(data['Y_PRED'][:, 0], 'energy')

        plot_diagonal(x=np.reshape(data['Y_TRUE'][:, 0],(-1)), y=np.reshape(data['Y_PRED'][:, 0],(-1)),
                      xlabel=name_True, ylabel=name_DNN, mode='Energy',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Energy_DNN_' + epoch + '.pdf'))
        plot_diagonal(x=data['Y_TRUE'][:, 0], y=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0], xlabel=name_True,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Energy_EXO_' + epoch + '.pdf'))
        plot_diagonal(x=data['Y_PRED'][:, 0], y=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0], xlabel=name_DNN,
                  ylabel=name_EXO, mode='Energy',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Energy_Both_' + epoch + '.pdf'))
        # plot_spectrum(dCNN=data['Y_PRED'][:, 0], dEXO=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0],
        #           dTrue=data['Y_TRUE'][:, 0],
        #           mode='Energy',
        #           fOUT=(folderOUT + dir_spectrum + sources + '_' + position + '_Energy_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['Y_TRUE'][:, 0], dDNN=data['Y_PRED'][:, 0],
                            dEXO=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0],
                            title='Energy', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=150,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_Energy_' + epoch + '.pdf')

        # plot_hexbin(data['EVENT_INFO']['MCPosX'][:, 0], data['EVENT_INFO']['MCPosY'][:, 0], np.absolute(data['Y_TRUE'][:, 0] - data['Y_PRED'][:, 0]), np.absolute(data['Y_TRUE'][:, 0] - data['EVENT_INFO']['CCCorrectedEnergy'][:, 0]),
        #             'X [mm]', 'Y[mm]',
        #             fOUT=folderOUT + dir_residual + sources + '_' + position + '_Energy_DNN_hexbin_' + epoch + '.pdf')

        # plot_hexbin(data['EVENT_INFO']['MCPosX'][:, 0], data['EVENT_INFO']['MCPosY'][:, 0], np.absolute(data['Y_TRUE'][:, 0] - data['EVENT_INFO']['CCCorrectedEnergy'][:, 0]), 'X [mm]', 'Y[mm]',
        #             fOUT=folderOUT + dir_residual + sources + '_' + position + '_Energy_EXO_hexbin_' + epoch + '.pdf')


    if target == 'position' or target == 'energy_and_UV_position' or target == 'U':

        data['Y_TRUE'][:, a] = denormalize(data['Y_TRUE'][:, a], 'U')
        data['Y_PRED'][:, a] = denormalize(data['Y_PRED'][:, a], 'U')

        plot_diagonal(x=data['Y_TRUE'][:, a], y=data['Y_PRED'][:, a], xlabel=name_True, ylabel=name_DNN, mode='U',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_U_DNN_' + epoch + '.pdf'))
        plot_diagonal(x=data['Y_TRUE'][:, a], y=data['EVENT_INFO']['CCPosU'][:, 0], xlabel=name_True,
                  ylabel=name_EXO, mode='U',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_U_EXO_' + epoch + '.pdf'))

        plot_spectrum(dCNN=data['Y_PRED'][:, a], dEXO=data['EVENT_INFO']['CCPosU'][:, 0], dTrue=data['Y_TRUE'][:, a],
                  mode='U', fOUT=(folderOUT + dir_spectrum + sources + '_' + position + '_U_' + epoch + '.pdf'))

        # plot_residual_histo(dTrue=data['Y_TRUE'][:, a], dDNN=data['Y_PRED'][:, a],
        #                     dEXO=data['EVENT_INFO']['CCPosU'][:, 0],
        #                     title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=5,
        #                     fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_' + epoch + '.pdf')


    # Check 1 wire and other events
        mask_SS = data['EVENT_INFO']['CCIsSS'] == 1

        mask_SS_peak = np.logical_and(mask_SS, data['EVENT_INFO']['MCEnergy'] > 2604)
        mask_SS_peak = np.logical_and(mask_SS_peak, data['EVENT_INFO']['MCEnergy'] < 2624)

        mask_num_U_is_1 = data['EVENT_INFO']['CCNumberUWires'] == 1
        mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, data['EVENT_INFO']['CCIsSS'] == 1)
        mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, data['EVENT_INFO']['MCEnergy'] > 2604)
        mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, data['EVENT_INFO']['MCEnergy'] < 2624)
        # mask_num_U_is_1 = np.logical_and(mask_num_U_is_1, np.reshape(data['Y_TRUE'][:, a] != 1000, (-1,)))


        mask_num_U_is_not_1 = data['EVENT_INFO']['CCNumberUWires'] > 1
        mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, data['EVENT_INFO']['CCIsSS'] == 1)
        mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, data['EVENT_INFO']['MCEnergy'] > 2604)
        mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, data['EVENT_INFO']['MCEnergy'] < 2624)
        # mask_num_U_is_not_1 = np.logical_and(mask_num_U_is_not_1, np.reshape(data['Y_TRUE'][:, a] != 1000, (-1,)))


        plot_boxplot(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1,))[mask_SS_peak], dTrue_masked=np.reshape(data['Y_TRUE'][:, a], (-1,))[mask_num_U_is_1],
                     dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_SS_peak], dEXO_masked=data['EVENT_INFO']['CCPosU'][:, 0][mask_num_U_is_1],
                     dDNN=data['Y_PRED'][:, a][mask_SS_peak], dDNN_masked=data['Y_PRED'][:, a][mask_num_U_is_1],
                     title='Boxplot', name_DNN_masked='DNN onewire', name_EXO_masked='EXO onewire',
                     fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_boxplot_' + epoch + '.pdf')


        #  TODO: sigma checken ob richtig; fuer EXO zu klein
        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1,))[mask_num_U_is_1], dDNN=data['Y_PRED'][:, a][mask_num_U_is_1],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_num_U_is_1], limit=10,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_onewire_' + epoch + '.pdf')

        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1))[mask_num_U_is_not_1], dDNN=data['Y_PRED'][:, a][mask_num_U_is_not_1],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_num_U_is_not_1], limit=10,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_multiwire_' + epoch + '.pdf')

        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1))[mask_SS], dDNN=data['Y_PRED'][:, a][mask_SS],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_SS], limit=10,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_' + epoch + '.pdf')

        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1,))[mask_num_U_is_1], dDNN=data['Y_PRED'][:, a][mask_num_U_is_1],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_num_U_is_1], limit=100,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_onewire_100_' + epoch + '.pdf')

        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1))[mask_num_U_is_not_1], dDNN=data['Y_PRED'][:, a][mask_num_U_is_not_1],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_num_U_is_not_1], limit=100,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_multiwire_100_' + epoch + '.pdf')

        plot_residual_histo(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1))[mask_SS], dDNN=data['Y_PRED'][:, a][mask_SS],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_SS], limit=100,
                            title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_100_' + epoch + '.pdf')



        plot_residual_correlation(dTrue=np.reshape(data['Y_TRUE'][:, a], (-1,))[mask_SS], dDNN=data['Y_PRED'][:, a][mask_SS],
                            dEXO=data['EVENT_INFO']['CCPosU'][:, 0][mask_SS], limit=6,
                            mode='U',
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_SS_correlation_' + epoch + '.pdf')

        #  Plot of residual as funciton of max of neighbouring channel:
        limit = 10
        # channel_1 = np.zeros((3, data['Y_TRUE'].shape[0]))
        # channel_2 = np.zeros((2, data['Y_TRUE'].shape[0]))
        # channel_3 = np.zeros((2, data['Y_TRUE'].shape[0]))

        # print '>>>', data['Y_TRUE'].shape
        # print channel_1.shape
        # values =[]

        # mask_channel_1 = np.zeros((data['Y_TRUE'].shape[0],))

        mask_channel_1 = data['EVENT_INFO']['CCNumberUWires'] == 1
        mask_channel_1 = np.logical_and(mask_channel_1, data['EVENT_INFO']['CCIsSS'] == 1)
        mask_channel_1 = np.logical_and(mask_channel_1, data['Y_TRUE'][:, a] - data['Y_PRED'][:, a] < limit)
        mask_channel_1 = np.logical_and(mask_channel_1, data['Y_TRUE'][:, a] - data['EVENT_INFO']['CCPosU'][:, a] < limit)
        mask_channel_1 = np.logical_and(mask_channel_1, mask_SS_peak)



        # for eventnumber in range(data['Y_TRUE'].shape[0]):
        #     if data['EVENT_INFO']['CCNumberUWires'][eventnumber] == 1:
        #         if data['EVENT_INFO']['CCIsSS'][eventnumber] == 1:
        #             max_channel_1_TPC1 = np.amax(data['wf'][0, eventnumber, :, :, :])
        #             max_channel_1_TPC2 = np.amax(data['wf'][2, eventnumber, :, :, :])
        #             tpc = 0
        #
        #             if max_channel_1_TPC1 > max_channel_1_TPC2:
        #                 max_channel = max_channel_1_TPC1
        #                 arg_max_1 = np.unravel_index(np.argmax(data['wf'][0, eventnumber, :, :, :]), data['wf'][0, eventnumber, :, :, :].shape)
        #                 tpc = 0
        #             else:
        #                 max_channel = max_channel_1_TPC2
        #                 arg_max_1 = np.unravel_index(np.argmax(data['wf'][2, eventnumber, :, :, :]), data['wf'][2, eventnumber, :, :, :].shape)
        #                 tpc = 2
        #
        #             i,j,k = arg_max_1
        #
        #             try:
        #                 channel_1plus = np.amax(data['wf'][tpc, eventnumber, i-25:i+25, j+1, k])
        #             except:
        #                 channel_1plus = 0
        #
        #             try:
        #                 channel_1minus = np.amax(data['wf'][tpc, eventnumber, i-25:i+25, j-1, k])
        #             except:
        #                 channel_1minus = 0
        #
        #             channel_1[0, eventnumber] = max([channel_1plus, channel_1minus])
        #
        #             # if channel_1[0, eventnumber] == 0:
        #                 # print channel_1plus, '\t', channel_1minus
        #
        #
        #             channel_1[1, eventnumber] = data['Y_TRUE'][eventnumber, a] - data['Y_PRED'][eventnumber, a]
        #             channel_1[2, eventnumber] = data['Y_TRUE'][eventnumber, a] - data['EVENT_INFO']['CCPosU'][eventnumber, a]
        #
        #             if np.absolute(channel_1[1, eventnumber]) < limit and channel_1[2, eventnumber] < limit:
        #                 mask_channel_1[eventnumber] = 1
        #             else:
        #                 print '>>>'
        #                 print 'true: \t', data['Y_TRUE'][eventnumber, a]
        #                 print 'DNN: \t', data['Y_PRED'][eventnumber, a]
        #                 print 'EXO: \t', data['EVENT_INFO']['CCPosU'][eventnumber, a]
        #                 print 'PCD: \t', data['EVENT_INFO']['PCDPosU'][eventnumber, :]
        #                 print 'PCD: \t', data['EVENT_INFO']['PCDEnergy'][eventnumber, :]
        #
        #             # values.append((channel_1[0, eventnumber], channel_1[1, eventnumber], channel_1[2, eventnumber]))
        #
        # # print mask_channel_1
        # mask_channel_1 = mask_channel_1 != 0

        channel_1 = data['EVENT_INFO']['wf_max_neighbour'][:][mask_channel_1], \
                    data['Y_TRUE'][:, a][mask_channel_1] - data['Y_PRED'][:, a][mask_channel_1],\
                    data['Y_TRUE'][:, a][mask_channel_1] - data['EVENT_INFO']['CCPosU'][:, a][mask_channel_1]


        # print mask_channel_1
        # dtype = [('max_channel', float), ('res_dnn', float), ('res_exo', float)]
        # channel_1_sorted = np.array(values, dtype=dtype)
        channel_1_sorted = zip(*sorted(zip(channel_1[0], channel_1[1], channel_1[2])))
        # print channel_1_sorted

        # channel_1_sorted = channel_1_sorted.view(np.float64).reshape(channel_1_sorted.shape + (-1,))

        # plt.scatter(channel_1[0][mask_channel_1], channel_1[1][mask_channel_1], alpha=0.25, marker='.')
        # plt.scatter(channel_1[0][mask_channel_1], channel_1[2][mask_channel_1], alpha=0.25, color='red', marker='.')
        # plt.plot(running_mean(channel_1_sorted[0], 50), running_mean(np.absolute(channel_1_sorted[1]), 50))
        # plt.plot(running_mean(channel_1_sorted[0], 50), running_mean(np.absolute(channel_1_sorted[2]), 50), color='red')
        # plt.plot(diag, diag, 'k--')
        # plt.legend(loc="best")
        # plt.ylabel('|Residual|')
        # plt.xlabel('maximum of neighbouring channel')

        # plt.gca().set_yscale('log')
        # plt.xlim(xmin=0, xmax=3300)

        # plt.ylim(ymin=-limit, ymax=limit)



        from matplotlib.ticker import NullFormatter

        # color = 'viridis'

        # cm = plt.cm.get_cmap(color)

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

        axLegend.text(0.1, 0.6, 'DNN: \n$\mu=%.1f,$ $\sigma=%.1f$' % (np.mean(channel_1[1]), np.std(channel_1[1])), color='navy', alpha=0.5)
        axLegend.text(0.1, 0.2, 'EXO: \n$\mu=%.1f,$ $\sigma=%.1f$' % (np.mean(channel_1[2]), np.std(channel_1[2])), color='red', alpha=0.5)

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
        axHistx.plot(running_mean(channel_1_sorted[0], 100), running_mean(np.absolute(channel_1_sorted[1]), 100), alpha=0.5, color='navy')
        axHistx.plot(running_mean(channel_1_sorted[0], 100), running_mean(np.absolute(channel_1_sorted[2]), 100), color='red', alpha=0.5)



        # col_y = n_y / max(n_y)
        # bin_centers_y = 0.5 * (bins_y[:-1] + bins_y[1:])
        # for c, p in zip(col_y, patches_y):
        #     plt.setp(p, 'facecolor', cm(c))
        #     plt.setp(p, 'edgecolor', cm(c), alpha=1)
        # axHisty.plot(n_y, bin_centers_y, color=cm(max(n_y)))

        plt.savefig(folderOUT + dir_residual + sources + '_' + position + '_U_wirecheck_' + epoch + '.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()




    if target == 'position' or target == 'energy_and_UV_position' or target == 'V':
        data['Y_TRUE'][:, b] = denormalize(data['Y_TRUE'][:, b], 'V')
        data['Y_PRED'][:, b] = denormalize(data['Y_PRED'][:, b], 'V')

        plot_diagonal(x=data['Y_TRUE'][:, b], y=data['Y_PRED'][:, b], xlabel=name_True, ylabel=name_DNN, mode='V',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_V_DNN_' + epoch + '.pdf'))
        plot_diagonal(x=data['Y_TRUE'][:, b], y=data['EVENT_INFO']['CCPosV'][:, 0], xlabel=name_True,
                  ylabel=name_EXO, mode='V',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_V_EXO_' + epoch + '.pdf'))

        plot_spectrum(dCNN=data['Y_PRED'][:, b], dEXO=data['EVENT_INFO']['CCPosV'][:, 0], dTrue=data['Y_TRUE'][:, b],
                      mode='V', fOUT=(folderOUT + dir_spectrum + sources + '_' + position + '_V_' + epoch + '.pdf'))

        plot_residual_histo(dTrue=data['Y_TRUE'][:, b], dDNN=data['Y_PRED'][:, b],
                            dEXO=data['EVENT_INFO']['CCPosV'][:, 0],
                            title='V', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=5,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_V_' + epoch + '.pdf')

        plot_residual_correlation(dTrue=data['Y_TRUE'][:, b], dDNN=data['Y_PRED'][:, b],
                            dEXO=data['EVENT_INFO']['CCPosV'][:, 0],
                            title='V', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=5,
                            fOUT=folderOUT + dir_residual + sources + '_' + position + '_V_correlation_' + epoch + '.pdf')

        #  HEXBIN of energy resolution: TODO: not ready yet
        # fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
        # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        # ax = axs[0]
        # hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')
        # ax.axis([xmin, xmax, ymin, ymax])
        # ax.set_title("Hexagon binning")
        # cb = fig.colorbar(hb, ax=ax)
        # cb.set_label('counts')
        #
        # fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))
        # fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)
        # ax = axs[0]
        # hb = ax.hexbin(x, y, gridsize=50, cmap='inferno')
        # ax.axis([xmin, xmax, ymin, ymax])
        # ax.set_title("Hexagon binning")
        # cb = fig.colorbar(hb, ax=ax)
        # cb.set_label('counts')
        # plt.savefig(folderOUT + dir_scatter + sources + '_' + position + '_hexbin_' + epoch + '.pdf', bbox_inches='tight')
        # plt.clf()
        # plt.close()


        # plot_diagonal(x=np.sqrt(data['Y_TRUE'][:, a] * data['Y_TRUE'][:, a] + data['Y_TRUE'][:, 2] * data['Y_TRUE'][:, 2]),
        #           y=np.sqrt(4 / 3 * (data['Y_PRED'][:, a] * data['Y_PRED'][:, a] + data['Y_PRED'][:, b] * data['Y_PRED'][:, b] - data['Y_PRED'][:, a] * data['Y_PRED'][:, b])),
        #           xlabel=name_EXO, ylabel=name_DNN, mode='R',
        #           fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_R_DNN_' + epoch + '.pdf'))
        # plot_diagonal(x=np.sqrt(data['Y_TRUE'][:, a] * data['Y_TRUE'][:, a] + data['Y_TRUE'][:, 2] * data['Y_TRUE'][:, 2]),
        #           y=np.sqrt(4 / 3 * (data['EVENT_INFO']['CCPosU'][:, 0] * data['EVENT_INFO']['CCPosU'][:, 0] + data['EVENT_INFO']['CCPosV'][:, 0] * data['EVENT_INFO']['CCPosV'][:, 0] - data['EVENT_INFO']['CCPosU'][:, 0] * data['EVENT_INFO']['CCPosV'][:, 0])),
        #           xlabel=name_EXO, ylabel=name_DNN, mode='R',
        #           fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_R_EXO_' + epoch + '.pdf'))





        #
        # plot_residual_histo(dTrue=np.reshape(denormalize(data['Y_TRUE'][:, a], 'U'), (-1,))[mask_num_V_is_not_1], dDNN=np.reshape(denormalize(data['Y_PRED'][:, a], 'U'), (-1,))[mask_num_V_is_not_1],
        #                     dEXO=np.reshape(data['EVENT_INFO']['CCPosU'][:, 0], (-1,))[mask_num_V_is_not_1],
        #                     title='U', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO, limit=1000,
        #                     fOUT=folderOUT + dir_residual + sources + '_' + position + '_U_not1wire_' + epoch + '.pdf')



    if target == 'Z' or target =='position' or target == 'energy_and_UV_position':
        data['Y_PRED'][:, c] = denormalize(data['Y_PRED'][:, c], 'Z')
        data['Y_TRUE'][:, c] = denormalize(data['Y_TRUE'][:, c], 'Z')


        # plot_diagonal(x=data['Y_TRUE'][:, c], y=data['Y_PRED'][:, c], xlabel=name_True, ylabel=name_DNN, mode='Z',
        #           fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Z_DNN_' + epoch + '.pdf'))
        # plot_diagonal(x=data['Y_TRUE'][:, c], y=data['EVENT_INFO']['CCPosZ'][:, 0], xlabel=name_True,
        #           ylabel=name_EXO, mode='Z',
        #           fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Z_EXO_' + epoch + '.pdf'))
        #
        # plot_spectrum(dCNN=data['Y_PRED'][:, c],
        #               dEXO=data['EVENT_INFO']['CCPosZ'][:, 0],
        #               dTrue=data['Y_TRUE'][:, c], mode='Z',
        #               fOUT=(folderOUT + dir_spectrum + sources + '_' + position + '_Z_' + epoch + '.pdf'))
        #
        #
        # plot_residual_histo(dTrue=data['Y_TRUE'][:, c], dDNN=data['Y_PRED'][:, c],
        #             dEXO=data['EVENT_INFO']['CCPosZ'][:, 0], limit=5,
        #             title='Z', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
        #             fOUT=folderOUT + dir_residual + sources + '_' + position + '_Z_' + epoch + '.pdf')



        # print data['Y_TRUE'].shape
        # print data['Y_PRED'].shape
        # print data['EVENT_INFO']['CCPosZ'].shape

        plot_diagonal(x=data['Y_TRUE'][:, c], y=data['Y_PRED'][:, c], xlabel=name_True, ylabel=name_DNN, mode='Z',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Z_DNN_' + epoch + '.pdf'))
        plot_diagonal(x=data['Y_TRUE'][:, c], y=data['EVENT_INFO']['CCPosZ'][:, 0], xlabel=name_True,
                  ylabel=name_EXO, mode='Z',
                  fOUT=(folderOUT + dir_scatter + sources + '_' + position + '_Z_EXO_' + epoch + '.pdf'))

        plot_spectrum(dCNN=data['Y_PRED'][:, c],
                      dEXO=data['EVENT_INFO']['CCPosZ'][:, 0],
                      dTrue=data['Y_TRUE'][:, c], mode='Z',
                      fOUT=(folderOUT + dir_spectrum + sources + '_' + position + '_Z_' + epoch + '.pdf'))


        plot_residual_histo(dTrue=data['Y_TRUE'][:, c], dDNN=data['Y_PRED'][:, c],
                    dEXO=data['EVENT_INFO']['CCPosZ'][:, 0], limit=5,
                    # dEXO=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]), limit=5,
                    title='Z', name_True=name_True, name_DNN=name_DNN, name_EXO=name_EXO,
                    fOUT=folderOUT + dir_residual + sources + '_' + position + '_Z_' + epoch + '.pdf')

    # plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
    #                     fOUT=folderOUT + '/3prediction-scatter/prediction_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
    # plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
    #                     fOUT=folderOUT + '/3prediction-scatter/prediction_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
    # plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
    #                      fOUT=folderOUT + '/5residual-scatter/residual_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
    # plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
    #                      fOUT=folderOUT + '/5residual-scatter/residual_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')


    return

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)



def validation_data_plots(folderOUT, data, epoch, sources, position, var_targets):
    name_DNN = 'DNN'
    name_EXO = 'EXO-Recon'
    peakpos = 2614.5


    if var_targets == 'energy':
        data['Y_PRED'][:, 0] = denormalize(data['Y_PRED'][:, 0], 'energy')

        plot_diagonal(x=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0], y=data['Y_PRED'][:, 0], xlabel=name_EXO,
                      ylabel=name_DNN, mode='Energy',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_Energy_no_correction_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0], dDNN=data['Y_PRED'][:, 0], dEXO=None,
                            title='Energy', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None, limit=100,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_Energy_no_correction_' + epoch + '.pdf')
        plot_spectrum(dCNN=data['Y_PRED'][:, 0], dEXO=data['EVENT_INFO']['CCCorrectedEnergy'][:, 0], dTrue=None,
                      mode='Energy',
                      fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_Energy_no_correction_' + epoch + '.pdf'))

        plot_diagonal(x=data['EVENT_INFO']['CCPurityCorrectedEnergy'][:, 0], y=data['Y_PRED'][:, 0], xlabel=name_EXO,
                      ylabel=name_DNN, mode='Energy',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_Energy_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCPurityCorrectedEnergy'][:, 0], dDNN=data['Y_PRED'][:, 0], dEXO=None,
                            title='Energy', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None, limit=100,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_Energy_' + epoch + '.pdf')
        plot_spectrum(dCNN=data['Y_PRED'][:, 0], dEXO=data['EVENT_INFO']['CCPurityCorrectedEnergy'][:, 0], dTrue=None,
                      mode='Energy',
                      fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_Energy_' + epoch + '.pdf'))


    if var_targets == 'U':
        data['Y_PRED'][:, 0] = denormalize(data['Y_PRED'][:, 0], 'U')

        plot_diagonal(x=data['EVENT_INFO']['CCPosU'][:, 0], y=data['Y_PRED'][:, 0], xlabel=name_EXO, ylabel=name_DNN, mode='U',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_U_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCPosU'][:, 0], dDNN=data['Y_PRED'][:, 0], dEXO=None,
                            title='U', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_U_' + epoch + '.pdf')
        plot_spectrum(dCNN=data['Y_PRED'][:, 0], dEXO=data['EVENT_INFO']['CCPosU'][:, 0], dTrue=None,
                      mode='U', fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_U_' + epoch + '.pdf'))



    if var_targets == 'V':
        data['Y_PRED'][:, 0] = denormalize(data['Y_PRED'][:, 0], 'V')

        plot_diagonal(x=data['EVENT_INFO']['CCPosV'][:, 0], y=data['Y_PRED'][:, 0], xlabel=name_EXO, ylabel=name_DNN, mode='V',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_V_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCPosV'][:, 0], dDNN=data['Y_PRED'][:, 0], dEXO=None,
                            title='V', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_V_' + epoch + '.pdf')
        plot_spectrum(dCNN=data['Y_PRED'][:, 0], dEXO=data['EVENT_INFO']['CCPosV'][:, 0], dTrue=None,
                      mode='V', fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_V_' + epoch + '.pdf'))

    if var_targets == 'Z':
        data['Y_PRED'][:, 0] = denormalize(data['Y_PRED'][:, 0], 'Z')

        plot_diagonal(x=data['EVENT_INFO']['CCPosZ'][:, 0], y=data['Y_PRED'][:, 0],
                      xlabel=name_EXO, ylabel=name_DNN, mode='Z',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_Z_' + epoch + '.pdf'))
        plot_spectrum(dCNN=data['Y_PRED'][:, 0],
                      dEXO=data['EVENT_INFO']['CCPosZ'][:, 0],
                      dTrue=None, mode='Z',
                      fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_Z_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCPosZ'][:, 0], dDNN=data['Y_PRED'][:, 0],
                            dEXO=None,
                            title='Z', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None, limit=100,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_Z_' + epoch + '.pdf')


        # plot_diagonal(x=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]), y=fromTimeToZ(data['Y_PRED'][:, 0]),
        #               xlabel=name_EXO, ylabel=name_DNN, mode='Z',
        #               fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_Z_' + epoch + '.pdf'))
        # plot_spectrum(dCNN=fromTimeToZ(data['Y_PRED'][:, 0]),
        #               dEXO=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]),
        #               dTrue=None, mode='Z',
        #               fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_Z_' + epoch + '.pdf'))
        # plot_residual_histo(dTrue=fromTimeToZ(data['EVENT_INFO']['CCCollectionTime'][:, 0]), dDNN=fromTimeToZ(data['Y_PRED'][:, 0]),
        #                     dEXO=None,
        #                     title='Z', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None,
        #                     fOUT=folderOUT + 'residual_' + sources + '_' + position + '_Z_' + epoch + '.pdf')



    if var_targets == 'time':
        plot_diagonal(x=data['EVENT_INFO']['CCCollectionTime'][:, 0], y=data['Y_PRED'][:, 3], xlabel=name_EXO, ylabel=name_DNN, mode='Time',
                      fOUT=(folderOUT + 'prediction_' + sources + '_' + position + '_Time_' + epoch + '.pdf'))
        plot_spectrum(dCNN=data['Y_PRED'][:, 3], dEXO=data['EVENT_INFO']['CCCollectionTime'][:, 0], dTrue=None,
                      mode='Time', fOUT=(folderOUT + 'spectrum_' + sources + '_' + position + '_Time_' + epoch + '.pdf'))
        plot_residual_histo(dTrue=data['EVENT_INFO']['CCCollectionTime'][:, 0], dDNN=data['Y_PRED'][:, 3], dEXO=None,
                            title='Time', name_True=name_EXO, name_DNN=name_DNN, name_EXO=None,
                            fOUT=folderOUT + 'residual_' + sources + '_' + position + '_Time_' + epoch + '.pdf')


    # plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
    #                     fOUT=folderOUT + '/3prediction-scatter/prediction_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
    # plot_scatter_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
    #                     fOUT=folderOUT + '/3prediction-scatter/prediction_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')
    # plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_EXO'][Multi], name_x=name_True, name_y=name_EXO,
    #                      fOUT=folderOUT + '/5residual-scatter/residual_' + sources + '_' + position + '_Standard_' + Multi + '_' + epoch + '.pdf')
    # plot_residual_hist2d(E_x=data['E_True'][Multi], E_y=data['E_CNN'][Multi], name_x=name_True, name_y=name_CNN,
    #                      fOUT=folderOUT + '/5residual-scatter/residual_' + sources + '_' + position + '_ConvNN_' + Multi + '_' + epoch + '.pdf')


# ----------------------------------------------------------
# Plots
# ----------------------------------------------------------
def doCalibration(data_True, data_Recon):
    m, b = np.polyfit(data_True, data_Recon, 1)
    return m, b

def fromTimeToZ(data):
    return -1.71 * data + 1949.89
# # New:
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

# # #  Old:
# def denormalize(data, mode):
#     data /= 100
#     if mode == 'energy':
#         data_denorm = (data / 2          * 2950) + 550
#     elif mode == 'U' or mode == 'V':
#         data_denorm = (data / 1          * 340) - 170
#     elif mode == 'time':
#         data_denorm = (data / 1          * 110) + 1030
#     elif mode == 'Z':
#         data_denorm = (data / 2          * 190)
#     return data_denorm


def fit_spectrum(data, peakpos, fit, name, color, isMC, peakfinder='max', zorder=3):
    hist, bin_edges = np.histogram(data, bins=1200, range=(0, 12000), density=False)
    norm_factor = float(len(data))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    if name != 'MC':
        if fit:
            from scipy.optimize import curve_fit
            peak = find_peak(hist=hist, bin_centres=bin_centres, peakpos=peakpos, peakfinder=peakfinder)
            coeff = [hist[peak], bin_centres[peak], 50., -0.005]
            for i in range(5):
                try:
                    if isMC==True: #fit range for MC spectra
                        low = np.digitize(coeff[1] - (5.5 * abs(coeff[2])), bin_centres)
                        up = np.digitize(coeff[1] + (3.0 * abs(coeff[2])), bin_centres)
                    else: #fit range from RotationAngle script #original was 3. for lower bound
                        low = np.digitize(coeff[1] - (3.5 * abs(coeff[2])), bin_centres)
                        up = np.digitize(coeff[1] + (2.5 * abs(coeff[2])), bin_centres)
                    coeff, var_matrix = curve_fit(gaussErf, bin_centres[low:up], hist[low:up], p0=coeff)
                    coeff_err = np.sqrt(np.absolute(np.diag(var_matrix)))
                except:
                    print name, 'fit did not work\t', i
                    coeff, coeff_err = [hist[peak], bin_centres[peak], 50.0*(i+1), -0.005], [0.0] * len(coeff)
            delE = abs(coeff[2]) / coeff[1] * 100.0
            delE_err = delE * np.sqrt((coeff_err[1] / coeff[1]) ** 2 + (coeff_err[2] / coeff[2]) ** 2)

            # plt.plot(bin_centres[low:up], gauss_zero(bin_centres[low:up], *coeff[:3]) / norm_factor, lw=1, ls='--', color=color)
            # plt.plot(bin_centres[low:up], erf(bin_centres[low:up], *coeff[1:]) / norm_factor, lw=1, ls='--', color=color)
            # plt.plot(bin_centres[low:up], gaussErf(bin_centres[low:up], *coeff) / norm_factor, lw=1 , ls='--', color=color)

            plt.plot(bin_centres, gauss_zero(bin_centres, *coeff[:3]) / norm_factor, lw=1, ls='--', color=color, zorder=zorder)
            # plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s: $%.4f \pm %.4f$ %% $(\sigma)$' % (name, delE, delE_err), zorder=zorder+1)
            plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s' % (name), zorder=zorder+1)
            # plt.axvline(x=2614, c='k', lw=2)
            return (coeff[1], coeff_err[1]), (abs(coeff[2]), coeff_err[2])
        else:
            plt.step(bin_centres, hist / norm_factor, where='mid', color=color, label='%s' % (name), zorder=zorder)
            return (-1000., 1000.), (-1000., -1000.)
    else:
        plt.plot(bin_centres, hist / norm_factor, label=name, color=color, lw=0.5, zorder=2)
        plt.fill_between(bin_centres, 0.0, hist / norm_factor, facecolor='black', alpha=0.3, interpolate=True, zorder=1)
        return


def find_peak(hist, bin_centres, peakpos, peakfinder='max'):
    peak = hist[hist.size/2]
    if peakfinder == 'max':
        peak = np.argmax(
            hist[np.digitize(peakpos - 300, bin_centres):np.digitize(peakpos + 300, bin_centres)]) + np.digitize(
            peakpos - 300, bin_centres)
    elif peakfinder == 'fromright':
        length = hist.size
        inter = 20
        # from math import  sqrt
        for i in range(len(hist) - inter, inter, -1):
            if hist[i] < 50: continue
            if np.argmax(hist[ np.max( [i - inter, 0] ) : np.min( [i + inter, length] ) ]) == inter: peak = i ; break
            # if hist[i + 1] <= 0: continue
            # sigma = sqrt(hist[i] + hist[i + 1])
            # if abs((hist[i + 1] - hist[i]) / sigma) >= 5.0:
            #     peak = i + 1
            #     break
    return peak


def calibrate_spectrum(data, name, peakpos, fOUT, isMC, peakfinder):
    import matplotlib.backends.backend_pdf
    from matplotlib.backends.backend_pdf import PdfPages
    if fOUT is not None: pp = PdfPages(fOUT)
    mean_recon = (peakpos, 0.0)
    CalibrationFactor = 1.
    data_new = data
    # print '==========='
    # print 'calibrating\t isMC:\t', isMC
    # print '==========='
    for i in range(7):
        data_new = data_new / (mean_recon[0] / peakpos)
        plot = plt.figure()
        mean_recon, sig_recon = fit_spectrum(data=data_new, peakpos=peakpos, fit=True, name=(name + '_' + str(i)), color='k', isMC=isMC, peakfinder=peakfinder)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Probability')
        plt.legend(loc="lower left")
        plt.axvline(x=2614.5, lw=2, color='k')
        plt.xlim(xmin=500, xmax=3500)
        plt.ylim(ymin=(1.0 / float(len(data))), ymax=0.1)
        plt.grid(True)
        plt.gca().set_yscale('log')
        if fOUT is not None: pp.savefig(plot)
        plt.clf()
        plt.close()
        CalibrationFactor *= mean_recon[0] / peakpos
    if fOUT is not None: pp.close()
    return CalibrationFactor


def plot_spectrum(dCNN, dEXO, dTrue, mode, fOUT, mode2='mc'):
    from matplotlib.gridspec import GridSpec

    if mode2 == 'real':
       bins = 1200
       # bins = 2400
       bins = 600
       hist_DNN, bin_edges = np.histogram(dCNN, bins=bins, range=(-6000, 6000), density=True)
       if dEXO is not None:
           hist_EXO, bin_edges = np.histogram(dEXO, bins=bins, range=(-6000, 6000), density=True)
       hist_True, bin_edges = np.histogram(dTrue, bins=bins, range=(-6000, 6000), density=True)
       bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

       coeff_EXO = [0, 2614.53]
       coeff_DNN = [0, 2614.53]

       #  Fit Gauss to Th228 Peak:
       if mode == 'Energy':
           peak = find_peak(hist=hist_True, bin_centres=bin_centres, peakpos=2614.5, peakfinder='max')
           coeff = [hist_True[peak], bin_centres[peak], 50., -0.005]
           low = np.digitize(coeff[1] - (5.5 * abs(coeff[2])), bin_centres)
           up = np.digitize(coeff[1] + (3.0 * abs(coeff[2])), bin_centres)

           from scipy.optimize import curve_fit
           coeff_DNN, var_matrix_DNN = curve_fit(gaussErf, bin_centres[low:up], hist_DNN[low:up], p0=coeff)
           coeff_EXO, var_matrix_EXO = curve_fit(gaussErf, bin_centres[low:up], hist_EXO[low:up], p0=coeff)


           plt.plot(bin_centres[low+7:up]*2614.53/coeff_DNN[1], gaussErf(bin_centres[low+7:up], *coeff_DNN), ls='--', color=(0,0.2,0.4),     #color='blue',
                    label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('DNN:', coeff_DNN[1]*2614.53/coeff_DNN[1], np.absolute(coeff_DNN[2]*2614.53/coeff_DNN[1])))
           plt.plot(bin_centres[low+7:up]*2614.53/coeff_EXO[1], gaussErf(bin_centres[low+7:up], *coeff_EXO), ls='--', color=(1., 0.49803922, 0.05490196), #color='firebrick',
                    label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('EXO:', coeff_EXO[1]*2614.53/coeff_EXO[1], np.absolute(coeff_EXO[2]*2614.53/coeff_EXO[1])))

           print 'DNN:'
           print np.sqrt(np.diag(var_matrix_DNN)) *2614.53/coeff_DNN[1]
           print 'EXO:'
           print np.sqrt(np.diag(var_matrix_EXO)) *2614.53/coeff_EXO[1]

       plt.step(bin_centres*2614.53/coeff_EXO[1], hist_EXO, where='mid', label='EXO', lw=1.1, alpha=0.6, color=(1., 0.49803922, 0.05490196)) #color='firebrick',
       plt.step(bin_centres*2614.53/coeff_DNN[1], hist_DNN, where='mid', label='DNN', lw=1.1, alpha=0.6, color=(0,0.2,0.4)) #color='blue',


       plt.xlabel(mode + ' [mm]')
       plt.ylabel('Probability')
       plt.legend(loc="best", prop={'size': 13})
       # plt.gca().set_yscale('log')
       if mode == 'Energy':
           # plt.gca().set_yscale('log')
           plt.xlim(xmin=1000, xmax=3000)
           # plt.ylim(ymin=5.e-6, ymax=1.e-2)
           plt.ylim(ymin=2.e-5, ymax=0.0012)
           plt.legend(loc='upper left', prop={'size': 10})
           plt.xlabel('Energy [keV]')
       elif mode == 'Time':
           plt.xlim(xmin=1030, xmax=1140)
           plt.gca().set_yscale('linear')
       elif mode == 'Z':
           plt.xlim(xmin=-200, xmax=200)
           plt.ylim(ymin=0)
       elif mode in ['X', 'Y', 'U', 'V']:
           plt.xlim(xmin=-200, xmax=200)
           plt.ylim(ymin=0)
       else:
           raise ValueError('wrong mode chosen')
       # plt.grid(True)
       plt.savefig(fOUT, bbox_inches='tight')
       plt.clf()
       plt.close()


    elif mode2 == 'real_Z':
        f = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[5, 2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)

        bins = 1200
        # bins = 2400
        bins = 40

        bin_leftedge = np.linspace(-200, -10, 15)
        bin_rightedge = np.linspace(10, 200, 15)
        bins = np.concatenate((bin_leftedge, np.asarray([-7, 7]), bin_rightedge))

        hist_DNN2, bin_edges = np.histogram(dCNN, bins=bins, range=(-200, 200))
        hist_EXO2, bin_edges = np.histogram(dEXO, bins=bins, range=(-200, 200))



        hist_DNN, bin_edges = np.histogram(dCNN, bins=bins, range=(-200, 200), density=True)
        hist_EXO, bin_edges = np.histogram(dEXO, bins=bins, range=(-200, 200), density=True)

        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        error = np.sqrt(hist_DNN2 + hist_EXO2)*(hist_DNN/hist_DNN2)

        ax1.step(bin_centres, hist_EXO, where='mid', label='EXO', lw=1.1, alpha=0.8,
                 color=(1., 0.49803922, 0.05490196))  # color='firebrick',
        ax1.step(bin_centres, hist_DNN, where='mid', label='DNN', lw=1.1, alpha=0.8,
                 color=(0, 0.2, 0.4))  # color='blue',



        ax2.errorbar(bin_centres, (hist_DNN-hist_EXO), error, color='k', fmt='.')

        plt.xlabel(mode + ' [mm]')


        ax1.set_xlim(xmin=-200, xmax=200)
        ax1.set_ylim(ymin=0)


        ax2.axhline(y=0., c='k', alpha=0.3)

        ax2.set_xlabel('Z [mm]')
        ax1.set_ylabel('Probability density')
        ax2.set_ylabel('DNN Z - EXO Z')
        ax1.legend(loc="best")
        # ax2.set_ylim(ymin=-40, ymax=40)

        # plt.grid(True)
        f.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()


    else:
        hist_DNN, bin_edges = np.histogram(dCNN, bins=1200, range=(-6000, 6000), density=True)
        if dEXO is not None:
            hist_EXO, bin_edges = np.histogram(dEXO, bins=1200, range=(-6000, 6000), density=True)
        hist_True, bin_edges = np.histogram(dTrue, bins=1200, range=(-6000, 6000), density=True)
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

        # plt.fill_between(bin_centres, 0.0, hist_True, facecolor='black', alpha=0.3, interpolate=True)
        # plt.plot(bin_centres, hist_True, color='k', label='MC', lw=0.5)
        plt.hist(dTrue, bins=1200, range=(-6000, 6000), density=True, histtype='stepfilled', align='mid', color='k', alpha=0.3, lw=0.0)
        plt.step(bin_centres, hist_True, where='mid', color='k', label='MC', lw=0.7)
        if dEXO is not None:
            plt.step(bin_centres, hist_EXO, where='mid', color='firebrick', label='EXO', lw=1.1)
        plt.step(bin_centres, hist_DNN, where='mid', color='blue', label='DNN', lw=1.1)


        #  Fit Gauss to Th228 Peak:
        if mode == 'Energy':
            peak = find_peak(hist=hist_True, bin_centres=bin_centres, peakpos=2614.5, peakfinder='max')
            coeff = [hist_True[peak], bin_centres[peak], 50., -0.005]
            low = np.digitize(coeff[1] - (5.5 * abs(coeff[2])), bin_centres)
            up = np.digitize(coeff[1] + (3.0 * abs(coeff[2])), bin_centres)

            from scipy.optimize import curve_fit
            coeff_DNN, var_matrix_DNN = curve_fit(gaussErf, bin_centres[low:up], hist_DNN[low:up], p0=coeff)
            coeff_EXO, var_matrix_EXO = curve_fit(gaussErf, bin_centres[low:up], hist_EXO[low:up], p0=coeff)


            plt.plot(bin_centres, gauss_zero(bin_centres, *coeff_DNN[:3]), ls='--', color='blue', label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('DNN:', coeff_DNN[1], np.absolute(coeff_DNN[2])))
            plt.plot(bin_centres, gauss_zero(bin_centres, *coeff_EXO[:3]), ls='--', color='firebrick', label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('EXO:', coeff_EXO[1], np.absolute(coeff_EXO[2])))

        plt.xlabel(mode + ' [mm]')
        plt.ylabel('Probability')
        plt.legend(loc="best", prop={'size': 13})
        # plt.gca().set_yscale('log')
        if mode == 'Energy':
            plt.gca().set_yscale('log')
            plt.xlim(xmin=700, xmax=2700)
            # plt.ylim(ymin=5.e-6, ymax=1.e-2)
            plt.ylim(ymin=2.e-5, ymax=1.e-2)
            plt.legend(loc='lower left', prop={'size': 10})
            plt.xlabel('Energy [keV]')
        elif mode == 'Time':
            plt.xlim(xmin=1030, xmax=1140)
            plt.gca().set_yscale('linear')
        elif mode == 'Z':
            plt.xlim(xmin=-200, xmax=200)
        elif mode in ['X', 'Y', 'U', 'V']:
            plt.xlim(xmin=-200, xmax=200)
            # plt.ylim(ymin=5.e-6, ymax=1.e-2)
        else: raise ValueError('wrong mode chosen')
        # plt.grid(True)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()



def plot_diagonal(x, y, xlabel, ylabel, mode, fOUT, mode2='mc'):
    if mode2 == 'real':
        dE = y - x

        if mode == 'Energy':
            lowE = 550
            upE = 2900
            resE = 100
            gridsize = 100
            shifts = [200, 400, 600, 800]
            shifts_res = [50, 100]
        elif mode == 'Time':
            lowE = 1020
            upE = 1140
            resE = 10
            gridsize = 100
            shifts = [10, 20, 30, 40]
            shifts_res = [5]
        elif mode == 'R':
            lowE = 0
            upE = 160
            resE = 20
            gridsize = 100
            shifts = [10, 20, 30, 40]
            shifts_res = [10, 20]
        elif mode in ['X', 'Y', 'Z', 'U', 'V']:
            lowE = -180
            upE = 180
            resE = 7
            gridsize = 50
            shifts = [20, 40, 60, 80]
            shifts_res = [5, 10, 15]

        diag = np.asarray([lowE, upE])
        extent1 = [lowE, upE, lowE, upE]
        extent2 = [lowE, upE, -resE, resE]
        # plt.ion()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [3, 1]},
                                       figsize=(8.5, 11.5))  # , sharex=True) #, gridspec_kw = {'height_ratios':[3, 1]})
        # plt.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.1, wspace=0.0)
        fig.subplots_adjust(wspace=0, hspace=0.05)
        ax1.set(aspect='equal', adjustable='box-forced')
        ax1.set(aspect='auto')

        ax1.plot(diag, diag, 'k--', lw=2)
        for idx, shift in enumerate(shifts):
            ax1.plot(diag, diag + shift, 'k--', alpha=(0.8 - 0.2 * idx), lw=2, label=str(shift))
            ax1.plot(diag, diag - shift, 'k--', alpha=(0.8 - 0.2 * idx), lw=2, label=str(shift))

        xvals = [2700., 3100., 2500., 3100., 2300., 3100]
        # labelLines(ax1.get_lines()[3:], xvals=xvals, align=True, color='k')

        ax2.axhline(y=0.0, ls='--', lw=2, color='black')
        for idx, shift in enumerate(shifts_res):
            ax2.axhline(y=-shift, ls='--', lw=2, alpha=(0.7 - 0.3 * idx), color='black')
            ax2.axhline(y=shift, ls='--', lw=2, alpha=(0.7 - 0.3 * idx), color='black')
        # ax2.axhline(y=-200.0, ls='--', lw=2, color='black')
        # ax2.axhline(y= 200.0, ls='--', lw=2, color='black')
        ax1.set(ylabel='DNN U [mm]')
        ax2.set(xlabel='EXO U [mm]', ylabel='Residual [mm]')
        # ax1.set(ylabel=ylabel + ' ' + mode)
        # ax2.set(xlabel=xlabel + ' ' + mode, ylabel='(%s - %s)' % (ylabel, xlabel))
        ax1.set_xlim([lowE, upE])
        ax1.set_ylim([lowE, upE])
        ax2.set_ylim([-resE, resE])
        # ax1.xaxis.grid(True)
        # ax1.yaxis.grid(True)
        # ax2.xaxis.grid(True)
        # ax2.yaxis.grid(True)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        # plt.setp(ax2, yticks=[-100, -50, 0, 50, 100])
        ax1.hexbin(x, y, bins='log', extent=extent1, gridsize=2*gridsize, mincnt=1, cmap=plt.get_cmap('viridis'),
                   linewidths=0.1)
        ax2.hexbin(x, dE, bins='log', extent=extent2, gridsize=gridsize, mincnt=1, cmap=plt.get_cmap('viridis'),
                   linewidths=0.1)
        # plt.show()
        # raw_input("")
        plt.savefig(fOUT)
        plt.clf()
        plt.close()

    else:
        dE = y - x

        if mode == 'Energy':
            lowE = 550
            upE = 2900
            resE = 100
            gridsize = 100
            shifts = [200,400,600,800]
            shifts_res = [50,100]
        elif mode == 'Time':
            lowE = 1020
            upE = 1140
            resE = 10
            gridsize = 100
            shifts = [10, 20, 30, 40]
            shifts_res = [5]
        elif mode == 'R':
            lowE = 0
            upE = 160
            resE = 20
            gridsize = 100
            shifts = [10, 20, 30, 40]
            shifts_res = [10, 20]
        elif mode in ['X', 'Y', 'Z', 'U', 'V']:
            lowE = -180
            upE = 180
            resE = 5
            gridsize = 100
            shifts = [20, 40, 60, 80]
            shifts_res = [5, 10, 15]

        diag = np.asarray([lowE, upE])
        extent1 = [lowE, upE, lowE, upE]
        extent2 = [lowE, upE, -resE, resE]
        # plt.ion()

        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw = {'height_ratios':[3, 1]}, figsize=(8.5, 11.5)) #, sharex=True) #, gridspec_kw = {'height_ratios':[3, 1]})
        # plt.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.1, wspace=0.0)
        fig.subplots_adjust(wspace=0, hspace=0.05)
        ax1.set(aspect='equal', adjustable='box-forced')
        ax1.set(aspect='auto')

        ax1.plot(diag, diag, 'k--', lw=2)
        for idx,shift in enumerate(shifts):
            ax1.plot(diag, diag+shift, 'k--', alpha=(0.8-0.2*idx), lw=2, label=str(shift))
            ax1.plot(diag, diag-shift, 'k--', alpha=(0.8-0.2*idx), lw=2, label=str(shift))

        xvals = [2700., 3100., 2500., 3100., 2300., 3100]
        # labelLines(ax1.get_lines()[3:], xvals=xvals, align=True,color='k')

        ax2.axhline(y=0.0, ls='--', lw=2, color='black')
        for idx,shift in enumerate(shifts_res):
            ax2.axhline(y=-shift, ls='--', lw=2, alpha=(0.7-0.3*idx), color='black')
            ax2.axhline(y=shift , ls='--', lw=2, alpha=(0.7-0.3*idx), color='black')
        # ax2.axhline(y=-200.0, ls='--', lw=2, color='black')
        # ax2.axhline(y= 200.0, ls='--', lw=2, color='black')
        ax1.set(ylabel='DNN U [mm]')
        ax2.set(xlabel='True MC U [mm]', ylabel='Residual [mm]')
        # ax1.set(ylabel=ylabel + ' ' + mode)
        # ax2.set(xlabel=xlabel + ' ' + mode, ylabel='(%s - %s)' % (ylabel, xlabel))
        ax1.set_xlim([lowE, upE])
        ax1.set_ylim([lowE, upE])
        ax2.set_ylim([-resE, resE])
        # ax1.xaxis.grid(True)
        # ax1.yaxis.grid(True)
        # ax2.xaxis.grid(True)
        # ax2.yaxis.grid(True)

        fig.tight_layout()
        fig.subplots_adjust(hspace=0.05)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        # plt.setp(ax2, yticks=[-100, -50, 0, 50, 100])
        ax1.hexbin(x, y, bins='log', extent=extent1, gridsize=gridsize, mincnt=1, cmap=plt.get_cmap('viridis'), linewidths=0.1)
        ax2.hexbin(x, dE, bins='log', extent=extent2, gridsize=gridsize, mincnt=1, cmap=plt.get_cmap('viridis'), linewidths=0.1)
        # plt.show()
        # raw_input("")
        plt.savefig(fOUT)
        plt.clf()
        plt.close()
    return


# training curves
def plot_losses(folderOUT, history):
    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['loss'],     label='training')
    ax.plot(history['epoch'], history['val_loss'], label='validation')
    ax.set(xlabel='epoch', ylabel='loss')
    ax.grid(True)
    ax.semilogy()
    plt.legend(loc="best")
    fig.savefig(folderOUT+'loss-test.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    fig, ax = plt.subplots(1)
    ax.plot(history['epoch'], history['mean_absolute_error'],     label='training')
    ax.plot(history['epoch'], history['val_mean_absolute_error'], label='validation')
    ax.legend()
    ax.grid(True)
    plt.legend(loc="best")
    ax.set(xlabel='epoch', ylabel='mean absolute error')
    fig.savefig(folderOUT+'mean_absolute_error-test.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()
    return



def plot_boxplot(dTrue, dTrue_masked, dEXO, dEXO_masked, dDNN, dDNN_masked, dTrue_masked_zeroed, dEXO_masked_zeroed, dDNN_masked_zeroed, title, name_DNN_masked, name_EXO_masked, fOUT, name_DNN='DNN', name_EXO='EXO', mode='wirecheck'):
    from matplotlib.gridspec import GridSpec
    from matplotlib.font_manager import FontProperties

    points = 20000
    points = 100
    # points = 200
    # points = 5000

    if mode == 'SS':
        plt.rc('font', size=11)
        delDNN = dDNN - dTrue
        delDNN_masked = dDNN_masked - dTrue_masked
        delEXO = dEXO - dTrue
        delEXO_masked = dEXO_masked - dTrue_masked
        delDNN_zeroed = dDNN_masked_zeroed - dTrue_masked_zeroed
        delEXO_zeroed = dEXO_masked_zeroed - dTrue_masked_zeroed


        data = [delEXO, delDNN, delEXO_masked, delDNN_masked, delEXO_zeroed, delDNN_zeroed]
        labels = ['EXO', 'DNN', 'EXO', 'DNN', 'EXO', 'DNN']

        font = FontProperties()
        font.set_family('serif')

        positions = [1, 1.3, 2, 2.3, 3, 3.3]
        positions_violin = [1.15, 2.15, 3.15]
        limit = 5
        limit_max = 50
        whis = [2.5, 97.5]

        plt.text(-limit + 0.5, 3.2, '(a)', fontproperties=font)
        plt.text(-limit + 0.5, 2.2, '(b)', fontproperties=font)
        plt.text(-limit + 0.5, 1.2, '(c)', fontproperties=font)

        # color=(0, 0.2, 0.4))color=(0.89, 0.1, 0.11))

        plt.axvline(x=0, ymin=0, ymax=1, color='black', alpha=0.3)
        plt.xlim(xmax=limit, xmin=-limit)
        plt.xlabel('Residual [mm]', fontproperties=font)


        plt.boxplot([data[5], data[3], data[1]], notch=True, positions=[positions[1], positions[3], positions[5]],
                    vert=False, whis=whis, showfliers=False, widths=0.2)
        v2 = plt.violinplot([data[5], data[3], data[1]], positions=positions_violin, widths=0.9, showextrema=False,
                            vert=False, points=points, bw_method=0.03)

        plt.boxplot([data[4], data[2], data[0]], notch=True, vert=False, whis=whis, showfliers=False, widths=0.2)
        v1 = plt.violinplot([data[4], data[2], data[0]], positions=positions_violin, widths=0.9, showextrema=False,
                            vert=False, points=points, bw_method=0.03)

        for b in v1['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, m)
            # b.set_facecolor((0.86, 0.12, 0.12))
            # b.set_edgecolor((0.86, 0.12, 0.12))
            b.set_alpha(0.4)
        for b in v2['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            b.set_facecolor((0, 0.2, 0.4))
            b.set_edgecolor((0, 0.2, 0.4))
            b.set_alpha(0.5)

        plt.yticks(positions, labels, fontproperties=font)
        # plt.set_yticklabels(labels)
        plt.ylim([0.6, 3.7])
        plt.xticks(fontproperties=font)

        # plt.legend(loc="best", prop={'size': 13})
        # plt.xlim(xmin=-limit, xmax=limit)
        # plt.ylim(ymin=-20, ymax=20)
        # plt.yscale('symlog', basey=5)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()
        return

    if mode == 'wirecheck':
        plt.rc('font', size=11)
        delDNN = dDNN - dTrue
        delDNN_masked = dDNN_masked - dTrue_masked
        delEXO = dEXO - dTrue
        delEXO_masked = dEXO_masked - dTrue_masked
        delDNN_zeroed = dDNN_masked_zeroed - dTrue_masked_zeroed
        delEXO_zeroed = dEXO_masked_zeroed - dTrue_masked_zeroed

        data = [delEXO, delDNN, delEXO_masked, delDNN_masked, delEXO_zeroed, delDNN_zeroed]
        labels = ['EXO', 'DNN', 'EXO', 'DNN', 'EXO', 'DNN']

        font = FontProperties()
        font.set_family('serif')


        positions = [1, 1.3, 2, 2.3, 3, 3.3]
        positions_violin = [1.15, 2.15, 3.15]
        limit = 5
        limit_max = 50
        whis = [2.5, 97.5]

        plt.text(-limit + 0.5, 3.2, '(a)', fontproperties=font)
        plt.text(-limit + 0.5, 2.2, '(b)', fontproperties=font)
        # plt.text(-limit + 0.5, 1.15, 'c)', fontproperties=font)


        plt.axvline(x=0, ymin=0, ymax=1, color='black', alpha=0.3)
        plt.xlim(xmax=limit, xmin=-limit)
        plt.xlabel('Residual [mm]', fontproperties=font)


        plt.boxplot([data[1], data[5], data[3]], notch=True, positions=[positions[1], positions[3], positions[5]], vert=False, whis=whis, showfliers=False, widths=0.2)
        v2 = plt.violinplot([data[1], data[5], data[3]], positions=positions_violin, widths=0.9, showextrema=False, vert=False, points=points, bw_method=0.03)

        plt.boxplot([data[0], data[4], data[2]], notch=True, vert=False, whis=whis, showfliers=False, widths=0.2)
        v1 = plt.violinplot([data[0], data[4], data[2]], positions=positions_violin, widths=0.9, showextrema=False, vert=False, points=points, bw_method=0.03)


        # print
        # print
        # print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
        for b in v1['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], -np.inf, m)
            # b.set_facecolor((0.89, 0.1, 0.11))
            # b.set_edgecolor((0.89, 0.1, 0.11))
            b.set_alpha(0.4)
            # print b.get_facecolor()

        for b in v2['bodies']:
            m = np.mean(b.get_paths()[0].vertices[:, 1])
            b.get_paths()[0].vertices[:, 1] = np.clip(b.get_paths()[0].vertices[:, 1], m, np.inf)
            b.set_facecolor((0, 0.2, 0.4))
            b.set_edgecolor((0, 0.2, 0.4))
            b.set_alpha(0.5)

        plt.yticks(positions, labels, fontproperties=font)
        # plt.set_yticklabels(labels)
        plt.ylim([1.6, 3.7])
        plt.xticks(fontproperties=font)

        # plt.legend(loc="best", prop={'size': 13})
        # plt.xlim(xmin=-limit, xmax=limit)
        # plt.ylim(ymin=-20, ymax=20)
        # plt.yscale('symlog', basey=5)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()
    return


def plot_residual_correlation(dTrue, dDNN, dEXO, mode, fOUT, limit=10):
    from matplotlib.ticker import NullFormatter
    from matplotlib.font_manager import FontProperties
    from scipy.stats import pearsonr

    plt.rc('font', size=12)

    font = FontProperties()
    font.set_family('serif')
    # font.set_variant(variant)

    color = 'viridis'

    cm = plt.cm.get_cmap(color)

    delDNN = dDNN - dTrue
    delEXO = dEXO - dTrue

    print
    print 'Correlation coefficient (', mode, '): \n', pearsonr(delDNN, delEXO)

    nullfmt = NullFormatter()  # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    # rect_legend = [left_h, bottom_h, 0.2, 0.2]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)


    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    axHisty.set_xlabel('Counts', fontproperties=font)
    axHistx.set_ylabel('Counts', fontproperties=font)
    # axLegend = plt.axes(rect_legend)

    mask_limit_EXO = np.absolute(delEXO) < limit
    mask_limit_DNN = np.absolute(delDNN) < limit
    mask_limit = np.logical_and(mask_limit_DNN, mask_limit_EXO)

    if mode == 'U' or mode == 'V' or mode == 'Z':
        axScatter.set_xlabel('Residual EXO [mm]', fontproperties=font)
        axScatter.set_ylabel('Residual DNN [mm]', fontproperties=font)
        axHistx.text(0.03, 0.9, '$\mu=%.1f$\n$\sigma=%.2f$' % (np.mean(delEXO[mask_limit_EXO]), np.std(delEXO[mask_limit_EXO])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='left',
                     verticalalignment='top', transform=axHistx.transAxes)
        axHisty.text(0.5, 0.97, '$\mu=%.1f$\n$\sigma=%.2f$' % (np.mean(delDNN[mask_limit_DNN]), np.std(delDNN[mask_limit_DNN])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='center',
                     verticalalignment='top', transform=axHisty.transAxes)
    if mode == 'energy':
        axScatter.set_xlabel('Residual EXO [keV]', fontproperties=font)
        axScatter.set_ylabel('Residual DNN [keV]', fontproperties=font)
        axHistx.text(0.03, 0.9, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delEXO[mask_limit_EXO]), np.std(delEXO[mask_limit_EXO])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='left',
         verticalalignment='top', transform=axHistx.transAxes)
        axHisty.text(0.5, 0.97, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delDNN[mask_limit_DNN]), np.std(delDNN[mask_limit_DNN])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='center',
         verticalalignment='top', transform=axHisty.transAxes)

    if mode == 'UvsV':
        axScatter.set_xlabel('Residual V-wire-input [keV]', fontproperties=font)
        axScatter.set_ylabel('Residual U-wire input [keV]', fontproperties=font)
        axHistx.text(0.03, 0.9, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delEXO[mask_limit_EXO]), np.std(delEXO[mask_limit_EXO])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='left',
         verticalalignment='top', transform=axHistx.transAxes)
        axHisty.text(0.5, 0.97, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delDNN[mask_limit_DNN]), np.std(delDNN[mask_limit_DNN])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='center',
         verticalalignment='top', transform=axHisty.transAxes)
    if mode == 'UVvsU':
        axScatter.set_xlabel('Residual U-wire-input [keV]', fontproperties=font)
        axScatter.set_ylabel('Residual UV-wire input [keV]', fontproperties=font)
        axHistx.text(0.03, 0.9, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delEXO[mask_limit_EXO]), np.std(delEXO[mask_limit_EXO])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='left',
         verticalalignment='top', transform=axHistx.transAxes)
        axHisty.text(0.5, 0.97, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delDNN[mask_limit_DNN]), np.std(delDNN[mask_limit_DNN])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='center',
         verticalalignment='top', transform=axHisty.transAxes)
    if mode == 'UVvsV':
        axScatter.set_xlabel('Residual V-wire-input [keV]', fontproperties=font)
        axScatter.set_ylabel('Residual UV-wire input [keV]', fontproperties=font)
        axHistx.text(0.03, 0.9, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delEXO[mask_limit_EXO]), np.std(delEXO[mask_limit_EXO])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='left',
         verticalalignment='top', transform=axHistx.transAxes)
        axHisty.text(0.5, 0.97, '$\mu=%.1f$\n$\sigma=%.1f$' % (np.mean(delDNN[mask_limit_DNN]), np.std(delDNN[mask_limit_DNN])), fontproperties=font, bbox=dict(fc="none"), horizontalalignment='center',
         verticalalignment='top', transform=axHisty.transAxes)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    binwidth = 0.15
    binwidth = 100 / (2 * limit)

    lim = limit
    gridsize = 80

    # the scatter plot:
    axScatter.hexbin(delEXO[mask_limit], delDNN[mask_limit], gridsize=gridsize, mincnt=1,  # norm=colors.Normalize(),
           cmap=plt.get_cmap('viridis'), linewidths=0.1, bins='log')


    axScatter.set_xlim((-lim+0.1, lim-0.1))
    axScatter.set_ylim((-lim+0.1, lim-0.1))

    if mode == 'energy':
        axScatter.set_xlim((-lim, 99))
        axScatter.set_ylim((-lim, 99))


    # if mode == 'UvsV' or mode == 'UVvsU' or mode == 'UVvsV':
    #     axScatter.set_xlim((-lim, lim))
    #     axScatter.set_ylim((-lim, lim))

    # bins = np.arange(-lim, lim, binwidth)
    bins = 200

    n_x, bins_x, patches_x = axHistx.hist(delEXO[mask_limit_EXO], bins=bins, histtype='stepfilled', color=(1., 0.49803922, 0.05490196), alpha=0.4)#, color=(0.89, 0.1, 0.11))
    n_y, bins_y, patches_y = axHisty.hist(delDNN[mask_limit_DNN], bins=bins, orientation='horizontal', histtype='stepfilled', color=(0, 0.2, 0.4), alpha=0.5)

    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.xticks(fontproperties=font)
    plt.yticks(fontproperties=font)

    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# histogram of the data
def plot_residual_histo(dTrue, dDNN, dEXO, title, name_True, name_DNN, name_EXO, fOUT, dMC_EXO=None, dMC_DNN=None, limit=10, mode='mc'):
    from matplotlib.gridspec import GridSpec
    plt.rc('font', size=12)

    if mode == 'real':
        res_real = dDNN - dEXO
        if dEXO is not None:
            res_MC = dMC_DNN - dMC_EXO
            mask_range_EXO = [np.absolute(res_MC) < limit]
            res_MC = res_MC[mask_range_EXO]

        mask_range_DNN = [np.absolute(res_real) < limit]
        res_real = res_real[mask_range_DNN]
        bins = 100

        if np.mean(res_MC) > -0.005 and np.mean(res_MC) < 0.0:
            plt.hist(res_MC, bins=bins, range=(-limit, limit), density=True, color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.2f$' % ('MC:', 0.00, np.std(res_MC)))
        else:
            plt.hist(res_MC, bins=bins, range=(-limit, limit), density=True,
                                                 color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.2f$' % (
                                                 'MC:', np.mean(res_MC), np.std(res_MC)))

        if np.mean(res_real) > -0.005 and np.mean(res_real) < 0.0:
            plt.hist(res_real, bins=bins, range=(-limit, limit), density=True, color=(0, 0.2, 0.4), alpha=0.5, histtype='stepfilled',
                                                 # label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', 0.00, np.std(res_real)))
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Real:', 0.00, np.std(res_real)))
        else:
            plt.hist(res_real, bins=bins, range=(-limit, limit), density=True, color=(0, 0.2, 0.4), alpha=0.5, histtype='stepfilled',
                                                 # label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', np.mean(res_real), np.std(res_real)))
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Real:', np.mean(res_real), np.std(res_real)))

        plt.xlabel('DNN ' + title + ' - EXO ' + title + ' [mm]')
        plt.ylabel('Probability density')
        plt.legend(loc="best")
        plt.xlim(xmin=-limit, xmax=limit)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()

    elif mode == 'real_V':
        res_real = dDNN - dEXO
        if dEXO is not None:
            res_MC = dMC_DNN - dMC_EXO
            mask_range_EXO = [np.absolute(res_MC) < limit]
            res_MC = res_MC[mask_range_EXO]

        mask_range_DNN = [np.absolute(res_real) < limit]
        res_real = res_real[mask_range_DNN]

        bin_center = np.linspace(-3, 3, 10)
        bins = np.concatenate((np.asarray([-7, -4]), bin_center, np.asarray([4, 7])))

        f = plt.figure()
        gs = GridSpec(2, 1, height_ratios=[5, 2])
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1], sharex=ax1)

        if np.mean(res_MC) > -0.005 and np.mean(res_MC) < 0.0:

            # hist_MC1, bins_MC, patches = ax1.hist(res_MC, bins=bins_edge, range=(-limit, -2), density=True, color=(1., 0.49803922, 0.05490196), alpha=0.8, histtype='step')

            hist_MC, bins_MC, patches = ax1.hist(res_MC, bins=bins, range=(-limit, limit), density=True, color=(0.3, 0.69, 0.29), histtype='step', linewidth=1.5,
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('MC:', 0.00, np.std(res_MC)))
            #
            # hist_MC3, bins_MC, patches = ax1.hist(res_MC, bins=bins_edge, range=(2, limit), density=True, color=(1., 0.49803922, 0.05490196), alpha=0.8, histtype='step')
            #
            # hist_MC = np.concatenate(hist_MC1, hist_MC2, hist_MC3)

        else:
            hist_MC, bins_MC, patches = ax1.hist(res_MC, bins=bins, range=(-limit, limit), density=True,
                                                 color=(0.3, 0.69, 0.29), histtype='step', linewidth=1.5,
                                                 label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % (
                                                 'MC:', np.mean(res_MC), np.std(res_MC)))

        if np.mean(res_real) > -0.005 and np.mean(res_real) < 0.0:
            hist, bin_edges = np.histogram(res_real, bins=bins, range=(-limit, limit))

            error = np.sqrt(hist)
            hist2, bin_edges = np.histogram(res_real, bins=bins, range=(-limit, limit), density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

            error = error * hist2/hist
            ax1.errorbar(bin_centres, hist2, yerr=error, label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Real:', 0.0, np.std(res_real)), fmt='.', color='black')

        else:
            hist, bin_edges = np.histogram(res_real, bins=bins, range=(-limit, limit))
            error = np.sqrt(hist)
            hist2, bin_edges = np.histogram(res_real, bins=bins, range=(-limit, limit), density=True)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            error = error * hist2 / hist

            ax1.errorbar(bin_centres, hist2, yerr=error, label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('Real:', np.mean(res_real), np.std(res_real)), fmt='.', color='black')

        ax2.errorbar(bin_centres, (hist2 - hist_MC) / hist_MC*100, error / hist_MC*100, color='k', fmt='.')#, label='%s (%s)' % (source, position))
        ax2.axhline(y=0., c='k', alpha=0.3)

        ax2.set_xlabel('DNN ' + title + ' - EXO ' + title + ' [mm]')
        ax1.set_ylabel('Probability density')
        ax2.set_ylabel('(data-MC)/MC [%]')
        ax1.legend(loc="best")
        ax1.set_xlim(xmin=-limit, xmax=limit)
        ax2.set_ylim(ymin=-40, ymax=40)

        f.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()

    else:
        delDNN = dDNN - dTrue
        if dEXO is not None:

            delEXO = dEXO - dTrue
            mask_range_EXO = delEXO < 100
            mask_range_EXO = np.logical_and(mask_range_EXO, delEXO > -200)
            delEXO = delEXO[mask_range_EXO]

        mask_range_DNN = delDNN < 100
        mask_range_DNN = np.logical_and(mask_range_DNN, delDNN > -200)
        delDNN = delDNN[mask_range_DNN]

        bins = 200
        if limit == 100:


            if dEXO is not None:
                hist_delEXO, bin_edges, _ = plt.hist(delEXO, bins=bins, range=(-200, 100), density=False, histtype='stepfilled',color=(1., 0.49803922, 0.05490196), alpha=0.4, label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('EXO:', np.mean(delEXO), np.std(delEXO)))

            if np.mean(delDNN) > -0.005 and np.mean(delDNN) < 0.0:
                hist_delDNN, bin_edges, _ = plt.hist(delDNN, bins=bins, range=(-200, 100), density=False, histtype='stepfilled',color=(0, 0.2, 0.4), alpha=0.5, label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$'%('DNN:', 0.0, np.std(delDNN)))
            else:
                hist_delDNN, bin_edges, _ = plt.hist(delDNN, bins=bins, range=(-200, 100), density=False,histtype='stepfilled', color=(0, 0.2, 0.4), alpha=0.5, label='%s\n$\mu=%.1f$\n$ \sigma=%.1f$' % ('DNN:', np.mean(delDNN), np.std(delDNN)))

        else:
            bins = 200
            if np.mean(delEXO) > -0.005 and np.mean(delEXO) < 0.0:
                hist_delEXO, bin_edges, _ = plt.hist(delEXO, bins=bins, range=(-limit, limit), density=True, color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
                                                     label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('EXO:', 0.00, np.std(delEXO)))
            else:
                hist_delEXO, bin_edges, _ = plt.hist(delEXO, bins=bins, range=(-limit, limit), density=True,color=(1., 0.49803922, 0.05490196), alpha=0.4, histtype='stepfilled',
                                                     label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('EXO:', np.mean(delEXO), np.std(delEXO)))
            if np.mean(delDNN) > -0.005 and np.mean(delDNN) < 0.0:
                hist_delDNN, bin_edges, _ = plt.hist(delDNN, bins=bins, range=(-limit, limit), density=True, color=(0, 0.2, 0.4), alpha=0.5, histtype='stepfilled',
                                                    label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', 0.00, np.std(delDNN)))
            else:
                hist_delDNN, bin_edges, _ = plt.hist(delDNN, bins=bins, range=(-limit, limit), density=True, color=(0, 0.2, 0.4), alpha=0.5, histtype='stepfilled',
                                                     label='%s\n$\mu=%.2f$\n$ \sigma=%.2f$' % ('DNN:', np.mean(delDNN), np.std(delDNN)))

        plt.xlabel('Residual ' + title + ' [keV]')
        plt.ylabel('Probability density')
        # if limit == 100:
        #     plt.gca().set_yscale('log')

        plt.legend(loc="upper left")
        plt.xlim(xmin=-200, xmax=100)
        plt.savefig(fOUT, bbox_inches='tight')
        plt.clf()
        plt.close()
    return


def plot_scatter(E_x, E_y, name_x, name_y, fOUT, alpha=1):
    dE = E_x - E_y
    diag = np.arange(min(E_x), max(E_x)+1)
    plt.scatter(E_x, E_y, label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)), alpha=alpha)
    plt.plot(diag, diag, 'k--')
    plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    # plt.xlim(xmin=600, xmax=3300)
    # plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [-self.vmax, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_hexbin(X_true, Y_true, Z_true, DNN, EXO, name_x, name_y, fOUT):
    from matplotlib.ticker import NullFormatter
    nullfmt = NullFormatter()

    plt.rc('font', size=35)

    xmin = X_true.min() - 10
    xmax = X_true.max() + 10
    ymin = Y_true.min() - 10
    ymax = Y_true.max() + 10
    zmin = Z_true.min() - 10
    zmax = Z_true.max() + 10

    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(25, 25))  # , gridspec_kw={'height_ratios': [3, 1]})
    axes[0, 0].set(aspect='auto')#, adjustable='box-forced')
    axes[1, 0].set(aspect='auto')#, adjustable='box-forced')
    axes[0, 1].set(aspect='auto')#, adjustable='box-forced')
    axes[1, 1].set(aspect='auto')#, adjustable='box-forced')

    axes[0, 0].axis([xmin, xmax, zmin, zmax])
    axes[0, 1].axis([xmin, xmax, zmin, zmax])
    axes[1, 0].axis([xmin, xmax, ymin, ymax])
    axes[1, 1].axis([xmin, xmax, ymin, ymax])

    num_bins = 50
    vmax = 60

    ax = axes[0, 0]
    ax.set_title('DNN')
    im = ax.hexbin(X_true, Z_true, C=DNN, cmap='viridis', linewidths=0.1, gridsize=num_bins,
                   reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[0, 1]
    ax.set_title('EXO')
    im = ax.hexbin(X_true, Z_true, C=EXO, cmap='viridis', linewidths=0.1, gridsize=num_bins,
                   reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[1, 0]
    # ax.set_title('DNN')
    im = ax.hexbin(X_true, Y_true, C=DNN, cmap='viridis', linewidths=0.1, gridsize=num_bins,
                   reduce_C_function=np.std, vmin=0., vmax=vmax)

    ax = axes[1, 1]
    # ax.set_title('EXO')
    im = ax.hexbin(X_true, Y_true, C=EXO, cmap='viridis', linewidths=0.1, gridsize=num_bins,
                   reduce_C_function=np.std, vmin=0., vmax=vmax)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.set_ylabel('Energy resolution [keV]')

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

    # xmin = E_x.min()-10
    # xmax = E_x.max()+10
    # ymin = E_y.min()-10
    # ymax = E_y.max()+10
    # fig, axes = plt.subplots(1, 2, sharey=True, figsize=(25, 10))#, gridspec_kw={'height_ratios': [3, 1]})
    # axes[0].set(aspect='equal', adjustable='box-forced')
    # axes[1].set(aspect='equal', adjustable='box-forced')
    # num_bins = 80
    #
    # ax = axes[0]
    # ax.set_title('DNN')
    # # im = ax.hexbin(E_x, E_y, C=DNN, cmap='RdBu_r', linewidths=0.1, gridsize=num_bins, norm=MidpointNormalize(midpoint=0.))
    # im = ax.hexbin(E_x, E_y, C=DNN, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, bins='log')
    #
    # ax = axes[1]
    # ax.set_title('EXO')
    # plt.setp(ax.get_yticklabels(), visible=False)
    # # im = ax.hexbin(E_x, E_y, C=EXO, cmap='RdBu_r', linewidths=0.1, gridsize=num_bins, norm=MidpointNormalize(midpoint=0.))
    # im = ax.hexbin(E_x, E_y, C=EXO, cmap='viridis', linewidths=0.1, gridsize=num_bins, reduce_C_function=np.std, bins='log')
    # ax.axis([xmin, xmax, ymin, ymax])
    #
    # fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # cbar_ax.set_label('energy residual')
    #
    # axes[0].set_xlabel('%s' % (name_x))
    # axes[1].set_xlabel('%s' % (name_x))
    # axes[0].set_ylabel('%s' % (name_y))
    # plt.savefig(fOUT, bbox_inches='tight')
    # plt.clf()
    # plt.close()
    # return





def plot_hist2D(E_x, E_y, name_x, name_y, fOUT):
    dE = E_x - E_y
    diag = np.arange(min(E_x),max(E_x))
    plt.hist2d(E_x, E_y, cmin=1)
    # plt.plot(diag, diag, 'k--')
    plt.legend(loc="best")
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    # plt.xlim(xmin=600, xmax=3300)
    # plt.ylim(ymin=600, ymax=3300)
    # plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


def  plot_scatter_MS_CR(y_true_cluster, y_true_e, y_pred_cluster, y_pred_e, number_cluster, name_x, name_y, fOUT):
    #
    diag = np.arange(min(y_true_e), max(y_true_e))
    mask_true = [y_true_cluster >= number_cluster]

    # print '>>>>>>>'
    # print y_true_cluster.shape
    # print len(mask_true)

    mask_pred = [y_pred_cluster >= number_cluster]
    mask_both = np.logical_and(mask_pred, mask_true)
    mask_both_not = np.logical_and(np.logical_not(mask_pred), np.logical_not(mask_true))

    # print number_cluster
    # print y_true_cluster
    # print y_pred_cluster
    #
    # print mask_true
    # print mask_pred
    # print '--'
    mask_true = np.logical_and(mask_true, np.logical_not(mask_both))
    mask_pred = np.logical_and(mask_pred, np.logical_not(mask_both))


    #
    # print mask_true
    # print mask_pred
    # print mask_both
    # print mask_both_not

    dE = y_true_e - y_pred_e

    mask_both, mask_pred, mask_true, mask_both_not = np.reshape(mask_both, (-1,)), np.reshape(mask_pred, (-1,)), np.reshape(mask_true, (-1,)), np.reshape(mask_both_not, (-1,))

    # print mask_both.shape
    # print y_true_cluster.shape
    # print y_true_e.shape
    # print y_pred_e.shape

    dE = dE[mask_both]
    # print '>>>>>'
    # print mask_both.shape
    # print y_true_e.shape

    plt.scatter(y_true_e[mask_both], y_pred_e[mask_both], color='green', label='%s\n$\mu=%.1f, \sigma=%.1f$'%('training set', np.mean(dE), np.std(dE)))
    plt.scatter(y_true_e[mask_true], y_pred_e[mask_true], alpha=0.3, color='red', label='true-not_pred') # label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('training set', np.mean(dE), np.std(dE)))
    plt.scatter(y_true_e[mask_pred], y_pred_e[mask_pred], alpha=0.3, color='maroon', label='not_true-pred') #label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('training set', np.mean(dE), np.std(dE)), alpha=0.5, color='red')
    plt.scatter(y_true_e[mask_both_not], y_pred_e[mask_both_not], alpha=0.3, color='blue', label='not_true-not_pred')
    plt.plot(diag, diag, 'k--')
    plt.legend(loc="best", prop={'size': 13})
    plt.xlabel('%s' % (name_x))
    plt.ylabel('%s' % (name_y))
    # plt.xlim(xmin=600, xmax=3300)
    # plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return



# scatter (Hist2D)
def plot_scatter_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, E_y, bins=200, normed=True)
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    diag = np.asarray([600,3300])
    plt.plot(diag, diag, 'k--')
    plt.plot(diag, diag + 200, 'k--')
    plt.plot(diag, diag - 200, 'k--')
    plt.plot(diag, diag + 400, 'k--')
    plt.plot(diag, diag - 400, 'k--')
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'),
                    origin='lower', norm=mpl.colors.LogNorm(),
                    label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('MC data', np.mean(dE), np.std(dE)))
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    cbar.set_label('Probability')
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# scatter (Density)
def plot_scatter_density(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    # Calculate the point density
    from scipy.stats import gaussian_kde
    xy = np.vstack([E_x, E_y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    E_x, E_y, z = E_x[idx], E_y[idx], z[idx]

    plt.scatter(E_x, E_y, c=z, s=2, edgecolor='', cmap=plt.get_cmap('viridis'),
               label='%s\n$\mu=%.1f, \sigma=%.1f$' % ('physics data', np.mean(dE), np.std(dE)))
    plt.plot((600, 3300), (600, 3300), 'k--')
    plt.colorbar()
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=600, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT)
    plt.clf()
    plt.close()
    return

# residual
def plot_residual_scatter(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    plt.scatter(E_x, dE)
    plt.plot((600, 3300), (0,0), color='black')
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-300, ymax=300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# residual (Hist2D)
def plot_residual_hist2d(E_x, E_y, name_x, name_y, fOUT):
    dE = E_y - E_x
    hist, xbins, ybins = np.histogram2d(E_x, dE, range=[[600,3300],[-250,250]], bins=180, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (2700)/(500)
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    plt.plot((600, 3300), (0, 0), color='black')
    cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    cbar.set_label('Probability')
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-250, ymax=250)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# residual (Density)
def plot_residual_density(E_x, E_y, name_x, name_y, fOUT):
    dE = np.array(E_y - E_x)
    E_x = np.array(E_x)
    # Calculate the point density
    from scipy.stats import gaussian_kde
    xy = np.vstack([E_x, dE])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    E_x, E_y, z = E_x[idx], dE[idx], z[idx]

    plt.scatter(E_x, dE, c=z, s=5, edgecolor='', cmap=plt.get_cmap('viridis'))
    plt.plot((600, 3300), (0,0), color='black')
    plt.colorbar()
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-300, ymax=300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# mean-residual
def plot_residual_scatter_mean(E_x, E_y, name_x, name_y, fOUT):
    import warnings
    dE = E_y - E_x
    bin_edges = [ x for x in range(0,4000,150) ]
    bins = [ [] for x in range(0,3850,150) ]
    for i in range(len(dE)):
        bin = np.digitize(E_x[i], bin_edges) - 1
        bins[bin].append(dE[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bins = [ np.array(bin) for bin in bins]
        mean = [ np.mean(bin)  for bin in bins]
        stda = [ np.std(bin)/np.sqrt(len(bin))  for bin in bins]
    bin_width=((bin_edges[1]-bin_edges[0])/2.0)
    plt.errorbar((np.array(bin_edges[:-1])+bin_width), mean, xerr=bin_width, yerr=stda, fmt="none")
    plt.axhline(y=0.0, lw=2, color='k')
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=-50, ymax=50)
    plt.grid(True)
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Mean Residual (%s - %s) [keV]' % (name_y, name_x))
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# mean-residual violin
def plot_residual_violin(E_x, E_CNN, E_EXO, name_x, name_CNN, name_EXO, fOUT):
    import seaborn as sns
    import pandas as pd
    sns.set_style("whitegrid")
    dE_CNN = E_CNN - E_x
    dE_EXO = E_EXO - E_x
    bin_edges = [x for x in range(0, 4250, 250)]
    bin_width = int((bin_edges[1] - bin_edges[0]) / 2.0)
    data_dic = {'energy': [], 'residual': [], 'type': []}
    for i in range(len(E_x)):
        bin_CNN = np.digitize(E_x[i], bin_edges) - 1
        data_dic['energy'].append(bin_edges[bin_CNN]+bin_width)
        data_dic['residual'].append(dE_CNN[i])
        data_dic['type'].append(name_CNN)
        data_dic['energy'].append(bin_edges[bin_CNN]+bin_width)
        data_dic['residual'].append(dE_EXO[i])
        data_dic['type'].append(name_EXO)
    data = pd.DataFrame.from_dict(data_dic)
    fig, ax = plt.subplots()
    ax.axhline(y=0.0, lw=2, color='k')
    sns.violinplot(x='energy', y='residual', hue='type', data=data, inner="quartile", palette='Set2', split=True, cut=0, scale='area', scale_hue=True, bw=0.4)
    ax.set_ylim(-150, 150)
    ax.set_xlabel('%s Energy [keV]' % (name_x))
    ax.set_ylabel('Residual ( xxx - %s ) [keV]' % (name_x))
    fig.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    sns.reset_orig()
    return


# mean-residual
def plot_residual_scatter_sigma(E_x, E_CNN, E_EXO, name_x, name_CNN, name_EXO, fOUT):
    import warnings
    dE_CNN = E_CNN - E_x
    dE_EXO = E_EXO - E_x
    bin_edges = [ x for x in range(0,4000,150) ]
    bins_CNN = [ [] for x in range(0,3850,150) ]
    bins_EXO = [[] for x in range(0, 3850, 150)]
    for i in range(len(dE_CNN)):
        bin_CNN = np.digitize(E_x[i], bin_edges) - 1
        bins_CNN[bin_CNN].append(dE_CNN[i])
        bin_EXO = np.digitize(E_x[i], bin_edges) - 1
        bins_EXO[bin_EXO].append(dE_EXO[i])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bins_EXO = [ np.asarray(bin) for bin in bins_EXO]
        bins_CNN = [np.asarray(bin) for bin in bins_CNN]
        stda_EXO = [ np.std(bin)  for bin in bins_EXO]
        stda_CNN = [np.std(bin) for bin in bins_CNN]
    bin_width=((bin_edges[1]-bin_edges[0])/2.0)
    bin_centers = np.asarray(bin_edges[:-1])+bin_width
    plt.errorbar(bin_centers, 100.*np.asarray(stda_EXO)/bin_centers , xerr=bin_width, fmt="o", label=name_EXO)
    plt.errorbar(bin_centers, 100.*np.asarray(stda_CNN) / bin_centers, xerr=bin_width, fmt="o", label=name_CNN)
    plt.axhline(y=0.0, lw=2, color='k')
    plt.xlim(xmin=600, xmax=3300)
    plt.ylim(ymin=0, ymax=15.)
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('Energy Resolution')
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# anticorrelation
def plot_anticorrelation_hist2d(E_x, E_y, name_x, name_y, name_title, fOUT):
    hist, xbins, ybins = np.histogram2d(E_x, E_y, range=[[0,3300],[0,3300]], bins=250, normed=True )
    extent = [xbins.min(), xbins.max(), ybins.min(), ybins.max()]
    aspect = (3300.0) / (3300.0)
    # plt.plot((500, 3200), (500, 3200), 'k--')
    im = plt.imshow(hist.T, extent=extent, interpolation='nearest', cmap=plt.get_cmap('viridis'), origin='lower',
                    aspect=aspect, norm=mpl.colors.LogNorm())
    #cbar = plt.colorbar(im, fraction=0.025, pad=0.04, ticks=mpl.ticker.LogLocator(subs=range(10)))
    #cbar.set_label('Probability')
    plt.title('%s' % (name_title))
    plt.xlabel('%s Energy [keV]' % (name_x))
    plt.ylabel('%s Energy [keV]' % (name_y))
    plt.xlim(xmin=500, xmax=3300)
    plt.ylim(ymin=500, ymax=3300)
    plt.grid(True)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# rotation vs resolution
def plot_rotationAngle_resolution(fOUT, data):
    for E_List_str in ['E_EXO', 'E_CNN']:
        if E_List_str == 'E_EXO':
            col = 'firebrick'
            label = 'EXO Recon'
        if E_List_str == 'E_CNN':
            col = 'blue'
            label = 'Neural Network'
        TestResolution_ss = np.array(data[E_List_str]['TestResolution_ss']) * 100.
        TestResolution_ms = np.array(data[E_List_str]['TestResolution_ms']) * 100.
        par0ss = data[E_List_str]['BestRes_ss'][0]
        par1ss = data[E_List_str]['Par1_ss'][0]
        par2ss = data[E_List_str]['Theta_ss'][0]
        par0ms = data[E_List_str]['BestRes_ms'][0]
        par1ms = data[E_List_str]['Par1_ms'][0]
        par2ms = data[E_List_str]['Theta_ms'][0]
        print E_List_str, '\tSS\t', par0ss * 100., '\t', par2ss
        print E_List_str, '\tMS\t', par0ms * 100., '\t', par2ms
        limit = 0.07
        x = np.arange(par2ms-limit, par2ms+limit, 0.005)
        plt.errorbar(data[E_List_str]['TestTheta'], TestResolution_ss[:, 0], yerr=TestResolution_ss[:, 1], color=col, fmt="o", label='%s-SS (%.3f%%)' % (label, par0ss * 100.))
        plt.errorbar(data[E_List_str]['TestTheta'], TestResolution_ms[:, 0], yerr=TestResolution_ms[:, 1], color=col, fmt="o", mec=col, mfc='None', label='%s-MS (%.3f%%)' % (label, par0ms * 100.))
        plt.plot(x, parabola(x, par0ss, par1ss, par2ss) * 100., color='k', lw=2)
        plt.plot(x, parabola(x, par0ms, par1ms, par2ms) * 100., color='k', lw=2)
    plt.grid(True)
    plt.legend(loc="best")
    plt.xlabel('Theta [rad]')
    plt.ylabel('Resolution @ Tl208 peak [%]')
    plt.xlim(xmin=(par2ms-0.15), xmax=(par2ms+0.15))
    plt.ylim(ymin=1.0, ymax=2.0)
    plt.savefig(fOUT[:-4] + "_zoom" + fOUT[-4:])
    plt.xlim(xmin=0.0, xmax=1.5)
    plt.ylim(ymin=1.2, ymax=5.5)
    plt.savefig(fOUT, bbox_inches='tight')
    plt.clf()
    plt.close()
    return


# deprecated spectrum
def get_energy_spectrum(args, files):
    import h5py
    entry = []
    for filename in files:
        f = h5py.File(str(filename), 'r')
        temp=np.array(f.get('trueEnergy'))
        for i in range(len(temp)):
            entry.append(temp[i])
        f.close()
    hist, bin_edges = np.histogram(entry, bins=210, range=(500,3000), density=True)
    plt.plot(bin_edges[:-1], hist)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=500, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    hist_inv=np.zeros(hist.shape)
    for i in range(len(hist)):
        try:
            hist_inv[i]=1.0/float(hist[i])
        except:
            pass
    hist_inv = hist_inv / hist_inv.sum(axis=0, keepdims=True)
    plt.plot(bin_edges[:-1], hist_inv)
    plt.gca().set_yscale('log')
    plt.grid(True)
    plt.xlabel('Energy [keV]')
    plt.ylabel('Weight')
    plt.xlim(xmin=500, xmax=3000)
    plt.savefig(args.folderOUT + 'spectrum_inverse.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return (hist_inv, bin_edges[:-1])

# input energy spectrum
def get_energy_spectrum_mixed(args, files, add):
    import h5py
    entry, hist, entry_mixed = {}, {}, []
    for source in args.sources:
        entry[source] = []
        for filename in files[source]:
            f = h5py.File(str(filename), 'r')
            temp = np.array(f.get('trueEnergy')).tolist()
            f.close()
            entry[source].extend(temp)
        entry_mixed.extend(entry[source])
    num_counts =  float(len(entry_mixed))
    hist_mixed, bin_edges = np.histogram(entry_mixed, bins=500, range=(0, 5000), density=False)
    bin_width = ((bin_edges[1] - bin_edges[0]) / 2.0)
    plt.plot(bin_edges[:-1] + bin_width, np.array(hist_mixed)/num_counts, label="combined", lw = 2, color='k')

    for source in args.sources:
        if len(entry[source])==0: continue
        label = args.label[source]
        hist[source], bin_edges = np.histogram(entry[source], bins=500, range=(0,5000), density=False)
        plt.plot(bin_edges[:-1] + bin_width, np.array(hist[source])/num_counts, label=label)
        # print "%s\t%s\t%i" % (add, source , len(entry[source]))
    plt.axvline(x=2614.5, lw=2, color='k')
    plt.gca().set_yscale('log')
    plt.gcf().set_size_inches(10,5)
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.xlabel('Energy [keV]')
    plt.ylabel('Probability')
    plt.xlim(xmin=500, xmax=3500)
    plt.ylim(ymin=(1.0/1000000), ymax=1.0)
    plt.savefig(args.folderOUT + 'spectrum_mixed_' + add + '.pdf', bbox_inches='tight')
    plt.close()
    plt.clf()
    return


#Label line with line2D label data
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.get_axes()
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        # ang = degrees(atan2(dy,dx))
        # ang = degrees(dy/dx)
        ang = 60.

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_axis_bgcolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)


def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].get_axes()
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)

# ----------------------------------------------------------
# Final Plots
# ----------------------------------------------------------
def final_plots(folderOUT, obs):
    if obs == {} :
        print 'final plots \t save.p empty'
        return
    obs_sort, epoch = {}, []
    key_list = list(set( [ key for key_epoch in obs.keys() for key in obs[key_epoch].keys() if key not in ['E_true', 'E_pred']] ))
    for key in key_list:
        obs_sort[key]=[]

    for key_epoch in obs.keys():
        epoch.append(int(key_epoch))
        for key in key_list:
            try:
                obs_sort[key].append(obs[key_epoch][key])
            except KeyError:
                obs_sort[key].append(0.0)

    order = np.argsort(epoch)
    epoch = np.array(epoch)[order]

    for key in key_list:
        obs_sort[key] = np.array(obs_sort[key])[order]
        if key not in ['loss', 'val_loss', 'mean_absolute_error', 'val_mean_absolute_error']:
            obs_sort[key] = np.array([x if type(x) in [np.ndarray,tuple] and len(x)==2 else (x,0.0) for x in obs_sort[key]])

    try:
        plt.plot(epoch, obs_sort['loss'], label='Training set')
        plt.plot(epoch, obs_sort['val_loss'], label='Validation set')
        plt.xlabel('Training time [epoch]')
        plt.ylabel('Loss [keV$^2$]')
        plt.grid(True, which='both')
        # plt.ylim(ymin=7.e2, ymax=2.e4)
        plt.gca().set_yscale('log')
        plt.legend(loc="best")
        plt.savefig(folderOUT + 'loss.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()

        plt.plot(epoch, obs_sort['mean_absolute_error'], label='Training set')
        plt.plot(epoch, obs_sort['val_mean_absolute_error'], label='Validation set')
        plt.grid(True)
        # plt.ylim(ymin=0.0, ymax=100.0)
        plt.legend(loc="best")
        plt.xlabel('Training time [epoch]')
        plt.ylabel('Mean absolute error [keV]')
        plt.savefig(folderOUT + 'mean_absolute_error.pdf', bbox_inches='tight')
        plt.clf()
        plt.close()
    except:
        print 'no loss / mean_err plot possible'

    plt.errorbar(epoch, obs_sort['peak_pos'][:,0], xerr=0.5, yerr=obs_sort['peak_pos'][:,1], fmt="none", lw=2)
    plt.axhline(y=2614.5, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Energy [keV]')
    plt.savefig(folderOUT + '2prediction-spectrum/ZZZ_Peak.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['peak_sig'][:,0], xerr=0.5, yerr=obs_sort['peak_sig'][:,1], fmt="none", lw=2)
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Reconstructed Photopeak Width [keV]')
    plt.savefig(folderOUT + '2prediction-spectrum/ZZZ_Width.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_pos'][:, 0], xerr=0.5, yerr=obs_sort['resid_pos'][:, 1], fmt="none", lw=2)
    plt.axhline(y=0, lw=2, color='black')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Offset [keV]')
    plt.savefig(folderOUT + '4residual-histo/ZZZ_Offset.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    plt.errorbar(epoch, obs_sort['resid_sig'][:, 0], xerr=0.5, yerr=obs_sort['peak_sig'][:, 1], fmt="none", lw=2)
    plt.xlabel('Epoch')
    plt.ylabel('Residual Width [keV]')
    plt.grid(True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.savefig(folderOUT + '4residual-histo/ZZZ_Width.pdf', bbox_inches='tight')
    plt.clf()
    plt.close()

    return

# ----------------------------------------------------------
# Math Functions
# ----------------------------------------------------------
def gauss(x, A, mu, sigma, off):
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2)) + off


def gauss_zero(x, A, mu, sigma):
    return gauss(x, A, mu, sigma, 0.0)


def erf(x, mu, sigma, B):
    import scipy.special
    return B * scipy.special.erf((x - mu) / (np.sqrt(2) * sigma)) + abs(B)


def shift(a, b, mu, sigma):
    return np.sqrt(2./np.pi)*float(b)/a*sigma


def gaussErf(x, A, mu, sigma, B):
    return gauss_zero(x, mu=mu, sigma=sigma, A=A) + erf(x, B=B, mu=mu, sigma=sigma)


def get_weight(Y, hist, bin_edges):
    return hist[np.digitize(Y, bin_edges) - 1]


def round_down(num, divisor):
    return num - (num%divisor)


def parabola(x, par0, par1, par2):
    return par0 + par1 * ((x - par2) ** 2)