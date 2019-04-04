#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import os
from sys import path
from sys import exit
path.append('/home/hpc/capm/sn0515/UVWireRecon')

from utilities import generator as gen

def main():
    folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/Data/bb0n_WFs_Uni_MC/'
    folderOUT = '/home/vault/capm/mppi053h/Master/UV-wire/Waveforms/'
    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
    number = 50

    print
    print len(files)
    # print files[0]['wf'].shape

    # sys.exit()

    # generator = gen.generate_batches_from_files(files, 1, class_type='energy_and_position', f_size=None, yield_mc_info=False, multiplicity='SS', inputImages='UV')
    generator = gen.generate_batches_from_files(files, 1, class_type='energy_and_UV_position', f_size=None, yield_mc_info=1, multiplicity='SS', inputImages='UV')
    # for idx in xrange(number):
    idx = 0
    max = 0
    min = 200

    while idx < number:

        # wf, _ = generator.next()
        # plot_waveforms(np.asarray(wf), idx, folderOUT)

        # print '>>'
        wf, Y_TRUE, EVENT_INFO = generator.next()

        # print wf
        # print EVENT_INFO.keys()

        # wirecheck_data_1 = np.zeros((7, 200))
        # wirecheck_data_2 = np.zeros((7, 200))
        #
        # if EVENT_INFO['MCPosZ'] < 0:
        #     if EVENT_INFO['CCNumberUWires'][0] == 1:
        #         if EVENT_INFO['CCIsSS'][0] == 1:
        #             # print EVENT_INFO['PCDDepositChannel']
        #             if EVENT_INFO['PCDDepositChannel'][0, 0] != 999.0:
        #                 if EVENT_INFO[]
        #                     wirecheck_data =

                    # if max < EVENT_INFO['PCDDepositChannel'][0, 0]: max = EVENT_INFO['PCDDepositChannel'][0, 0]
                    # if min > EVENT_INFO['PCDDepositChannel'][0, 0]: min = EVENT_INFO['PCDDepositChannel'][0, 0]
        print 'plot waveform \t', idx
        plot_waveforms(np.asarray(wf), idx, folderOUT)
        idx += 1

    # plot_wirecheck(wf, idx, folderOUT, EVENT_INFO['PCDDepositChannel'][0, 0], EVENT_INFO['CCCollectionTime'][0, 0])

    print 'max: ', max
    print 'min: ', min

    return

def plot_wirecheck(wf, idx, folderOUT, depositchannel, time):
    time = range(0, 400)

    # xs_i = np.swapaxes(xs_i, 0, 1)
    wf = np.swapaxes(wf, 1, 2)
    wf = np.swapaxes(wf, 2, 3)

    plt.clf()
    # make Figure
    fig, axarr = plt.subplots(2, 2)

    # set size of Figure
    fig.set_size_inches(w=20., h=8.)

    for i in xrange(wf.shape[0]):
        if i == 0 : x = 1; y = 0
        elif i == 1: x = 0; y = 0
        elif i == 2: x = 1; y = 1
        elif i == 3: x = 0; y = 1
        axarr[x, y].set_xlim([0.0, 400])
        axarr[x, y].set_ylim([-40, 780])
        plt.setp(axarr[x, y].get_yticklabels(), visible=False)
        for j in xrange(wf.shape[2]):
            axarr[x, y].plot(time, wf[i, : , j, 0, 0] + 20. * j, color='k')

    axarr[1, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[0, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[1, 1].set_xlabel(r'Time [$\mu$s]')
    axarr[1, 0].set_xlabel(r'Time [$\mu$s]')

    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    fig.savefig(folderOUT + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

def plot_waveforms(wf, idx, folderOUT):
    time = range(0, 2048)

    # xs_i = np.swapaxes(xs_i, 0, 1)
    wf = np.swapaxes(wf, 1, 2)
    wf = np.swapaxes(wf, 2, 3)

    plt.clf()
    # make Figure
    fig, axarr = plt.subplots(2, 2)

    # set size of Figure
    fig.set_size_inches(w=20., h=8.)

    for i in xrange(wf.shape[0]):
        if i == 0 : x = 1; y = 0
        elif i == 1: x = 0; y = 0
        elif i == 2: x = 1; y = 1
        elif i == 3: x = 0; y = 1
        axarr[x, y].set_xlim([0.0, 2048])
        axarr[x, y].set_ylim([-40, 780])
        plt.setp(axarr[x, y].get_yticklabels(), visible=False)
        for j in xrange(wf.shape[2]):
            axarr[x, y].plot(time, wf[i, : , j, 0, 0] + 20. * j, color='k')

    axarr[1, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[0, 0].set_ylabel(r'Amplitude + offset [a.u.]')
    axarr[1, 1].set_xlabel(r'Time [$\mu$s]')
    axarr[1, 0].set_xlabel(r'Time [$\mu$s]')

    alpha = 0.3

    axarr[1, 0].fill_between([0,1000], -40, 780, color='red', alpha=alpha)
    axarr[1, 0].fill_between([1350, 2048], -40, 780, color='red', alpha=alpha)
    axarr[0, 0].fill_between([0,1000], -40, 780, color='red', alpha=alpha)
    axarr[0, 0].fill_between([1350, 2048], -40, 780, color='red', alpha=alpha)
    axarr[1, 1].fill_between([0,1000], -40, 780, color='red', alpha=alpha)
    axarr[1, 1].fill_between([1350, 2048], -40, 780, color='red', alpha=alpha)
    axarr[0, 1].fill_between([0,1000], -40, 780, color='red', alpha=alpha)
    axarr[0, 1].fill_between([1350, 2048], -40, 780, color='red', alpha=alpha)


    plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    fig.savefig(folderOUT + str(idx) + '.png', bbox_inches='tight')
    plt.close()
    plt.clf()
    return

# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
