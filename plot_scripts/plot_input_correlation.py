import numpy as np
# import matplotlib as mpl
# mpl.use('PDF')
import matplotlib.pyplot as plt
import os
import utilities.generator as gen


def main():
    # folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/UniformGamma_ExpWFs_MC_SS/'
    folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/Data/EnergyCorrectionNewNew/'
    folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/Dummy/'

    files = [os.path.join(folderIN, f) for f in os.listdir(folderIN) if os.path.isfile(os.path.join(folderIN, f))]
    print files
    number = gen.getNumEvents(files)
    generator = gen.generate_batches_from_files(files, 1, class_type='energy_and_position', f_size=None, yield_mc_info=True)

    for idx in xrange(number):
        if idx % 500 == 0:
            print idx, 'of', number
        # wf_temp, ys_temp, event_info = generator.next()
        ys_temp, event_info = generator.next()
        z_position = event_info['MCPosZ'][:, 0].reshape((1, 1))
        ys_temp = np.append(ys_temp, z_position, axis=1)

        if idx == 0:
            ys = ys_temp
        else:
            ys = np.concatenate((ys, ys_temp))

    plot_input_correlations(ys, folderOUT)
    return


def plot_input_correlations(ys, folderOUT):

    from pandas.tools.plotting import scatter_matrix
    from pandas import DataFrame

    ys_data = DataFrame(ys, columns=['Energy', 'x-Position', 'y-Position', 'Time', 'z-Position'])

    sm = scatter_matrix(ys_data, figsize=(25, 25), alpha=0.15, hist_kwds={'bins': 20})     # diagonal='kde')

    for s in sm.reshape(-1):
        s.xaxis.label.set_size(40)
        s.yaxis.label.set_size(40)
        plt.setp(s.yaxis.get_majorticklabels(), 'size', 20)
        plt.setp(s.xaxis.get_majorticklabels(), 'size', 20)

    plt.show()
    plt.draw()
    raw_input('')

    # plt.savefig(folderOUT + 'Correlation_matrix' + '.png')

    return


# ----------------------------------------------------------
# Program Start
# ----------------------------------------------------------
if __name__ == '__main__':
    main()
