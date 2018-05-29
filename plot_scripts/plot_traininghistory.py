import numpy as np
import matplotlib.pyplot as plt


def plot_traininghistory(args):

    # folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/180525-1626/'

    data = np.loadtxt(args.folderOUT + 'training_history.csv', delimiter='\t', skiprows=1)

    plt.plot(data[0:59, 0], data[0:59, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(args.folderOUT + 'plot_loss.png', bbox_inches='tight')

    plt.clf()
    plt.plot(data[0:59, 0], data[0:59, 2])
    plt.savefig(args.folderOUT + 'plot_meanabserror.png', bbox_inches='tight')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.show()
    plt.draw()

