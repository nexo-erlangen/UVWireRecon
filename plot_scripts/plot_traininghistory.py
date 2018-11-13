import numpy as np
import matplotlib.pyplot as plt


def plot_traininghistory(self):

    data = np.loadtxt(self.args.folderOUT + 'log_train.txt', delimiter='\t', skiprows=1)
    folder = self.args.folderOUT

    x = np.arange(len(data)) / 12000. * 500

    fig, ax = plt.subplots()
    ax.semilogy(x, data[:, 2], label='Training', color='blue')
    ax.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
    plt.legend(loc="best")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(folder + 'loss_run.png', bbox_inches='tight', dpi=300)
    plt.clf()

    if self.args.resume:
        while folder[:-15] != '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/':
            folder = folder[:-15]
            data2 = np.loadtxt(folder + 'log_train.txt', delimiter='\t', skiprows=1)
            data = np.append(data2, data, axis=0)

        # data = np.append(data2, data, axis=0)

    # data1 = np.genfromtxt(folderIN + 'log_train.txt', delimiter='\t')  # , skiprows=1)
    # data2 = np.loadtxt(folderIN + '181010-1034-45/' + 'log_train.txt', delimiter='\t', skiprows=1)
    # data3 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/' + 'log_train.txt', delimiter='\t', skiprows=1)
    # data4 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/181012-0958-00/' + 'log_train.txt', delimiter='\t',
    #                    skiprows=1)
    # data5 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/181012-0958-00/181016-1631-25/' + 'log_train.txt',
    #                    delimiter='\t', skiprows=1)

    x = np.arange(len(data)) / 12000. * 500

    fig, ax = plt.subplots()
    ax.semilogy(x, data[:, 2], label='Training', color='blue')
    ax.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
    plt.legend(loc="best")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(folder + 'loss_total.png', bbox_inches='tight', dpi=300)
    plt.clf()

    fig, ax = plt.subplots()
    ax.semilogy(x, data[:, 3], label='Training', color='blue')
    ax.semilogy(x, data[:, 5], label='Validation', alpha=0.5, color='green')
    plt.legend(loc="best")
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.savefig(folder + 'mae_total.png', bbox_inches='tight', dpi=300)
    plt.clf()





'''
    data = np.loadtxt(folderOUT + 'training_history.csv', delimiter='\t', skiprows=1)

    # if self.args.resume:
    #
    #     data2 = np.loadtxt(self.args.folderOUT[:-15] + 'training_history.csv', delimiter='\t', skiprows=1)
    #     print data
    #     print '++++'
    #     print data2
    #     # data = np.append(data2, data, axis=0)

    try:
        fig, ax = plt.subplots()
        ax.semilogy(data[:, 0], data[:, 1], label='Training', color='blue')
        ax.semilogy(data[:, 0], data[:, 3], label='Validation', alpha=0.5, color='green')
        plt.legend(loc="best")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight', dpi=300)
        plt.clf()


    except:
        fig, ax = plt.subplots()
        ax.semilogy(data[0], data[1], label='Loss', color='blue')
        ax.semilogy(data[0], data[3], label='Validation Loss', alpha=0.5, color='green')
        plt.legend(loc="best")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight')
        plt.clf()

'''

'''
folderIN = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/181009-1001-03/' #180606-1511-44/180607-1014-05'
# folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/180612-1614-49/'

data1 = np.genfromtxt(folderIN + 'log_train.txt', delimiter='\t')   # , skiprows=1)
data2 = np.loadtxt(folderIN + '181010-1034-45/' + 'log_train.txt', delimiter='\t', skiprows=1)
data3 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/' + 'log_train.txt', delimiter='\t', skiprows=1)
data4 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/181012-0958-00/' + 'log_train.txt', delimiter='\t', skiprows=1)
data5 = np.loadtxt(folderIN + '181010-1034-45/181011-0937-18/181012-0958-00/181016-1631-25/' + 'log_train.txt', delimiter='\t', skiprows=1)


data = np.append(data1, data2, axis=0)
data = np.append(data, data3, axis=0)
data = np.append(data, data4, axis=0)
data = np.append(data, data5, axis=0)
#
# data = data1

x = np.arange(len(data)) / 12000. * 500
# x =

fig, ax = plt.subplots()
ax.semilogy(x, data[:, 2], label='Training', color='blue')
ax.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
legend = ax.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(folderIN + 'lossAndVallos.png', bbox_inches='tight', dpi=300)
plt.clf()


# plt.clf()
# plt.semilogy(data[:, 0], data[:, 2])
# plt.xlabel('Epoch')
# plt.ylabel('Mean Absolute Error')
# plt.savefig(folderOUT + 'total_mae.png', bbox_inches='tight')
#
# plt.clf()
# plt.semilogy(data[:, 0], data[:, 3])
# plt.xlabel('Epoch')
# plt.ylabel('Validation Loss')
# plt.savefig(folderOUT + 'total_val_loss.png', bbox_inches='tight')

'''
