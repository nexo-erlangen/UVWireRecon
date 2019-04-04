import numpy as np
import matplotlib.pyplot as plt


def plot_traininghistory(self):
    try:
        data = np.loadtxt(self.args.folderOUT + 'log_train.txt', delimiter='\t', skiprows=1)
        folder = self.args.folderOUT

        x = np.arange(len(data)) / 12000. * 500

        fig, ax = plt.subplots()
        ax.semilogy(x, data[:, 2], label='Training', color='blue')
        ax.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
        plt.legend(loc="best")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folder + 'loss_run_log.png', bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(x, data[:, 2], label='Training', color='blue')
        ax.plot(x, data[:, 4], label='Validation', alpha=0.5, color='green')
        plt.legend(loc="best")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folder + 'loss_run.png', bbox_inches='tight', dpi=300)
        plt.clf()
        plt.close()

        if self.args.resume:
            while folder[:-15] != '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/':
                folder = folder[:-15]

                # print data.shape

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
        plt.close()

    except:
        plt.clf()
        plt.close()
        print 'plot_trainingshistory not possible'

'''
# try:
folder = '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/190206-1314-19/190207-1335-29/190208-1000-21/190209-1133-59/'
data = np.loadtxt(folder + 'log_train.txt', delimiter='\t', skiprows=1)
# folder = self.args.folderOUT

x = np.arange(len(data)) / 12000. * 500

fig, ax = plt.subplots()
ax.semilogy(x, data[:, 2], label='Training', color='blue')
ax.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
plt.legend(loc="best")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(folder + 'loss_run_log.png', bbox_inches='tight', dpi=300)
plt.clf()

fig, ax = plt.subplots()
ax.plot(x, data[:, 2], label='Training', color='blue')
ax.plot(x, data[:, 4], label='Validation', alpha=0.5, color='green')
plt.legend(loc="best")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(folder + 'loss_run.png', bbox_inches='tight', dpi=300)
plt.clf()


while folder[:-15] != '/home/vault/capm/mppi053h/Master/UV-wire/TrainingRuns/':
    folder = folder[:-15]

    # print data.shape

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

# fig, ax = plt.subplots()
# plt.figure(figsize=(16, 9))
plt.semilogy(x, data[:, 2], label='Training', color='blue')
plt.semilogy(x, data[:, 4], label='Validation', alpha=0.5, color='green')
plt.legend(loc="best")
plt.xlabel('Time [epoch]')
plt.ylabel('Loss')

plt.savefig(folder + 'loss_total.png', bbox_inches='tight', dpi=300)
plt.clf()

'''

