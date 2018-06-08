import numpy as np
import matplotlib.pyplot as plt


def plot_traininghistory(folderOUT):

    data = np.loadtxt(folderOUT + 'training_history.csv', delimiter='\t', skiprows=1)

    try:

        plt.plot(data[:, 0], data[:, 1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight')

        plt.clf()
        plt.plot(data[:, 0], data[:, 2])
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(folderOUT + 'plot_meanabserror.png', bbox_inches='tight')
        # plt.show()
        # plt.draw()
    except:
        plt.plot(data[0], data[1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(folderOUT + 'plot_loss.png', bbox_inches='tight')

        plt.clf()
        plt.plot(data[0], data[2])
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')
        plt.savefig(folderOUT + 'plot_meanabserror.png', bbox_inches='tight')

"""
folderIN = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/180601-1441/' #180606-1511-44/180607-1014-05'
folderOUT = '/home/vault/capm/sn0515/PhD/DeepLearning/UV-wire/TrainingRuns/180601-1441/'

data1 = np.genfromtxt(folderIN + 'training_history.csv', delimiter='\t')   # , skiprows=1)
data2 = np.loadtxt(folderIN + '180606-1511-44/' + 'training_history.csv', delimiter='\t', skiprows=1)
data3 = np.loadtxt(folderIN + '180606-1511-44/180607-1014-05/' + 'training_history.csv', delimiter='\t', skiprows=1)


print data1.shape
print data2.shape
print data3.shape

data = np.append(data1, data2, axis=0)
data = np.append(data, data3, axis=0)

print data.shape

plt.semilogy(data[:, 0], data[:, 1])
plt.semilogy(data[:, 0], data[:, 3])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(folderOUT + 'total_loss.png', bbox_inches='tight')


plt.clf()
plt.semilogy(data[:, 0], data[:, 2])
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.savefig(folderOUT + 'total_mae.png', bbox_inches='tight')

plt.clf()
plt.semilogy(data[:, 0], data[:, 3])
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.savefig(folderOUT + 'total_val_loss.png', bbox_inches='tight')
"""
