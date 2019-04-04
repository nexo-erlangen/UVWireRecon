#!/usr/bin/python
import numpy as np
from scipy.spatial import cKDTree
import time
# Fuer 3D-Plot
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


from generator import *
import h5py
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import os


def main():
    # Erzeugung von Zufallsdaten in der Form, die hoffentlich
    # auch du fuer deine richtigen Daten hast. Erzeugt werden
    # N zufaellige Werte, wobei Positionen auch doppelt vor-
    # kommen duerfen.

#-----------

    # filename = '/home/vault/capm/mppi053h/Master/UV-wire/Data/ClusteringTestbb0nE/ED_Source_bb0nE_0.hdf5'
    '/home/vault/capm/mppi053h/Master/UV-wire/Data/GammaExp_WFs_Uni_MC_SS+MS/'
    histo = []
    for filename in os.listdir('/home/vault/capm/mppi053h/Master/UV-wire/Data/GammaExp_WFs_Uni_MC_SS+MS/'):
        y_dict = h5py.File('/home/vault/capm/mppi053h/Master/UV-wire/Data/GammaExp_WFs_Uni_MC_SS+MS/' + str(filename), "r")
        print filename
        # print y_dict.keys()

        histoDBSCAN = []
        histoCC = []

        number_timesteps = 3
        eventnumber = 0

        train_y = np.zeros((8000, number_timesteps, 4), dtype='float32')

        for eventnumber in xrange(8000):
            if (eventnumber+1) %1000 == 0: print '>>'
            numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])

            # TODO: in x, y, z clustern; dann in u, v, z umrechnen; PCDs mit Depositchannel < 0 NACH clustern
            # TODO: wegwerfen um ihr Energie nicht fuer Berechnung der Cluster Energie zu verwenden
            x, y, z, energy, depositChannel = y_dict['MCPosX'][eventnumber][0:numPCDs], y_dict['MCPosY'][eventnumber][
                                                                                        0:numPCDs], y_dict['MCPosZ'][
                                                                                                        eventnumber][
                                                                                                    0:numPCDs], \
                                              y_dict['MCEnergy'][eventnumber][0:numPCDs], y_dict['MCDepositChannel'][
                                                                                              eventnumber][0:numPCDs]
            X = []

            for i in range(len(x)):
                X.append([x[i], y[i], z[i]])

            cluster_distance = 5  # in mm
            clustering = DBSCAN(eps=cluster_distance, min_samples=1).fit(X)
            label = clustering.labels_

            x_mean, y_mean, z_mean, energy_sum = [], [], [], []

            energy_sum2 = []

            histo.append(max(label) + 1)

            for j in range(max(label) + 1):

                mask = np.logical_and(depositChannel >= 0.0, label == j)

                x_temp = x[mask]
                y_temp = y[mask]
                z_temp = z[mask]
                energy_temp = energy[mask]

                # if j==5:
                #     counter_5+=1

                if len(energy_temp) != 0:
                    x_mean.append(np.average(x_temp, weights=energy_temp))
                    y_mean.append(np.average(y_temp, weights=energy_temp))
                    z_mean.append(np.average(z_temp, weights=energy_temp))
                    energy_sum.append(np.sum(energy_temp))
                    energy_sum2.append(np.sum(energy_temp))
                    test2 = np.sum(energy_temp)
                    # print '>>>>>', len(energy_temp)

            # print test2

            while len(x_mean) < number_timesteps:
                x_mean.append(0.0)
                y_mean.append(0.0)
                z_mean.append(0.0)
                energy_sum.append(0.0)

            # TODO erst nach sortierung beschneiden sonst wird evtl groesster cluster weggeworfen
            if len(x_mean) > number_timesteps:
                x_mean = x_mean[:5]
                y_mean = y_mean[:5]
                z_mean = z_mean[:5]
                energy_sum = energy_sum[:5]

            x_mean = np.array(x_mean)
            y_mean = np.array(y_mean)
            z_mean = np.array(z_mean)
            energy_sum = np.array(energy_sum)

            # x_mean, y_mean to u, v
            if [z_mean > 0.0]:
                u_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                v_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
            else:
                u_mean = 0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean
                v_mean = -0.5 * x_mean + 0.5 * np.sqrt(3.0) * y_mean

            # u_mean = normalize(u_mean, 'U')
            # v_mean = normalize(v_mean, 'V')
            # z_mean = normalize(z_mean, 'Z')
            # energy_sum = normalize(energy_sum, 'energy')
            # print energy_sum

            dtype = [('energy', float), ('U', float), ('V', float), ('Z', float)]
            values = [(energy_sum[i], u_mean[i], v_mean[i], z_mean[i]) for i in range(number_timesteps)]

            target = np.array(values, dtype=dtype)
            target = np.sort(target, order='energy')


            for timestep in range(number_timesteps):
                train_y[eventnumber][timestep] = [
                    normalize(target[number_timesteps - 1 - timestep]['energy'], 'energy'),
                    normalize(target[number_timesteps - 1 - timestep]['U'], 'U'),
                    normalize(target[number_timesteps - 1 - timestep]['V'], 'V'),
                    normalize(target[number_timesteps - 1 - timestep]['Z'], 'Z')]

    print np.histogram(histo)
'''
        numPCDs = int(y_dict['MCNumberPCDs'][eventnumber])
        numCCs = int(y_dict['CCNumberClusters'][eventnumber])
        # print numPCDs, '\t', data['CCNumberClusters'][eventnumber]
        # print '---------'
        # print 'Event number:', eventnumber

        x, y, z, energy, depositChannel = y_dict['MCPosX'][eventnumber][0:numPCDs], y_dict['MCPosY'][eventnumber][0:numPCDs], y_dict['MCPosZ'][eventnumber][0:numPCDs], y_dict['MCEnergy'][eventnumber][0:numPCDs], y_dict['MCDepositChannel'][eventnumber][0:numPCDs]
        x_cc, y_cc, z_cc, energy_cc = y_dict['CCPosX'][eventnumber][0:numCCs], y_dict['CCPosY'][eventnumber][0:numCCs], y_dict['CCPosZ'][eventnumber][0:numCCs], y_dict['CCCorrectedEnergy'][eventnumber][0:numCCs]

        # print 'PCDs:'
        print x
        print y
        print z
        print depositChannel

        X = []
        x_noDeposit, y_noDeposit, z_noDeposit = [], [], []

        for i in range(len(x)):
            X.append([x[i], y[i], z[i]])
            if depositChannel[i] == -999:
                x_noDeposit.append(x[i])
                y_noDeposit.append(y[i])
                z_noDeposit.append(z[i])

        clustering = DBSCAN(eps=5, min_samples=1).fit(X)
        label = clustering.labels_

        x_mean, y_mean, z_mean, energy_sum = [], [], [], []
        for j in range(max(label) + 1):
            mask = label == j
            x_temp = x[mask]
            y_temp = y[mask]
            z_temp = z[mask]
            energy_temp = energy[mask]

            x_mean.append(np.average(x_temp, weights=energy_temp))
            y_mean.append(np.average(y_temp, weights=energy_temp))
            z_mean.append(np.average(z_temp, weights=energy_temp))
            energy_sum.append(sum(energy_temp))

        # print 'Cluster: ', max(clustering.labels_) + 1
        # print 'Charge Cluster:', numCCs
        # print x_mean, '\n', y_mean, '\n', z_mean, '\n', energy_sum
        # print 'CC energy:', energy_cc

        # print '-------'
        # energy_limit = 270
        # energy_sum = np.array(energy_sum)
        # mask = energy_sum > energy_limit
        # print np.count_nonzero(mask)
            # print eventnumber, '\t', max(clustering.labels_) + 1
        # print energy_sum

        histoDBSCAN.append(np.count_nonzero(mask))
        histoCC.append(numCCs)

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection='3d')

        ax3.scatter(x, y, z, c=clustering.labels_, alpha=0.4), #s=energy#, edgecolor='none')
        ax3.scatter(x_noDeposit, y_noDeposit, z_noDeposit, c='blue')
        ax3.scatter(x_cc, y_cc, z_cc, edgecolor='black',  facecolor='none') #s=energy_cc,
        ax3.scatter(x_mean, y_mean, z_mean, c='black',  marker='x')    #s=energy_sum,          # c=range(max(label) + 1), s=energy_sum, marker='X')
        ax3.set_xlabel('x [mm]')
        ax3.set_ylabel('y [mm]')
        ax3.set_zlabel('z [mm]')
        ax3.set_title('DBSCAN')

        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min(), x_cc.max() - x_cc.min(), y_cc.max() - y_cc.min(), z_cc.max() - z_cc.min()]).max() / 2.0

        mid_x = (np.array([x.max(), x_cc.max()]).max() + np.array([x.min(), x_cc.min()]).min()) * 0.5
        mid_y = (np.array([y.max(), y_cc.max()]).max() + np.array([y.min(), y_cc.min()]).min()) * 0.5
        mid_z = (np.array([z.max(), z_cc.max()]).max() + np.array([z.min(), z_cc.min()]).min()) * 0.5
        ax3.set_xlim(mid_x - max_range - 2, mid_x + max_range + 2)
        ax3.set_ylim(mid_y - max_range - 2, mid_y + max_range + 2)
        ax3.set_zlim(mid_z - max_range - 2, mid_z + max_range + 2)

        fig3.show()
        # fig3.savefig('/home/vault/capm/mppi053h/Master/UV-wire/Clustering/' + 'ClusteredEvent_' + str(eventnumber), bbox_inches='tight', dpi=300)

        raw_input('')

        fig3.clf()
        eventnumber += 1


    # plt.hist([histoDBSCAN, histoCC], range=[0,10], bins=11, label=['DBSCAN', 'CC'])
    # plt.title('Anzahl DBSCAN Cluster > ' + str(energy_limit) + 'keV')
    # plt.legend()
    # plt.show()

'''

def randomUniform(N):
    # Positionen der jeweiligen Pixel, nun aufgespalten in
    # x- und y-Werte
    x, y = np.random.randint(0, 255, N), np.random.randint(0, 255, N)

    # Zugehoerige Zeitangaben in [ns]
    t = np.random.randint(0, N // 10, N)

    # Zugehoerige ToT-Werte in [arbitrary units]
    tot = np.random.randint(1, 1000, N)

    return x, y, t, tot


def randomGauss(N, M, rStd=5, tStd=10):
    # Erzeuge Positionen der Zentren
    xCenter, yCenter, tCenter = np.random.randint(0, 255, M), np.random.randint(0, 255, M), np.random.randint(0,
                                                                                                              N // 10,
                                                                                                              N)

    # Listen fuer finale Punkte
    xTotal, yTotal, tTotal = [], [], []

    # Zugehoerige ToT-Werte in [arbitrary units]
    tot = np.random.randint(1, 1000, N)

    # Erzeuge Punkte um die Zentren. Jedes Zentrum ist
    # von der selben Anzahl an Punkten umgeben.
    # Gaussverteilung mit folgenden Standardabweichungen:
    xStd, yStd = rStd, rStd
    tStd = 10
    for i in range(M):
        x, y, t = xCenter[i], yCenter[i], tCenter[i]
        xRand = [int(n) for n in np.random.normal(x, xStd, N // M)]
        yRand = [int(n) for n in np.random.normal(y, yStd, N // M)]
        tRand = [int(n) for n in np.random.normal(t, tStd, N // M)]

        xTotal += list(xRand)
        yTotal += list(yRand)
        tTotal += list(tRand)

    return xTotal, yTotal, tTotal, tot



def findNN(x, y, z, tot, rDist, tDist):
    # maxDist gibt die raeumliche Distanz von Pixeln an,
    # innerhalb der diese zusammengefasst werden

    # kd-Tree Suche nach naechsten Nachbarn findet um Kugel
    # statt. Daher muss Zeit entsprechend skaliert werden.
    # D.h. rDist und tDist muessen gleiche Einheit besitzen
    # t = [float(time) / tDist * rDist for time in t]

    # Erzeuge kd-Tree mit drei Dimensionen: x, y und z
    # Kombiniere dazu benoetigte Werte zu einem Datensatz
    data = np.array(zip(x, y, z))
    tree = cKDTree(data)

    # Finde alle naechsten Nachbarn fuer jedes Event und
    # speichere deren Indizes in Liste
    res = tree.query_ball_tree(tree, r=rDist)


    # Konvertiere die Unterlisten der Nachbarn zu einem Tuple,
    # da dieses hashable ist und damit doppelte Eintraege
    # einfach ueber Erzeugung eines Sets entfernt werden koennen
    # res = list(set([tuple(r) for r in res]))
    res = set([tuple(r) for r in res])



    # In jeder Liste der Nachbarn ist der gepruefte Pixel selbst
    # vorhanden. Ist also die Dimensionalitaet = 1, so besitzt
    # dieser Pixel keine Nachbarn. Andernfalls fasse gefundene
    # Nachbarn in einem Pixel zusammen
    xNew, yNew = [], []
    zNew, energyNew = [], []
    for r in res:
        if len(r) > 1:
            # Suche Zeitwerte fuer gefundene Indizes
            tList = [z[idx] for idx in r]

            # Suche Wert mit kleinster Zeit
            tMinIdx = np.argmin(tList)

            # Summiere ToT-Werte auf
            totList = [tot[idx] for idx in r]
            totSum = sum(totList)

            # Setze zuerst getroffenen Pixel als neuen,
            # kombinierten Pixel
            idx = r[tMinIdx]

            # ToT-Wert ist Summe aller getroffener Pixel
            energyNew.append(totSum)

        else:
            idx = r[0]

            # ToT-Wert aendert sich nicht
            energyNew.append(tot[idx])

        # Schreibe Daten in neuen Pixel
        xNew.append(x[idx]), yNew.append(y[idx])
        zNew.append(z[idx])

    return xNew, yNew, zNew, energyNew


if __name__ == '__main__':
    main()

