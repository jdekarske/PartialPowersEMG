import h5py
import os
import matplotlib.pyplot as plt
import csv
import numpy as np


directory = os.path.abspath('data/2')
for subdir, _, files in os.walk(directory):
    if not files:
        continue
    h5filename = os.path.join(subdir, files[1])
    csvfilename = os.path.join(subdir, files[0])

    csvlist = []
    with open(csvfilename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csvlist.append(row)

    f = h5py.File(h5filename, 'r')
    for key, trial in f.items():
        emg = np.array(trial)
       
        n = 1000  # number of fft pts
        chan = np.shape(emg)[0]
        F = np.fft.fft(emg, n=n)
        F = F.reshape((chan, n))
        freq = np.fft.fftfreq(n, d=1 / 2000)
        power = np.square(F.real) / n
        posfreq = freq[:int(n / 2)]
        pospower = power[:, :int(n / 2)]

        plt.figure()
        plt.subplot(121)
        plt.plot(emg.flatten())
        plt.subplot(122)
        plt.plot(posfreq, pospower.flatten())
        plt.ylim((0, .05))
        plt.xlim(right=400)
        xfreqs = [40, 60, 80, 100]
        for xc in xfreqs:
            plt.axvline(x=xc, color='m')
        plt.show()