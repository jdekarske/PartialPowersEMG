import h5py
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib import gridspec


plotflag = False
FFTpts = 1000  # number of fft pts


def FFT(signal, n=1000):
    chan = np.shape(emg)[0]
    F = np.fft.fft(emg, n=n)
    F = F.reshape((chan, n))
    freq = np.fft.fftfreq(n, d=1 / 2000)
    power = np.square(F.real) / n
    posfreq = freq[:int(n / 2)]
    pospower = power[:, :int(n / 2)]
    return posfreq, pospower


def corrplot(ax, correlation, key):
    cax = ax.matshow(correlation, interpolation='gaussian', cmap='coolwarm', extent=[posfreq[0], posfreq[-1], posfreq[-1], posfreq[0]])
    ax.set_title(key)
    freq1 = [40, 60]
    freq2 = [80, 100]
    for c in freq1:
        ax.axvline(x=c, color='y', linestyle='--', linewidth=3)
        ax.axhline(y=c, color='y', linestyle='--', linewidth=3)
    for c in freq2:
        ax.axvline(x=c, color='g', linestyle='--', linewidth=3)
        ax.axhline(y=c, color='g', linestyle='--', linewidth=3)
    ax.set_ylim(bottom=300)
    ax.set_xlim(right=300)
    return cax

alltrials = {}
directory = os.path.abspath('data/2')
for subdir, _, files in os.walk(directory):
    # check if empty
    if not files:
        continue
    h5filename = os.path.join(subdir, files[1])
    csvfilename = os.path.join(subdir, files[0])

    # load the parameters from the csv
    csvlist = []
    with open(csvfilename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            csvlist.append(row)

    # grab the unique target combos and form a dictionary, yuck
    targetlist, targetcount = np.unique([eval(target['active_targets'].replace('  ', ',')) for target in csvlist], return_counts=True, axis=0)
    taskdata = {}
    for k, target in enumerate(targetlist):
        taskdata[str(target)] = []

    # load the emg data
    f = h5py.File(h5filename, 'r')
    for i, (key, trial) in enumerate(f.items()):
        if csvlist[i]['timeout'] == 'True':
            continue
        emg = np.array(trial)
        posfreq, pospower = FFT(emg, n=FFTpts)

        # send to the dictionary for this task
        activetarget = ''.join(csvlist[i]['active_targets'].split(" ", 2))  # there are extra spaces, this is reaalll ugly
        taskdata[activetarget].append(pospower)

        if plotflag:
            plt.figure()
            plt.subplot(121)
            plt.plot(emg.flatten())
            plt.subplot(122)
            plt.plot(posfreq, pospower.flatten())
            plt.ylim((0, .06))
            plt.xlim(right=400)
            xfreqs = [40, 60, 80, 100]
            for xc in xfreqs:
                plt.axvline(x=xc, color='m')
            plt.show()
    # combine chunks in dictionary
    taskcorrelation = {}
    for i, (key, value) in enumerate(taskdata.items()):
        if not value:
            continue
        tasktrials = np.vstack(value).T
        alltrials[key] = np.hstack((alltrials[key], tasktrials)) if key in alltrials.keys() else tasktrials

# calculate the correlation
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html normalized covariance matrix (Pearson)
fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i, (key, trial) in enumerate(alltrials.items()):
    correlation = np.corrcoef(trial)
    cax = corrplot(axs.flatten()[i], correlation, key)
fig.colorbar(cax)
plt.show()