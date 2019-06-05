import h5py
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from matplotlib import gridspec


plotflag = False
FFTpts = 1000  # number of fft pts


def FFT(signal, n=1000):
    chan = np.shape(signal)[0]
    corrected_signal = signal - np.mean(signal)
    F = np.fft.fft(corrected_signal, n=n)
    F = F.reshape((chan, n))
    freq = np.fft.fftfreq(n, d=1 / 2000)
    power = np.square(F.real) / n
    posfreq = freq[:int(n / 2)]
    pospower = power[:, :int(n / 2)]
    return posfreq, pospower


def corrplot(ax, correlation, key):
    cax = ax.matshow(correlation, cmap='coolwarm', extent=[posfreq[0], posfreq[-1], posfreq[-1], posfreq[0]]) # interpolation='gaussian',
    ax.set_title(key)
    freq1 = [40, 60]
    freq2 = [80, 100]
    for c in freq1:
        ax.axvline(x=c, color='y', linestyle='--', linewidth=2)
        ax.axhline(y=c, color='y', linestyle='--', linewidth=2)
    for c in freq2:
        ax.axvline(x=c, color='g', linestyle='--', linewidth=2)
        ax.axhline(y=c, color='g', linestyle='--', linewidth=2)
    ax.set_ylim(bottom=150)
    ax.set_xlim(right=150)
    return cax

alltrials = {}
allcounts = {}
alldist = {}
totalcount = []
directory = os.path.abspath('data/jasonold')
for subdir, _, files in os.walk(directory):
    if 'Exertion' in subdir:
        continue
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
    countdata = {}
    distdata = {}
    for k, target in enumerate(targetlist):
        taskdata[str(target)] = []
        countdata[str(target)] = []
        distdata[str(target)] = []

    # load the emg data
    f = h5py.File(h5filename, 'r')
    for i, (key, trial) in enumerate(f.items()):
        emg = np.array(trial)
        posfreq, pospower = FFT(emg, n=FFTpts)

        # send to the dictionary for this task
        activetarget = ' '.join(csvlist[i]['active_targets'].split("  "))  # there are extra spaces, this is reaalll ugly
        activetarget = ''.join(activetarget.split(" ", 1))  # there are extra spaces, this is reaalll ugly        
        pospower[0, 0] = 0  # theres a spike at zero
        taskdata[activetarget].append(pospower)

        # final distance
        finalpos = eval(csvlist[i]['final_cursor_pos'])
        targetpos = eval(",".join(activetarget.split(" ")))
        distdata[activetarget].append(np.linalg.norm(np.subtract(finalpos,targetpos)))

        if csvlist[i]['timeout'] == 'True':
            countdata[activetarget].append(0)
            totalcount.append(0)
        else:
            countdata[activetarget].append(1)
            totalcount.append(1)
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
    for i, (key, value) in enumerate(taskdata.items()):
        if not value:
            continue
        tasktrials = np.vstack(value).T
        alltrials[key] = np.hstack((alltrials[key], tasktrials)) if key in alltrials.keys() else tasktrials
    for i, (key, value) in enumerate(countdata.items()):
        if not value:
            continue
        allcounts[key] = np.hstack((allcounts[key], value)) if key in allcounts.keys() else value
    for i, (key, value) in enumerate(distdata.items()):
        if not value:
            continue
        alldist[key] = np.hstack((alldist[key], value)) if key in alldist.keys() else value

# calculate the correlation
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.corrcoef.html normalized covariance matrix (Pearson)
fig, axs = plt.subplots(2, 2, constrained_layout=True)
for i, (key, trial) in enumerate(alltrials.items()):
    # bin the frequencies with widths of 5, change if fft changes
    binned_trial = np.mean(trial.reshape(-1, trial.shape[1], 5), axis=2)
    correlation = np.corrcoef(binned_trial)
    cax = corrplot(axs.flatten()[i], correlation, key)
    cax.set_clim(-1, 1)

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(cax, cax=cbar_ax)

# totalcounts
plt.figure()
incount = np.cumsum(totalcount)
cumulativerecord = np.divide(incount, np.arange(1, len(totalcount) + 1))
plt.plot(cumulativerecord)

# separatecounts
plt.figure()
for key, data in allcounts.items():
    incount = np.cumsum(data)
    cumulativetask = np.divide(incount, np.arange(1, len(data) + 1))
    plt.plot(cumulativetask, label=key)
plt.legend()

# finaldistances
plt.figure()
for key, data in alldist.items():
    plt.plot(data, label=key)
plt.legend()

plt.show()
