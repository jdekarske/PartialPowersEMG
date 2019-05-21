from axopy import util
from axopy.daq import NoiseGenerator
from axopy.task import Task, Oscilloscope
from axopy import pipeline
from axopy.experiment import Experiment
from axopy.gui.canvas import Canvas, Circle
from axopy.features.time import integrated_emg
from axopy.timing import Counter

import numpy as np
from scipy.signal import butter

from PyQt5 import QtGui
import pyqtgraph as pg


class RLSMapping(pipeline.Block):
    """Linear mapping of EMG amplitude to position updated by RLS.
    Parameters
    ----------
    m : int
        Number of vectors in the mapping.
    k : int
        Dimensionality of the mapping vectors.
    lam : float
        Forgetting factor.
    """

    def __init__(self, m, k, lam, delta=0.001):
        super(RLSMapping, self).__init__()
        self.m = m
        self.k = k
        self.lam = lam
        self.delta = delta
        self._init()

    @classmethod
    def from_weights(cls, weights):
        """Construct an RLSMapping static weights."""
        obj = cls(1, 1, 1)
        obj.weights = weights
        return obj

    def _init(self):
        self.w = np.zeros((self.k, self.m))
        self.P = np.eye(self.m) / self.delta

    def process(self, data):
        """Just applies the current weights to the input."""
        self.y = data[:, None]  # didn't work with 2d shape
        self.xhat = self.y.dot(self.w.T)
        return self.xhat

    def update(self, x):
        """Update the weights with the teaching signal."""
        z = self.P.dot(self.y.T)
        g = z / (self.lam + self.y.dot(z))
        e = x - self.xhat
        self.w = self.w + np.outer(e, g)
        self.P = (self.P - np.outer(g, z)) / self.lam


class FFT(pipeline.Block):
    """Performs Numpy's Fast Fourier Transform on the windowed signal and returns the positive frequencies and associated rectified powers.
    ----------
    Returns: tuple, where k is number of input signals, n is number of fft pts:
        (freq, shape (n/2,), powers, shape (n/2,k))
    TODO: not sure how many fft pts is appropriate
    """

    def process(self, data):
        n = 1000  # number of fft pts
        chan = np.shape(data)[0]
        F = np.fft.fft(data, n=n).T
        F = F.reshape((n, chan))
        freq = np.fft.fftfreq(n, d=1/2000)
        power = np.square(F.real)/n
        posfreq = freq[:int(n/2)]
        pospower = power[:int(n/2), :]
        return (posfreq, pospower)  # add frequencies to match power shape


class plotFFT(Task):
    '''This will train the vertical mappings for the signals simultaneously'''

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def prepare_design(self, design):
        # yikes... evenly space along the horizontal
        self.cursorx = [(i+1.)*2/(config['numbands'] + 1.) -
                        1. for i in np.arange(config['numbands'])]
        self.active_targets = np.unpackbits(np.arange(np.power(2, config['numbands']), dtype=np.uint8)[
                                            :, None], axis=1)[:, -config['numbands']:]+.25  # returns array of all possible combinations

        for training in [True, True, True, False]:
            block = design.add_block()
            for active in self.active_targets:
                block.add_trial(attrs={
                    'active_targets': active,
                    'training': training
                })
            block.shuffle()

    def prepare_graphics(self, container):
        self.plt = pg.PlotWidget()
        self.plots = []
        self.canvas = Canvas()
        self.cursors = []
        self.targets = []
        for i in np.arange(config['numbands']):
            # Plotting
            self.plots.append(self.plt.plot(
                [], [], pen=pg.intColor(i, hues=config['numbands'])))

            # Feedback
            self.cursors.append(
                Circle(0.1, color=pg.intColor(i, hues=config['numbands'])))
            self.cursors[i].pos = self.cursorx[i], 0  # evenly space
            self.canvas.add_item(self.cursors[i])

            # Training
            self.targets.append(Circle(0.1, color='#32b124'))
            self.targets[i].pos = self.cursorx[i], .5  # evenly space
            self.canvas.add_item(self.targets[i])

        self.plt.setRange(yRange=(0, 1), xRange=(0, 200))

        # Qt
        layout = QtGui.QGridLayout()
        container.setLayout(layout)
        layout.addWidget(self.canvas, 0, 0)  # 1st column
        layout.addWidget(self.plt, 0, 1)  # 2nd column

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

        self.timer = Counter(50)
        self.timer.timeout.connect(self.finish_trial)

    def run_trial(self, trial):
        if not trial.attrs['training']:
            for i in np.arange(config['numbands']):
                self.targets[i].color = '#3224b1'
        self._reset()
        for i in np.arange(config['numbands']):
            self.targets[i].pos = self.cursorx[i], trial.attrs['active_targets'][i]
            self.targets[i].show()
        # self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update)

    def update(self, data):
        (self.freq, self.powers), self.integratedEMG = self.pipeline.process(data)
        for i in np.arange(config['numbands']):
            self.plots[i].setData(self.freq, self.powers[:, i])
            self.cursors[i].y = self.integratedEMG[i]/20

        target_pos = np.array(self.trial.attrs['active_targets']).flatten()
        if self.trial.attrs['training'] and 'RLSMapping' in self.pipeline.named_blocks:
            self.pipeline.named_blocks['RLSMapping'].update(target_pos)
        for i, cursor in enumerate(self.cursors):
            if cursor.collides_with(self.targets[i]):
                self.finish_trial()

        self.timer.increment()

    def finish_trial(self):
        self.disconnect(self.daqstream.updated, self.update)
        self._reset()
        self.next_trial()

    def _reset(self):
        for i in np.arange(config['numbands']):
            self.cursors[i].pos = self.cursorx[i], 0  # evenly space
            self.targets[i].hide()
        self.timer.reset()

    def finish(self):
        self.daqstream.stop()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        else:
            super().key_press(key)


dev = NoiseGenerator(rate=2000, num_channels=1, read_size=200)

exp = Experiment(daq=dev, subject='test')
config = exp.configure(numbands=int)

b, a = butter(4, (80, 100), fs=2000, btype='bandpass')
lowfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

b, a = butter(4, (130, 160), fs=2000, btype='bandpass')
highfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

main_pipeline = pipeline.Pipeline([
    pipeline.Windower(400),
    (lowfilter, highfilter),
    # , RLSMapping(config['numbands'], config['numbands'], 0.98)])
    (FFT(), [pipeline.Callable(integrated_emg)])

])

exp.run(
    # Oscilloscope(dev),
    plotFFT(main_pipeline)
)
