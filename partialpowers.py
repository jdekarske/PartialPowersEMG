from axopy import util
from axopy.daq import NoiseGenerator
from axopy.task import Task, Oscilloscope
from axopy import pipeline
from axopy.experiment import Experiment
from axopy.gui.canvas import Canvas, Circle, Text
from axopy.features.time import integrated_emg
from axopy.timing import Timer, Counter

import numpy as np
from scipy.signal import butter

from PyQt5 import QtGui
import pyqtgraph as pg

from pytrigno.pytrigno import TrignoEMG

import sys
import time


class FFT(pipeline.Block):
    """Performs Numpy's Fast Fourier Transform on the windowed signal and
    returns the positive frequencies and associated rectified powers.
    ----------
    Returns: tuple, where k is number of input signals, n is number of fft pts:
        (freq, shape (n/2,), powers, shape (n/2,k))
    TODO: not sure how many fft pts is appropriate
    """

    def process(self, data):
        n = 1000  # number of fft pts
        chan = np.shape(data)[0]
        F = np.fft.fft(data, n=n)
        F = F.reshape((chan, n))
        freq = np.fft.fftfreq(n, d=1 / 2000)
        power = np.square(F.real) / n
        posfreq = freq[:int(n / 2)]
        pospower = power[:, :int(n / 2)]
        return pospower  # add frequencies to match power shape


class exponentialsmoothing(pipeline.Block):
    """applies a low pass smoothing filter https://en.wikipedia.org/wiki/Exponential_smoothing
    Parameters
    ----------
    alpha : float
        smoothing factor, default - .98
    """
    def __init__(self, alpha=.06):
        super().__init__()
        self.xt = 0
        self.alpha = alpha

    def clear(self):
        self.xt = 0

    def process(self, data):
        self.xt = np.add(np.multiply(data, self.alpha), np.multiply(self.xt, (1 - self.alpha)))
        return self.xt


class Exertion(Task):

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def prepare_design(self, design):
        block = design.add_block()
        for trial in [True] * 2:
            block.add_trial()

    def prepare_storage(self, storage):
        self.writer = storage.create_task('Exertiontask_' + time.strftime("%Y%m%d-%H%M%S"))

    def prepare_graphics(self, container):
        self.countdown = Canvas()
        self.countdowntext = Text("Flex!")
        self.countdown.add_item(self.countdowntext)
        container.set_widget(self.countdown)

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

        self.timer = Counter(10)
        self.timer.timeout.connect(self.finish_trial)

    def run_trial(self, trial):
        self.timer.reset()
        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update)

    def update(self, data):
        self.integratedEMG = self.pipeline.process(data)[1:]
        self.timer.increment()

    def finish_trial(self):
        self.trial.attrs['Exertion'] = self.integratedEMG
        self.writer.write(self.trial)
        self.disconnect(self.daqstream.updated, self.update)
        self.next_trial()

    def finish(self):
        self.daqstream.stop()
        self.finished.emit()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        if key == util.key_q:
            sys.exit()
        else:
            super().key_press(key)


class PartialPowers(Task):

    def __init__(self, pipeline):
        super().__init__()
        self.pipeline = pipeline

    def prepare_design(self, design):
        # yikes... evenly space along the horizontal
        self.cursorx = [(i + 1.) * 2 / (config['numbands'] + 1.) -
                        1. for i in np.arange(config['numbands'])]
        # returns array of all possible combinations
        self.active_targets = np.unpackbits(np.arange(
            np.power(2, config['numbands']), dtype=np.uint8)[:, None],
            axis=1)[:, -config['numbands']:] / 2 + .25

        for training in [True] * 10:
            block = design.add_block()
            for active in self.active_targets:
                block.add_trial(attrs={
                    'active_targets': active,
                    'training': training
                })
            block.shuffle()

    def prepare_storage(self, storage):
        self.writer = storage.create_task('training task_' + time.strftime("%Y%m%d-%H%M%S"))

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

        self.plt.setRange(yRange=(0, .2), xRange=(0, 200))

        # Qt
        layout = QtGui.QGridLayout()
        container.setLayout(layout)
        layout.addWidget(self.canvas, 0, 0)  # 1st column
        layout.addWidget(self.plt, 0, 1)  # 2nd column

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

        self.timer = Counter(100)
        self.timer.timeout.connect(self.finish_trial)

    def run_trial(self, trial):
        # change the color of the target cursor depending if training
        if not trial.attrs['training']:
            for i in np.arange(config['numbands']):
                self.targets[i].color = '#3224b1'
        self._reset()
        # set the target positions
        for i in np.arange(config['numbands']):
            self.targets[i].pos = self.cursorx[i], trial.attrs['active_targets'][i]
            self.targets[i].show()
        self.pipeline.clear()
        self.connect(self.daqstream.updated, self.update)

    def update(self, data):
        self.weights = np.multiply([0.33271298928451853, 0.62291817090414547], .2)
        self.differencefactor = 0.1
        self.integratedEMG = self.pipeline.process(data)
        self.windoweddata = self.integratedEMG.pop(0)
        for i in np.arange(config['numbands']):
            # self.plots[i].setData(self.freq, self.powers[:, i])
            self.cursors[i].y = self.integratedEMG[i] / self.weights[i] - self.differencefactor * np.sum(np.delete(self.integratedEMG, i) / np.delete(self.weights, i))

        target_pos = np.array(self.trial.attrs['active_targets']).flatten()
        if all(cursor.collides_with(self.targets[i]) for i, cursor in enumerate(self.cursors)):
                self.finish_trial()

        self.timer.increment()

    def finish_trial(self):
        self.trial.attrs['final_cursor_pos'] = [self.cursors[i].pos[1] for i in np.arange(config['numbands'])]
        self.trial.attrs['time'] = self.timer.count
        self.trial.attrs['timeout'] = self.trial.attrs['time'] < 1
        self.trial.add_array('windowedEMG', data=self.windoweddata)  # TODO: complain to kenny about this
        self.writer.write(self.trial)
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
        self.finished.emit()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        if key == util.key_q:
            sys.exit()
        else:
            super().key_press(key)


# dev = NoiseGenerator(rate=2000, num_channels=1, read_size=200)
dev = TrignoEMG(channel_range=(0, 0), samples_per_read=200, units='mV')

exp = Experiment(daq=dev, subject='test')
config = exp.configure(numbands=int)


# TODO: figure out how to do this recursively
b, a = butter(1, .1, fs=2000, btype='lowpass')
lowpassfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

b, a = butter(4, (40, 60), fs=2000, btype='bandpass')
lowfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

b, a = butter(4, (80, 100), fs=2000, btype='bandpass')
highfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

b, a = butter(4, (120, 140), fs=2000, btype='bandpass')
midfilter = pipeline.Pipeline([
    pipeline.Filter(b, a=a, overlap=200)
])

main_pipeline = pipeline.Pipeline([
    pipeline.Windower(1000),
    pipeline.Passthrough(
    [(lowfilter, highfilter, midfilter), FFT(), pipeline.Callable(integrated_emg), exponentialsmoothing()])])

exp.screen.showFullScreen()
while True:  
    exp.run(
        # Oscilloscope(pipeline.Windower(2000)),
        # Exertion(main_pipeline)
        PartialPowers(main_pipeline)
    )
    break
