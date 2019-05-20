from axopy import util
from axopy.daq import NoiseGenerator
from axopy.task import Task, Oscilloscope
from axopy import pipeline
from axopy.experiment import Experiment
from axopy.gui.canvas import Canvas, Circle
from axopy.features.time import integrated_emg

import numpy as np
from scipy.signal import butter

from PyQt5 import QtGui
import pyqtgraph as pg


class FFT(pipeline.Block):
    """Performs Numpy's Fast Fourier Transform on the windowed signal and returns the positive frequencies and assocaiated rectified powers.
    ----------
    Returns: tuple, where k is number of input signals:
        freq of shape (500,), powers of shape (500,k)
    """

    def process(self, data):
        n = 1000 # number of fft pts
        chan = np.shape(data)[0]
        F = np.fft.fft(data, n = n).T
        F = F.reshape((n,chan))
        freq = np.fft.fftfreq(n, d=1/2000)
        power = np.square(F.real)/n
        posfreq = freq[:int(n/2)]
        pospower = power[:int(n/2),:]
        return (posfreq,pospower) #add frequencies to match power shape

class plotFFT(Task):
    def __init__(self, pipeline, numbands):
        super().__init__()
        self.pipeline = pipeline
        self.numbands = numbands

    def prepare_design(self, design):
        block = design.add_block()
        block.add_trial(attrs={
            'placeholder': 0
            })

    def prepare_graphics(self, container):
        self.plt = pg.PlotWidget()
        self.plots = []
        self.canvas = Canvas()
        self.cursors = []
        for i in np.arange(self.numbands):
            # Plotting
            self.plots.append(self.plt.plot([],[], pen= pg.intColor(i,hues =self.numbands)))

            # Feedback
            self.cursors.append(Circle(0.1, color= pg.intColor(i,hues =self.numbands)))
            self.cursors[i].pos = (i+1.)*2/(self.numbands+1.)-1.,-.9 # evenly space
            self.canvas.add_item(self.cursors[i])


        self.plt.setRange(yRange=(0,1), xRange=(0,200))

        # Qt
        layout=QtGui.QGridLayout()
        container.setLayout(layout)
        layout.addWidget(self.canvas, 0, 0)  # 1st column
        layout.addWidget(self.plt,0, 1) # 2nd column
        

    def prepare_daq(self, daqstream):
        self.daqstream = daqstream
        self.daqstream.start()

    def run_trial(self,trial):
        self.connect(self.daqstream.updated, self.update)

    def update(self, data):
        (self.freq,self.powers), self.integratedEMG = self.pipeline.process(data)
        for i in np.arange(self.numbands):
            self.plots[i].setData(self.freq, self.powers[:,i])
            self.cursors[i].y = self.integratedEMG[i]/20

    def finish(self):
        self.daqstream.stop()

    def key_press(self, key):
        if key == util.key_escape:
            self.finish()
        else:
            super().key_press(key)

dev = NoiseGenerator(rate=2000, num_channels=1, read_size=200)

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
    (lowfilter,highfilter),
    (FFT(),pipeline.Callable(integrated_emg))

])

Experiment(daq=dev, subject='test').run(
    #Oscilloscope(dev),
    plotFFT(main_pipeline, numbands=2)
)