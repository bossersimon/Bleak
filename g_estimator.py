
# g_estimator.py

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import butter,lfilter,filtfilt
import math

from bleak import BleakClient
import asyncio
import qasync
import signal

class PlotWindow(QWidget):

    def __init__(self):

        super().__init__()

        layout = QGridLayout()

        self.pw1 = pg.PlotWidget(title="x")
        self.pw2 = pg.PlotWidget(title="y")
        self.pw3 = pg.PlotWidget(title="X")
        self.pw4 = pg.PlotWidget(title="Y")
        self.pw5 = pg.PlotWidget(title="instantaneous phase. r = x, b = y, g = unambiguous x")
        self.pw6 = pg.PlotWidget(title="arg(Y)")

        self.pw1.resize(1000,800)
        self.pw2.resize(1000,800)
        self.pw3.resize(1000,800)
        self.pw4.resize(1000,800)
        self.pw5.resize(1000,800)
        self.pw6.resize(1000,800)
       
        layout.addWidget(self.pw1, 0,0)
        layout.addWidget(self.pw2, 0,1)
        layout.addWidget(self.pw3, 0,2)
        layout.addWidget(self.pw4, 1,0)
        layout.addWidget(self.pw5, 1,1)
        layout.addWidget(self.pw6, 1,2)

        self.setLayout(layout)

        self.curve1 = self.pw1.plot(pen="w")
        self.curve2 = self.pw2.plot(pen="w")
        self.curve3 = self.pw3.plot(pen="w")
        self.curve4 = self.pw4.plot(pen="w")
        self.curve5 = self.pw5.plot(pen="w")
        self.curve6 = self.pw6.plot(pen="w")

        self.curve7 = self.pw5.plot(symbol = "o", symbolSize=5, symbolBrush="r")
        self.curve8 = self.pw5.plot(symbol ="o", symbolSize=5, symbolBrush="b")
        self.curve9 = self.pw5.plot(pen="g")

        self.curve12 = self.pw1.plot(pen="r")
        self.curve22 = self.pw2.plot(pen="r")

        # create empty data buffers
        self.bufferSize = 500
        self.data1= np.zeros(self.bufferSize)
        self.data2= np.zeros(self.bufferSize)
        self.data3= np.zeros(self.bufferSize)
        self.data4= np.zeros(self.bufferSize)
        self.data5= np.zeros(self.bufferSize)
        self.latest_data = np.empty((6,0)) 
        self.data6= np.zeros(self.bufferSize)

        self.chunk_size = 100
        self.win_phasex= np.zeros(2*self.chunk_size)
        self.win_phasey= np.zeros(2*self.chunk_size)
        self.win_phase = np.zeros(2*self.chunk_size)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        self.count = 0

        self.counter = 0
        
    def update(self):
        i = self.counter

        order = 6
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.5
        b, a = butter(order, Wn, 'low')

        fs = 100
        t1 = np.arange(self.chunk_size,dtype=float)/fs

        freqs = fftshift(fftfreq(self.chunk_size, d = 1/fs))

        th = 1.0 # mask anything above 1 Hz 
        mask = freqs > th
        n_shift = 1

        if i<=990:
            self.data1 = self.latest_data[0] # acc_x
            self.data2 = self.latest_data[1] # acc_y

            d1 = filtfilt(b,a,self.data1[i*n_shift:i*n_shift+100])
            d2 = filtfilt(b,a,self.data2[i*n_shift:i*n_shift+100])

            fx = np.fft.fft(d1)
            fy = np.fft.fft(d2)

            peak_idx_x = np.argmax(np.abs(fx[mask])) # index of wanted frequency relative
            peak_idx_y = np.argmax(np.abs(fy[mask])) # to masked array

            masked_indices = np.where(mask)[0] # [0] gives indices of True condition
            peak_idx_x = masked_indices[peak_idx_x] # index in full array corresponding 
            peak_idx_y = masked_indices[peak_idx_y] # to wanted frequency
            
            X = fx[peak_idx_x]
            Y = fy[peak_idx_y]
            phase  = np.angle(X + 1j*Y)
            
            phase_x = np.angle(fx[peak_idx_x]) 
            phase_y = np.angle(fy[peak_idx_y])

            N = self.chunk_size
            j = self.count % N
            self.win_phasex[j]   = phase_x
            self.win_phasex[j+N] = phase_x
            self.win_phasey[j]   = phase_y
            self.win_phasey[j+N] = phase_y
            self.win_phase[j]    = phase
            self.win_phase[j+N]  = phase

            phase_x = self.win_phasex[j+1:j+N+1]
            phase_y = self.win_phasey[j+1:j+N+1]
            phi     = self.win_phase[j+1:j+N+1]

                # update
            self.curve1.setData(t1,d1)
            self.curve2.setData(t1,d2) 
            self.curve3.setData(freqs,np.abs(fx)) 
            self.curve4.setData(freqs,np.abs(fy)) 
            self.curve7.setData(t1,phase_x) 
            self.curve8.setData(t1,phase_y) 
            self.curve9.setData(t1,phi) 

            self.count +=1

        self.counter +=1

def generate_signals(plot):
    fs = 100 # sampling frequency
    T = 100 # signal length
    N = T*fs # number of samples
    f = 10
    phi = np.pi/6

    order = 6
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
    Wn = 0.5
    b, a = butter(order, Wn, 'low')

    chunk_size = 100

#    window = np.hanning(1000)
    window = 1
    t1 = np.arange(N,dtype=float)/fs
    c1 = np.random.normal(size=N)
    c1 += 10*np.sin(2*np.pi *f*t1+phi)*window

    c2 = np.random.normal(size=N)
    c2 += 10*np.cos(2*np.pi*f*t1+phi)*window

    d1 = output_signal = filtfilt(b, a, c1)
    d2 = output_signal = filtfilt(b, a, c2)

    new_arr = np.hstack((c1.reshape(-1,1),c2.reshape(-1,1)))
    np.savetxt("recording.txt", new_arr, delimiter=",", fmt="%.18e")

    fx = fftshift(fft(d1)/len(d1))
    fy = fftshift(fft(d2)/len(d2))

    fx[np.abs(fx)<1e-6]=0
    fy[np.abs(fy)<1e-6]=0

    freqs = fftshift(fftfreq(len(d1), d = 1/fs))

    argx = np.angle(fx)
    argy = np.angle(fy)

    rows = np.array([c1,c2,np.abs(fx),np.abs(fy), argx, argy])
    plot.latest_data = rows

def read_recording(plot):
    loaded_data = np.loadtxt("recording1.txt", delimiter = ",")
    loaded_data = loaded_data.T
    plot.latest_data= np.empty((6,loaded_data.shape[1]))
    plot.latest_data = loaded_data

if __name__ == "__main__":
    app = pg.mkQApp()
    plot = PlotWindow()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    generate_signals(plot)
    #read_recording(plot)

    plot.show()
    app.exec()
