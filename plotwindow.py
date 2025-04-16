
# PlotWindow class
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import butter,lfilter,filtfilt
#import math
import asyncio

from bleworker import BLEWorker

from collections import deque

class PlotWindow(QWidget):

    def __init__(self, loop):

        super().__init__()

        layout = QGridLayout()

        self.pw1 = pg.PlotWidget(title="x")
        self.pw2 = pg.PlotWidget(title="y")
        self.pw3 = pg.PlotWidget(title="X")
        self.pw4 = pg.PlotWidget(title="Y")
        #self.pw5 = pg.PlotWidget(title="arg(X)")
        #self.pw6 = pg.PlotWidget(title="arg(Y)")
        self.pw7 = pg.PlotWidget(title="phase_x")

        layout.addWidget(self.pw1, 0,0)
        layout.addWidget(self.pw2, 0,1)
        layout.addWidget(self.pw3, 0,2)
        layout.addWidget(self.pw4, 1,0)
        #layout.addWidget(self.pw5, 1,1)
        #layout.addWidget(self.pw6, 1,2)
        layout.addWidget(self.pw7, 1,1)

        self.setLayout(layout)

        self.curve1 = self.pw1.plot(pen="w")
        self.curve2 = self.pw2.plot(pen="w")
        self.curve3 = self.pw3.plot(pen="w")
        self.curve4 = self.pw4.plot(pen="w")
#        self.curve5 = self.pw5.plot(pen="w")
#        self.curve6 = self.pw6.plot(pen="w")

        self.curve12 = self.pw1.plot(pen="r")
        self.curve22 = self.pw2.plot(pen="r")

        self.curve7 = self.pw7.plot(symbol = "o", symbolSize=5, symbolBrush="r")
        self.curve8 = self.pw7.plot(symbol ="o", symbolSize=5, symbolBrush="b")

        self.windowSize = 1000 
        # create empty data buffers
        #self.bufferSize = 500 

        self.accx_buf = deque()
        self.accy_buf = deque()
        self.accz_buf = deque()
        self.gyrox_buf = deque()
        self.gyroy_buf = deque()
        self.gyroz_buf = deque()

        self.phase_x= np.zeros(self.windowSize)
        self.phase_y= np.zeros(self.windowSize)
        self.received_data = np.empty((6,0)) 

        self.plot_bufx = np.zeros(self.windowSize)
        self.plot_bufy = np.zeros(self.windowSize)

        self.loop = loop
        # for recorded data
        self.recorded_data = np.empty((6,0))
        self.readCount=0
        self.count=0
        self.recording_timer = QtCore.QTimer()
        self.recording_timer.timeout.connect(self.read_recording)
        self.recording_timer.start(50)

        # worker
        self.ble_worker = None
        self.timer = None

        # filter coefficients
        order = 3
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.5  # 100 Hz -> 10 Hz cutoff
        self.b, self.a = butter(order, Wn, 'low') 

        self.fs = 1000 # sampling frequency
        self.t = np.arange(self.windowSize,dtype=float)/self.fs

        self.freqs = fftshift(fftfreq(self.windowSize, d = 1/self.fs))
        th = 1.0
        self.mask = self.freqs > th
    
    def setup_worker(self, address):
        self.ble_worker = BLEWorker(self.loop)
        self.ble_worker.set_address(address)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_ble()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(10)


    def read_data(self, new_data):
        self.received_data = np.array(new_data).reshape(6,-1)
        self.accx_buf  = np.append(self.accx_buf, self.received_data[0])
        self.accy_buf  = np.append(self.accy_buf, self.received_data[1])
        self.accz_buf  = np.append(self.accz_buf, self.received_data[2])
        self.gyrox_buf  = np.append(self.gyrox_buf, self.received_data[3])
        self.gyroy_buf  = np.append(self.gyroy_buf, self.received_data[4])
        self.gyroz_buf  = np.append(self.gyroz_buf,self.received_data[5])

    def read_recording(self):
        chunk_size=10
        start = self.readCount * chunk_size
        end = (self.readCount+1) * chunk_size

        self.received_data = self.recorded_data[:,start:end]
        self.accx_buf.extend(self.received_data[0])
        self.accy_buf.extend(self.received_data[1])
        self.accz_buf.extend(self.received_data[2])
        self.gyrox_buf.extend(self.received_data[3])
        self.gyroy_buf.extend(self.received_data[4])
        self.gyroz_buf.extend(self.received_data[5])

        self.readCount +=1


    async def cleanup(self):
        print("Disconnecting...")

        if self.ble_worker.client and self.ble_worker.client.is_connected:
            await self.ble_worker.client.disconnect()
            print("Device disconnected.")
        

    def update(self):

        print(f"len deque: {len(self.accx_buf)}")
        N = self.windowSize

        # shift one sample 
        j = self.count % N
        self.plot_bufx[j]   = self.accx_buf.popleft()
#        self.plot_bufx[j+N] = self.accx_buf[j]
        self.plot_bufy[j]   = self.accy_buf.popleft()
#        self.plot_bufy[j+N] = self.accy_buf[j]

        """
        self.plot_bufx[:-1] = self.plot_bufx[1:]
        self.plot_bufy[:-1] = self.plot_bufy[1:]

        if self.accx_buf:
            self.plot_bufx[-1] = self.accx_buf.popleft()
            self.plot_bufy[-1] = self.accy_buf.popleft()
        else:
            return
        """
#        acc_x = self.plot_bufx[j+1:j+N+1] # current window
#        acc_y = self.plot_bufy[j+1:j+N+1]
        xl = self.plot_bufx
        yl = self.plot_bufy

        # calculate fft, arguments
        #xl = filtfilt(self.b, self.a, acc_x)
        #yl = filtfilt(self.b, self.a, acc_y)

        f1 = fftshift(fft(xl)/len(xl)) # fft_x
        f2 = fftshift(fft(yl)/len(yl)) # fft_y

#        f1[np.abs(f1)<1e-6]=0  # remove tiny noise components
#        f2[np.abs(f2)<1e-6]=0
        """
        # DC masking
        peak_x_idx = np.argmax(np.abs(f1[self.mask]))
        peak_y_idx = np.argmax(np.abs(f2[self.mask]))
        peak_x = self.freqs[self.mask][peak_x_idx]
        peak_y = self.freqs[self.mask][peak_y_idx]

        argx = np.angle(f1)
        argy = np.angle(f2)

        peak_phase_x = np.angle(f1[self.mask][peak_x_idx]) # extracts argument at wanted frequencies
        peak_phase_y = np.angle(f2[self.mask][peak_y_idx])

        # for phase plotting
        self.phase_x[:-1] = self.phase_x[1:]
        self.phase_y[:-1] = self.phase_y[1:]
        self.phase_x[-1] = peak_phase_x
        self.phase_y[-1] = peak_phase_y
        """
        """
        # update
        self.curve1.setData(self.t,xl) # ax
        self.curve2.setData(self.t,yl) # ay
        self.curve3.setData(self.freqs,np.abs(f1)) 
        self.curve4.setData(self.freqs,np.abs(f2)) 
        #self.curve5.setData(freqs,argx)
        #self.curve6.setData(freqs,argy)

        self.curve7.setData(self.phase_x)
        self.curve8.setData(self.phase_y)

        # filtered curves
        #self.curve12.setData(t1,xl)
        #self.curve22.setData(t1,yl) 
        """

