
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

class PlotWindow(QWidget):

    def __init__(self, loop):

        super().__init__()

        layout = QGridLayout()

        self.pw1 = pg.PlotWidget(title="x")
        self.pw2 = pg.PlotWidget(title="y")
        self.pw3 = pg.PlotWidget(title="X")
        self.pw4 = pg.PlotWidget(title="Y")
        self.pw5 = pg.PlotWidget(title="arg(X)")
        self.pw6 = pg.PlotWidget(title="arg(Y)")
        self.pw7 = pg.PlotWidget(title="phase_x")

        layout.addWidget(self.pw1, 0,0)
        layout.addWidget(self.pw2, 0,1)
        layout.addWidget(self.pw3, 0,2)
        layout.addWidget(self.pw4, 1,0)
        layout.addWidget(self.pw5, 1,1)
        layout.addWidget(self.pw6, 1,2)
        layout.addWidget(self.pw7, 2,0)

        self.setLayout(layout)

        self.curve1 = self.pw1.plot(pen="w")
        self.curve2 = self.pw2.plot(pen="w")
        self.curve3 = self.pw3.plot(pen="w")
        self.curve4 = self.pw4.plot(pen="w")
        self.curve5 = self.pw5.plot(pen="w")
        self.curve6 = self.pw6.plot(pen="w")

        self.curve12 = self.pw1.plot(pen="r")
        self.curve22 = self.pw2.plot(pen="r")

        self.curve7 = self.pw7.plot(symbol = "o", symbolSize=5, symbolBrush="r")
        self.curve8 = self.pw7.plot(symbol ="o", symbolSize=5, symbolBrush="b")

        self.windowSize = 200

        # create empty data buffers
        self.bufferSize = 500 
        self.data1= np.zeros(self.bufferSize)
        self.data2= np.zeros(self.bufferSize)
        self.data3= np.zeros(self.bufferSize)
        self.data4= np.zeros(self.bufferSize)
        self.data5= np.zeros(self.bufferSize)
        self.data6= np.zeros(self.bufferSize)
        self.phase_x= np.zeros(self.windowSize)
        self.phase_y= np.zeros(self.windowSize)
        self.received_data = np.empty((6,0)) 

        self.loop = loop
        # for recorded data
        self.recorded_data = np.empty((6,0))
        self.count=0

        # worker
        self.ble_worker = None
        self.timer = None
    
    def setup_worker(self, address):
        self.ble_worker = BLEWorker(self.loop)
        self.ble_worker.set_address(address)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_ble()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)


    def read_data(self, new_data):
        self.received_data = np.array(new_data).reshape(6,-1)
        self.data1 = self.data1.append(self.received_data[0])
        self.data2 = self.data2.append(self.received_data[1])
        self.data3 = self.data3.append(self.received_data[2])
        self.data4 = self.data4.append(self.received_data[3])
        self.data5 = self.data5.append(self.received_data[4])
        self.data6 = self.data6.append(self.received_data[5])


    async def cleanup(self):
        print("Disconnecting...")

        if self.ble_worker.client and self.ble_worker.client.is_connected:
            await self.ble_worker.client.disconnect()
            print("Device disconnected.")
        

    def update(self):
            
        if self.recorded_data.size>0: # if recording
            chunk_size=10
            self.received_data = self.recorded_data[:,self.count*chunk_size:(self.count+1)*chunk_size]

        elif self.received_data.shape[1]: 
            chunk_size = self.received_data.shape[1]

        else: # no data received yet
            return

        order = 6
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.5  # 100 Hz -> 10 Hz cutoff
        b, a = butter(order, Wn, 'low') 

        fs = 100 # sampling frequency
        t1 = np.arange(self.windowSize,dtype=float)/fs

        # shift one sample 
        self.data1 = np.roll(self.data1, -1) 
        self.data2 = np.roll(self.data2, -1) 

        acc_x = self.data1[:self.windowSize]
        acc_y = self.data2[:self.windowSize]

        # calculate fft, arguments
        xl = filtfilt(b, a, acc_x)
        yl = filtfilt(b, a, acc_y)

        f1 = fftshift(fft(xl)/len(xl)) # fft_x
        f2 = fftshift(fft(yl)/len(yl)) # fft_y

        f1[np.abs(f1)<1e-6]=0  # remove tiny noise components
        f2[np.abs(f2)<1e-6]=0
    
        freqs = fftshift(fftfreq(len(xl), d = 1/fs))

        # DC masking
        th = 1.0
        mask = freqs > th
        peak_x_idx = np.argmax(np.abs(f1[mask]))
        peak_y_idx = np.argmax(np.abs(f2[mask]))
        peak_x = freqs[mask][peak_x_idx]
        peak_y = freqs[mask][peak_y_idx]

        argx = np.angle(f1)
        argy = np.angle(f2)

        peak_phase_x = np.angle(f1[mask][peak_x_idx])
        peak_phase_y = np.angle(f2[mask][peak_y_idx])

        # for phase plotting
        self.phase_x = np.roll(self.phase_x, -1) 
        self.phase_y = np.roll(self.phase_y, -1)

        phase_values = np.full((2, chunk_size), np.nan)
        phase_values[0,-1] = peak_phase_x
        phase_values[1,-1] = peak_phase_y

        self.data3[-chunk_size:] = phase_values[0,:]
        self.data4[-chunk_size:] = phase_values[1,:]

        # update
        self.curve1.setData(t1,self.data1) # ax
        self.curve2.setData(t1,self.data2) # ay
        self.curve3.setData(freqs,np.abs(f1)) 
        self.curve4.setData(freqs,np.abs(f2)) 
        self.curve5.setData(freqs,argx) 
        self.curve6.setData(freqs,argy) 

        self.curve7.setData(self.data3)
        self.curve8.setData(self.data4)

        # filtered curves
        self.curve12.setData(t1,xl)
        self.curve22.setData(t1,yl) 

        self.received_data = np.empty((6, 0))
        self.count +=1


