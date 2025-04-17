
# PlotWindow class
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import butter,lfilter,filtfilt
import asyncio
from collections import deque

from bleworker import BLEWorker


class PlotWindow(QWidget):

    def __init__(self, loop):

        super().__init__()

        self.loop = loop
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

        # data buffers
        self.accx_buf = deque()
        self.accy_buf = deque()
        self.accz_buf = deque()
        self.gyrox_buf = deque()
        self.gyroy_buf = deque()
        self.gyroz_buf = deque()

        self.received_data = np.empty((6,0)) 

        self.windowSize = 100
        self.plot_bufx = np.zeros(2*self.windowSize)
        self.plot_bufy = np.zeros(2*self.windowSize)
        self.phase_bufx= np.zeros(2*self.windowSize)
        self.phase_bufy= np.zeros(2*self.windowSize)

        # for recorded data
        self.recorded_data = np.empty((6,0))
        self.recording_timer = QtCore.QTimer()
        self.recording_timer.timeout.connect(self.read_recording)
        #self.recording_timer.start(50)
        self.readCount=0

        self.timer = QtCore.QTimer() # Timer to shift samples
        self.timer.timeout.connect(self.shift_window)
        self.timer.start(10) # 100Hz
        self.count=0

        # worker
        self.ble_worker = None
        self.update_timer = None

        # filtering, masking, axis values
        order = 3
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.5  # 100 Hz -> 10 Hz cutoff
        self.b, self.a = butter(order, Wn, 'low') 

        self.fs = 100 # sampling frequency
        self.t = np.arange(self.windowSize,dtype=float)/self.fs

        self.freqs = fftshift(fftfreq(self.windowSize, d = 1/self.fs))
        th = 1.0
        self.mask = self.freqs > th
    
    def setup_worker(self, address):
        self.ble_worker = BLEWorker(self.loop)
        self.ble_worker.set_address(address)
        self.ble_worker.data_received.connect(self.read_data)
#        self.ble_worker.start_ble()

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)


    def read_data(self, new_data):
        self.received_data = np.array(new_data).reshape(6,-1)

        self.accx_buf.extend(self.received_data[0])
        self.accy_buf.extend(self.received_data[1])
        self.accz_buf.extend(self.received_data[2])
        self.gyrox_buf.extend(self.received_data[3])
        self.gyroy_buf.extend(self.received_data[4])
        self.gyroz_buf.extend(self.received_data[5])


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

    def shift_window(self):
#        print(f"len deque: {len(self.accx_buf)}")

        if not self.accx_buf:
            return

        N = self.windowSize

        # shift one sample 
        j = self.count % N
        new_x = self.accx_buf.popleft()
        new_y = self.accy_buf.popleft()

        self.plot_bufx[j]   = new_x
        self.plot_bufx[j+N] = new_x
        self.plot_bufy[j]   = new_y
        self.plot_bufy[j+N] = new_y

        acc_x = self.plot_bufx[j+1:j+N+1] # current window
        acc_y = self.plot_bufy[j+1:j+N+1]

        # calculate fft, arguments
        filtered_x = filtfilt(self.b, self.a, acc_x)
        filtered_y = filtfilt(self.b, self.a, acc_y)

        fx = np.fft.fft(filtered_x)
        fy = np.fft.fft(filtered_y)

        # DC masking
        masked_indices = np.where(self.mask)[0]
        peak_idx_x = np.argmax(np.abs(fx[self.mask]))
        peak_idx_y = np.argmax(np.abs(fy[self.mask]))

        peak_idx_x = masked_indices[peak_idx_x]
        peak_idx_y = masked_indices[peak_idx_y]

        peak_phase_x = np.angle(fx[peak_idx_x]) 
        peak_phase_y = np.angle(fy[peak_idx_y])

        self.phase_bufx[j]   = peak_phase_x
        self.phase_bufx[j+N] = peak_phase_x
        self.phase_bufy[j]   = peak_phase_y
        self.phase_bufy[j+N] = peak_phase_y

        self.count+=1


    async def cleanup(self):
        print("Disconnecting...")

        if self.ble_worker.client and self.ble_worker.client.is_connected:
            await self.ble_worker.client.disconnect()
            print("Device disconnected.")
        

    def update(self):

        # shift one sample 
        N = self.windowSize
        j = self.count % N
        acc_x = self.plot_bufx[j+1:j+N+1] # current window
        acc_y = self.plot_bufy[j+1:j+N+1]
        phase_x = self.phase_bufx[j+1:j+N+1]
        phase_y = self.phase_bufy[j+1:j+N+1]
        
        fx = fftshift(np.fft.fft(acc_x))
        fy = fftshift(np.fft.fft(acc_y))

        """
        # update
        self.curve1.setData(self.t,acc_x) # ax
        self.curve2.setData(self.t,acc_y) # ay
        self.curve3.setData(self.freqs,np.abs(fx)) 
        self.curve4.setData(self.freqs,np.abs(fy)) 
        #self.curve5.setData(freqs,argx)
        #self.curve6.setData(freqs,argy)

        self.curve7.setData(phase_x)
        self.curve8.setData(phase_y)

        # filtered curves
        #self.curve12.setData(t1,xl)
        #self.curve22.setData(t1,yl) 
        """
