
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
address = "DC:1E:D5:1B:E9:FE" # ESP MAC address


class PlotWindow(QWidget):

    def __init__(self, loop, playback = False, file_writer=None):

        super().__init__()

        layout = QGridLayout()

        self.pw1 = pg.PlotWidget(title="x")
        self.pw2 = pg.PlotWidget(title="y")
        self.pw3 = pg.PlotWidget(title="gz")
        self.pw4 = pg.PlotWidget(title="fft(x)")
        self.pw5 = pg.PlotWidget(title="fft(y)")
        self.pw6 = pg.PlotWidget(title="arg x,y")
#        self.pw7 = pg.PlotWidget(title="arg(y)")

        layout.addWidget(self.pw1, 0,0)
        layout.addWidget(self.pw2, 0,1)
        layout.addWidget(self.pw3, 0,2)
        layout.addWidget(self.pw4, 1,0)
        layout.addWidget(self.pw5, 1,1)
        layout.addWidget(self.pw6, 1,2)
        #layout.addWidget(self.pw7, 1,1)

        self.setLayout(layout)

        self.curve1 = self.pw1.plot(pen="w")
        self.curve2 = self.pw2.plot(pen="w")
        self.curve3 = self.pw3.plot(pen="w")
        self.curve4 = self.pw4.plot(pen="w")
        self.curve5 = self.pw5.plot(pen="w")
        self.curve6 = self.pw6.plot(pen="w")

#        self.curve12 = self.pw1.plot(pen="r")
#        self.curve22 = self.pw2.plot(pen="r")

        self.curve7 = self.pw6.plot(symbol = "o", symbolSize=5, symbolBrush="r")
        self.curve8 = self.pw6.plot(symbol ="o", symbolSize=5, symbolBrush="b")

        # data buffers
        self.accx_buf = deque()
        self.accy_buf = deque()
        self.accz_buf = deque()
        self.gyrox_buf = deque()
        self.gyroy_buf = deque()
        self.gyroz_buf = deque()

        self.received_data = np.empty((6,0)) 

        self.windowSize = 2000
        self.win_accx = np.zeros(2*self.windowSize)
        self.win_accy = np.zeros(2*self.windowSize)
        self.win_phasex= np.zeros(2*self.windowSize)
        self.win_phasey= np.zeros(2*self.windowSize)
        self.win_gyroz = np.zeros(2*self.windowSize)
        self.win_phase = np.zeros(2*self.windowSize)

        self.loop = loop
        self.ble_worker = BLEWorker(loop, address)

        # for recorded data
        if playback:
            self.recorded_data = np.empty((6,0))
            self.recording_timer = QtCore.QTimer()
            self.recording_timer.timeout.connect(self.read_recording)
            self.recording_timer.start(50)
            self.readCount=0

        else: 
            self.ble_worker.data_received.connect(self.read_data)
            self.ble_worker.start_ble()

        self.timer = QtCore.QTimer() # Timer to shift samples
        self.timer.timeout.connect(self.shift_window)
        self.timer.start(2) # 100Hz
        self.count=0

        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.update)
        self.update_timer.start(50)

        # filtering, masking, axis values
        order = 3
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.13  # 100 Hz -> 10 Hz cutoff
        self.b, self.a = butter(order, Wn, 'low') 

        self.fs = 100 # sampling frequency
        self.t = np.arange(self.windowSize,dtype=float)/self.fs

        self.freqs = fftshift(fftfreq(self.windowSize, d = 1/self.fs))
        th = 1.0 # mask anything above 1 Hz 
        self.mask = self.freqs > th

        self.writer = file_writer

    
    def read_data(self, new_data):
        self.received_data = np.array(new_data).reshape(6,-1)
        if self.writer:
            self.writer.writerows(self.received_data.T)

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

        if not self.accx_buf:
            return

        N = self.windowSize

        # shift one sample 
        j = self.count % N
        new_x = self.accx_buf.popleft()
        new_y = self.accy_buf.popleft()
        new_gz = self.gyroz_buf.popleft()

        self.win_accx[j]   = new_x
        self.win_accx[j+N] = new_x
        self.win_accy[j]   = new_y
        self.win_accy[j+N] = new_y

        self.win_gyroz[j]   = new_gz
        self.win_gyroz[j+N] = new_gz

        acc_x = self.win_accx[j+1:j+N+1] # current window
        acc_y = self.win_accy[j+1:j+N+1]

        # calculate fft, arguments
        filtered_x = filtfilt(self.b, self.a, acc_x)
        filtered_y = filtfilt(self.b, self.a, acc_y)

        fx = np.fft.fft(filtered_x)
        fy = np.fft.fft(filtered_y)

        # DC masking
        peak_idx_x = np.argmax(np.abs(fx[self.mask])) # index of wanted frequency relative
        peak_idx_y = np.argmax(np.abs(fy[self.mask])) # to masked array

        masked_indices = np.where(self.mask)[0] # [0] gives indices of True condition
        peak_idx_x = masked_indices[peak_idx_x] # index in full array corresponding 
        peak_idx_y = masked_indices[peak_idx_y] # to wanted frequency


        X = fx[peak_idx_x]
        Y = fy[peak_idx_y]
        phase  = np.angle(X + 1j*Y)
        
        phase_x = np.angle(fx[peak_idx_x]) 
        phase_y = np.angle(fy[peak_idx_y])

      #  ReX = np.abs(fx[peak_idx_x])*np.cos(phase_x)
      #  ReY = np.abs(fy[peak_idx_y])*np.sin(phase_y)
      #  phase = np.arctan2(ReY, ReX)


        self.win_phasex[j]   = phase_x
        self.win_phasex[j+N] = phase_x
        self.win_phasey[j]   = phase_y
        self.win_phasey[j+N] = phase_y
        self.win_phase[j]    = phase
        self.win_phase[j+N]  = phase

        self.count+=1


    async def cleanup(self):
        print("Disconnecting...")
        
        print(f" if client: {self.ble_worker.client}")
        print(f" is connected: {self.ble_worker.client.is_connected}")

        if self.ble_worker.client and self.ble_worker.client.is_connected:
            await self.ble_worker.client.disconnect()
            print("Device disconnected.")
        

    def update(self):
        #print(f"len deque: {len(self.accx_buf)}")

        # shift one sample 
        N = self.windowSize
        j = (self.count-1) % N
        acc_x =   self.win_accx[j+1:j+N+1] # current window
        acc_y =   self.win_accy[j+1:j+N+1]
        phase_x = self.win_phasex[j+1:j+N+1]
        phase_y = self.win_phasey[j+1:j+N+1]
        gyro_z =  self.win_gyroz[j+1:j+N+1]
        phi =   self.win_phase[j+1:j+N+1]
        
        fx = fftshift(np.fft.fft(acc_x))
        fy = fftshift(np.fft.fft(acc_y))
    
        masked_indices = np.where(self.mask)[0]
        peak_idx_x = np.argmax(np.abs(fx[self.mask]))
        peak_idx_x = masked_indices[peak_idx_x]
        peak_freq = self.freqs[peak_idx_x]

       # print(f"frequency: {peak_freq}, DPS: {peak_freq*360}")

#        heading_rate = gx*np.cos(phi)-gy*np.sin(phi)
#        roll_rate    = gx*np.sin(phi)+gy*np.cos(phi)

        # update
        self.curve1.setData(self.t,acc_x) # ax
        self.curve2.setData(self.t,acc_y) # ay
        self.curve3.setData(self.t,gyro_z)
        self.curve4.setData(self.freqs,np.abs(fx)) 
        self.curve5.setData(self.freqs,np.abs(fy)) 
#        self.curve6.setData(freqs,argy)

        self.curve7.setData(phi)
        self.curve8.setData(phase_x)

        # filtered curves
        #self.curve12.setData(t1,xl)
        #self.curve22.setData(t1,yl) 
