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

        self.pw1 = pg.PlotWidget(title="acc_x, acc_y")
        self.pw2 = pg.PlotWidget(title="gyro_z")
        self.pw3 = pg.PlotWidget(title="fft(x)")
        self.pw4 = pg.PlotWidget(title="fft(y)")
        self.pw5 = pg.PlotWidget(title="dphi")
        self.pw6 = pg.PlotWidget(title="phi")
        
        layout.addWidget(self.pw1, 0,0)
        layout.addWidget(self.pw2, 0,1)
        layout.addWidget(self.pw3, 0,2)
        layout.addWidget(self.pw4, 1,0)
        layout.addWidget(self.pw5, 1,1)
        layout.addWidget(self.pw6, 1,2)

        self.setLayout(layout)

        self.curve1 = self.pw1.plot(pen="w")
        self.curve2 = self.pw1.plot(pen="r")
        self.curve3 = self.pw2.plot(pen="w")
        self.curve4 = self.pw3.plot(pen="w")
        self.curve5 = self.pw4.plot(pen="w")

        self.curve6 = self.pw5.plot(symbol = "o", symbolSize=5, symbolBrush="r")
        self.curve7 = self.pw5.plot(symbol ="o", symbolSize=5, symbolBrush="b")
        self.curve8 = self.pw5.plot(pen="g")

        # heading + roll
        self.curve9 = self.pw6.plot(pen="y")
#        self.curveA = self.pw6.plot(pen="g")

        # data buffers
        self.accx_buf = deque()
        self.accy_buf = deque()
        self.accz_buf = deque()
        self.gyrox_buf = deque()
        self.gyroy_buf = deque()
        self.gyroz_buf = deque()

        self.received_data = np.empty((6,0)) 

        self.windowSize = 500
        self.win_accx = np.zeros(2*self.windowSize)
        self.win_accy = np.zeros(2*self.windowSize)
        #self.win_accz = np.zeros(2*self.windowSize)
        self.win_phasex= np.zeros(2*self.windowSize)
        self.win_phasey= np.zeros(2*self.windowSize)
        self.win_gx = np.zeros(2*self.windowSize)
        self.win_gy = np.zeros(2*self.windowSize)
        self.win_gz = np.zeros(2*self.windowSize)
        self.win_phase = np.zeros(2*self.windowSize)
        self.win_dphi = np.zeros(2*self.windowSize)

        self.win_heading = np.zeros(2*self.windowSize)
        self.win_roll = np.zeros(2*self.windowSize)


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

        self.freqs = fftfreq(self.windowSize, d = 1/self.fs)
        th = 0.5 # mask anything above 1 Hz 
        self.mask = self.freqs > th

        self.writer = file_writer

    
    def read_data(self, new_data):
        import datetime
        self.received_data = np.array(new_data).reshape(6,-1)
        print(f"[DEBUG] Received data shape: {self.received_data.shape}")
        print(f"[DEBUG] First row: {self.received_data[:,0] if self.received_data.shape[1] > 0 else 'N/A'}")
        if self.writer:
            # Add timestamp to each row
            for row in self.received_data.T:
                timestamp = datetime.datetime.now().isoformat()
                row_with_time = [timestamp] + row.tolist()
                print(f"[DEBUG] Writing row: {row_with_time}")
                self.writer.writerow(row_with_time)

        self.accx_buf.extend(self.received_data[0])
        self.accy_buf.extend(self.received_data[1])
        #self.accz_buf.extend(self.received_data[2])
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
        #self.accz_buf.extend(self.received_data[2])
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
        #new_z = self.accz_buf.popleft()
        new_gx = self.gyrox_buf.popleft()
        new_gy = self.gyroy_buf.popleft()
        new_gz = self.gyroz_buf.popleft()

        self.win_accx[j]   = new_x
        self.win_accx[j+N] = new_x
        self.win_accy[j]   = new_y
        self.win_accy[j+N] = new_y

#        self.win_accz[j]   = new_z
#        self.win_accz[j+N] = new_z
        self.win_gx[j]   = new_gx
        self.win_gx[j+N] = new_gx
        self.win_gy[j]   = new_gy
        self.win_gy[j+N] = new_gy
        self.win_gz[j]   = new_gz
        self.win_gz[j+N] = new_gz

        acc_x = self.win_accx[j+1:j+N+1] # current window
        acc_y = self.win_accy[j+1:j+N+1]

        # calculate fft, arguments
        filtered_x = filtfilt(self.b, self.a, acc_x)
        filtered_y = filtfilt(self.b, self.a, acc_y)

        fx = np.fft.fft(filtered_x)
        fy = np.fft.fft(filtered_y)

        # DC masking
        masked_indices = np.where(self.mask)[0] # [0] gives indices of True condition
        peak_idx = masked_indices[np.argmax(np.abs(fx[self.mask]))] # index in full array corresponding to wanted frequency

        #combined = np.abs(fx)**2 + np.abs(fy)**2
        #peak_idx = np.argmax(combined[self.mask])
        #peak_idx = masked_indices[peak_idx]

        X = fx[peak_idx] # magnitudes at peak frequency
        Y = fy[peak_idx]
        phase  = np.angle(X + 1j*Y)
        
        phase_x = np.angle(fx[peak_idx]) 
        phase_y = np.angle(fy[peak_idx])

        self.win_phasex[j]   = phase_x
        self.win_phasex[j+N] = phase_x
        self.win_phasey[j]   = phase_y
        self.win_phasey[j+N] = phase_y
        self.win_phase[j]    = phase
        self.win_phase[j+N]  = phase

        heading_rate = new_gx*np.cos(phase)-new_gy*np.sin(phase)
        roll_rate    = new_gx*np.sin(phase)+new_gy*np.cos(phase)

        peak_freq = self.freqs[peak_idx]
        dphi = self.freqs[peak_idx]*360

        #print(f"frequency: {peak_freq}, DPS: {dphi}")

        self.win_dphi[j]     = dphi
        self.win_dphi[j+N]   = dphi

        self.win_heading[j]  = heading_rate
        self.win_heading[j+N]= heading_rate
        self.win_roll[j]     = roll_rate
        self.win_roll[j+N]   = roll_rate

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
        acc_x   = self.win_accx[j+1:j+N+1] # current window
        acc_y   = self.win_accy[j+1:j+N+1]
 #       acc_z   = self.win_accz[j+1:j+N+1]
        phase_x = self.win_phasex[j+1:j+N+1]
        phase_y = self.win_phasey[j+1:j+N+1]
        gx      = self.win_gx[j+1:j+N+1]
        gy      = self.win_gy[j+1:j+N+1]
        gz      = self.win_gz[j+1:j+N+1]
        phi     = self.win_phase[j+1:j+N+1]
        dphi    = self.win_dphi[j+1:j+N+1]

        heading_rate = self.win_heading[j+1:j+N+1]
        roll_rate    = self.win_roll[j+1:j+N+1]

        fx = fftshift(np.fft.fft(acc_x))
        fy = fftshift(np.fft.fft(acc_y))

        freqs = fftshift(self.freqs)

        # update
        self.curve1.setData(self.t,acc_x) # ax
        self.curve2.setData(self.t,acc_y) # ay
        self.curve3.setData(self.t,gz)
        self.curve4.setData(freqs,np.abs(fx)) 
        self.curve5.setData(freqs,np.abs(fy)) 

        #self.curve6.setData(phase_x)
        #self.curve7.setData(phase_y)
        self.curve9.setData(phi)

        self.curve8.setData(dphi)
        #self.curveA.setData(roll_rate)
