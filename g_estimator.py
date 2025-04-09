
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
#import sys
import qasync
import signal

"""
address = "DC:1E:D5:1B:E9:FE" # ESP MAC address
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c" # Data characteristic
PARAMS_UUID = "97b28d55-f227-4568-885a-4db649a8e9fd" # Parameter characteristic

# Scale parameters
acc_divider = 16384
gyro_divider = 131
dividers = [acc_divider, acc_divider, acc_divider, gyro_divider, gyro_divider, gyro_divider]
bias_values = [0,0,0]
"""

class PlotWindow(QWidget):

    def __init__(self):

        super().__init__()

        layout = QGridLayout()

        self.pw1 = pg.PlotWidget(title="x")
        self.pw2 = pg.PlotWidget(title="y")
        self.pw3 = pg.PlotWidget(title="X")
        self.pw4 = pg.PlotWidget(title="Y")
        self.pw5 = pg.PlotWidget(title="arg(X)")
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

        """
        self.loop = loop
        self.ble_worker = BLEWorker(address,loop)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_ble()
        """

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        self.counter = 0

    """
    def read_data(self, new_data):
        self.latest_data = np.array(new_data).reshape(6,-1)
    """
        
    def update(self):
        i = self.counter

        order = 6
        # . One gotcha is that Wn is a fraction of the Nyquist frequency. So if the sampling rate is 1000Hz and you want a cutoff of 250Hz, you should use Wn=0.5
        Wn = 0.5
        b, a = butter(order, Wn, 'low')

        chunk_size = 100
        fs = 100
        t1 = np.arange(chunk_size,dtype=float)/fs

        freqs = fftshift(fftfreq(100, d = 1/fs))

        self.data1 = self.latest_data[0] # acc_x
        self.data2 = self.latest_data[1] # acc_y

        max_idx = int(np.shape(self.latest_data)[1]/10)//10 *10 

        if i<max_idx-2:

            c1 = self.data1[i*10:i*10+100]
            c2 = self.data2[i*10:i*10+100]

            d1 = output_signal = filtfilt(b, a, c1)
            d2 = output_signal = filtfilt(b, a, c2)

            f1 = fftshift(fft(c1)/len(c1)) # fft_x
            f2 = fftshift(fft(c2)/len(c2)) # fft_y

            f1[np.abs(f1)<1e-6]=0
            f2[np.abs(f2)<1e-6]=0

            #f1[np.abs(f1) != np.max(np.abs(f1))] = 0
            #f2[np.abs(f2) != np.max(np.abs(f2))] = 0

            freqs = fftshift(fftfreq(len(c1), d = 1/fs))

            argx = np.angle(f1)
            argy = np.angle(f2)

            """
            self.data3 = np.abs(f1)
            self.data4 = np.abs(f2)
            self.data5 = argx
            self.data6 = argy
            """

            """
            self.data3 = self.latest_data[2]
            self.data4 = self.latest_data[3]
            self.data5 = self.latest_data[4]
            self.data6 = self.latest_data[5]
            """

            """ # shift old data self.data1 = np.roll(self.data1, -chunk_size) self.data2 = np.roll(self.data2, -chunk_size) self.data3 = np.roll(self.data3, -chunk_size)
            self.data4 = np.roll(self.data4, -chunk_size)

            # read new data into data buffers
            self.data1[-chunk_size:] = self.latest_data[0]
            self.data2[-chunk_size:] = self.latest_data[1]
            self.data3[-chunk_size:] = self.latest_data[2]
            self.data4[-chunk_size:] = self.latest_data[3]
            """

                # update
            self.curve1.setData(t1,c1)
            self.curve2.setData(t1,c2) 
            self.curve3.setData(freqs,np.abs(f1)) 
            self.curve4.setData(freqs,np.abs(f2)) 
            self.curve5.setData(freqs,argx) 
            self.curve6.setData(freqs,argy) 

            self.curve12.setData(t1,d1)
            self.curve22.setData(t1,d2) 

        self.counter +=1

#        self.latest_data = np.empty((4, 0))

"""
# Manages BLE communication and reads data asynchronously
class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(list)

    def __init__(self, address, loop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address #ESP MAC

    # Called when new data is received
    async def notification_handler(self, sender, data):
        
        received = convert_to_float(data)
        #print(f"shape received: {np.shape(received)}\n")
        
        self.data_received.emit(received.flatten().tolist()) # emits to read_data()


    async def read_ble(self):
        # Connect to ESP
        async with BleakClient(self.address) as client:

            #a_scale = ACC_SCALES["2G"]
            #g_scale = GYRO_SCALES["250DPS"]

            # send scale parameters to ESP
            #scales = bytes([a_scale, g_scale], 'big')
            #await client.write_gatt_char(PARAMS_UUID, scales)

            # for bias and scale correction
            param_data = await client.read_gatt_char(PARAMS_UUID)
            global bias_values
            bias_values = [int.from_bytes(param_data[i:i+2], 'little', signed=True) / 100 for i in range(0, len(param_data), 2)]

            # print("Adjustment values:", bias_values)

            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            # print("start notify complete")

            while True:
                await asyncio.sleep(0.1)

    # create tasks
    def start_ble(self):
        asyncio.run_coroutine_threadsafe(self.read_ble(), self.loop)  # Submit coroutine to loop 



def convert_to_float():
   
    # Scale and bias correction from raw data
    data_arr = np.frombuffer(buffer, dtype='>i2').astype(np.float32)
    data_arr = data_arr.reshape(-1,6)
    scaled = data_arr / dividers - bias_values
    scaled = scaled.T

    # Implement windowing here? 

    # fft (of accelerometer in x, y)
    X = fft(scaled[0,:])
    Y = fft(scaled[1,:])
#    argx = 

    return scaled
"""

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

    f1 = fftshift(fft(d1)/len(d1))
    f2 = fftshift(fft(d2)/len(d2))

    f1[np.abs(f1)<1e-6]=0
    f2[np.abs(f2)<1e-6]=0

    freqs = fftshift(fftfreq(len(d1), d = 1/fs))

    argx = np.angle(f1)
    argy = np.angle(f2)

#    argx= np.unwrap(argx)
#    argy= np.unwrap(argy)

    rows = np.array([c1,c2,np.abs(f1),np.abs(f2), argx, argy])
    plot.latest_data = rows

def read_recording(plot):
    loaded_data = np.loadtxt("recording2.txt", delimiter = ",")
#    print(f"loaded_data shape: {np.shape(loaded_data)}\n")
    
    loaded_data = loaded_data.T

    plot.latest_data= np.empty((6,loaded_data.shape[1]))

 #   plot.latest_data[:2,:] = loaded_data
    plot.latest_data = loaded_data

if __name__ == "__main__":
    app = pg.mkQApp()
    plot = PlotWindow()

    signal.signal(signal.SIGINT, signal.SIG_DFL)
    #generate_signals(plot)
    read_recording(plot)

    plot.show()
    app.exec()
