
# g_estimator.py

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
from scipy.fft import fft, ifft, fftshift, fftfreq

import csv
import math

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

        # create empty data buffers
        self.bufferSize = 500
        self.data1= np.zeros(self.bufferSize)
        self.data2= np.zeros(self.bufferSize)
        self.data3= np.zeros(self.bufferSize)
        self.data4= np.zeros(self.bufferSize)
        self.data5= np.zeros(self.bufferSize)
        self.data6= np.zeros(self.bufferSize)
        self.latest_data = np.empty((6,0)) 

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        self.counter = 0
        
    def update(self):
        i = self.counter
        if i<=990:
            chunk_size = 100
            fs = 100
            t1 = np.arange(chunk_size,dtype=float)/fs
            freqs = fftshift(fftfreq(100, d = 1/fs))

            self.data1 = self.latest_data[0]
            self.data2 = self.latest_data[1]

            c1 = self.data1[i*10:i*10+100]
            c2 = self.data2[i*10:i*10+100]

            f1 = fftshift(fft(c1)/len(c1))
            f2 = fftshift(fft(c2)/len(c2))

            f1[np.abs(f1)<1e-6]=0
            f2[np.abs(f2)<1e-6]=0

            freqs = fftshift(fftfreq(len(c1), d = 1/fs))

            argx = np.angle(f1)
            argy = np.angle(f2)

            self.data3 = np.abs(f1)
            self.data4 = np.abs(f2)
            self.data5 = argx
            self.data6 = argy

            """
            self.data3 = self.latest_data[2]
            self.data4 = self.latest_data[3]
            self.data5 = self.latest_data[4]
            self.data6 = self.latest_data[5]
            """

            """
            # shift old data
            self.data1 = np.roll(self.data1, -chunk_size)
            self.data2 = np.roll(self.data2, -chunk_size)
            self.data3 = np.roll(self.data3, -chunk_size)
            self.data4 = np.roll(self.data4, -chunk_size)

            # read new data into data buffers
            self.data1[-chunk_size:] = self.latest_data[0]
            self.data2[-chunk_size:] = self.latest_data[1]
            self.data3[-chunk_size:] = self.latest_data[2]
            self.data4[-chunk_size:] = self.latest_data[3]
            """

                # update
            self.curve1.setData(t1,self.data1[i*10:i*10+100])
            self.curve2.setData(t1,self.data2[i*10:i*10+100]) 
            self.curve3.setData(freqs,self.data3) 
            self.curve4.setData(freqs,self.data4) 
            self.curve5.setData(freqs,self.data5) 
            self.curve6.setData(freqs,self.data6) 

        self.counter +=1

#        self.latest_data = np.empty((4, 0))

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


def generate_signals(plot):
    fs = 100 # sampling frequency
    T = 100 # signal length
    N = T*fs # number of samples

#    window = np.hanning(1000)
    window = 1
    t1 = np.arange(N,dtype=float)/fs
    c1 = np.random.normal(size=N)
    c1 += 10*np.sin(2*np.pi *30*t1)*window

    c2 = np.random.normal(size=N)
    c2 += 10*np.cos(2*np.pi*30*t1)*window

    f1 = fftshift(fft(c1)/len(c1))
    f2 = fftshift(fft(c2)/len(c2))

    f1[np.abs(f1)<1e-6]=0
    f2[np.abs(f2)<1e-6]=0

    freqs = fftshift(fftfreq(len(c1), d = 1/fs))

    argx = np.angle(f1)
    argy = np.angle(f2)

#    argx= np.unwrap(argx)
#    argy= np.unwrap(argy)

    rows = np.array([c1,c2,np.abs(f1),np.abs(f2), argx, argy])
    plot.latest_data = rows


if __name__ == "__main__":
    app = pg.mkQApp()
    plot = PlotWindow()
    generate_signals(plot)
    plot.show()
    app.exec()


    """
    with open("recording.txt") as file:
#        print(file.read())
        lines = file.readlines()

        for line in lines:
            datapoint = (line.strip()).split(",")
    """
