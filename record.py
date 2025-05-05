
# pyqt_read.py

import asyncio
from bleak import BleakClient
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
import qasync
import csv
import signal

from plotwindow import PlotWindow


address = "DC:1E:D5:1B:E9:FE" # ESP MAC address

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

    rows = np.array([c1,c2,np.abs(f1),np.abs(f2), argx, argy])
    plot.received_data = rows

def setup_csv():
    f = open("recording.txt", "a", newline="")
    writer = csv.writer(f)
    return f,writer


def setup_graceful_shutdown(loop, plot):
    def signal_handler(*args):
        print("Caught SIGINT, shutting down...")
        loop.create_task(shutdown())

    async def shutdown():
        await plot.cleanup()
        await asyncio.sleep(0.2)
        loop.stop()

    signal.signal(signal.SIGINT,signal_handler)
    app.aboutToQuit.connect(lambda: loop.create_task(shutdown()))
        

if __name__ == "__main__":

    # Application for managing GUI application
    app = pg.mkQApp()
    loop = qasync.QEventLoop(app) 
    asyncio.set_event_loop(loop)

    f,writer = setup_csv()
    plot = PlotWindow(loop, False, writer) # no playback
    setup_graceful_shutdown(loop,plot)
    generate_signals(plot)

    plot.show()
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_ble) # Ensures GUI is fully initialized (may also work without singleShot)
    
    with loop:
        loop.run_forever()

