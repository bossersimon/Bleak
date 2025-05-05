
# g_estimator.py

import numpy as np
import pyqtgraph as pg
from scipy.fft import fft, ifft, fftshift, fftfreq
from scipy.signal import butter,lfilter,filtfilt
import math
from pyqtgraph.Qt import QtCore

import asyncio
import qasync
import signal

from plotwindow import PlotWindow

address = "DC:1E:D5:1B:E9:FE" # ESP MAC address

def read_recording(plot):
    loaded_data = np.loadtxt("recordings/screwdriver2.txt", delimiter = ",")
    loaded_data = loaded_data.T
    plot.recorded_data = loaded_data

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
    app = pg.mkQApp()
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    playback = True
    plot = PlotWindow(loop, playback)
    setup_graceful_shutdown(loop,plot)

    #read_recording(plot)
    plot.show()
    QtCore.QTimer.singleShot(0,plot.ble_worker.start_ble)

    with loop:
        loop.run_forever()
  #  app.exec()

