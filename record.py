
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
import datetime

address = "DC:1E:D5:1B:E9:FE" # ESP MAC address


def setup_csv():
    #f = open("recording.txt", "a", newline="")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.csv"
    print(f"[INFO] Recording to: {filename}")
    f = open(filename, "a", newline="")
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
#    generate_signals(plot)

    plot.show()
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_ble) # Ensures GUI is fully initialized (may also work without singleShot)
    
    with loop:
        loop.run_forever()

