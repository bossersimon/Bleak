
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

    #signal.signal(signal.SIGINT, signal.SIG_DFL)

    f,writer = setup_csv()
    plot = PlotWindow(loop, writer)
    setup_graceful_shutdown(loop,plot)

    plot.show()
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_ble) # Ensures GUI is fully initialized (may also work without singleShot)
    
    with loop:
        loop.run_forever()

    #sys.exit(plot.app.exec())

