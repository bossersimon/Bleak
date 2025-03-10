
# pyqt_read.py

import asyncio
import struct
from bleak import BleakClient
#import csv
#from multiprocessing import Process, Queue
#import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from PyQt6.QtWidgets import QApplication
import sys
import qasync

address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"


class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(float)

    def __init__(self, address, loop: asyncio.AbstractEventLoop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address
        self.client = None

# called when new data is received
    async def notification_handler(self, sender, data):
        x, y, z = struct.unpack('<fff', data)
        print(f"received data: {x, y, z}")
        self.data_received.emit(x)
    
    async def read_BLE(self):
        # Connect to ESP
        async with BleakClient(self.address) as client:
            await client.start_notify(CHARACTERISTIC_UUID, self,notification_handler)

            while True:
                await asyncio.sleep(0.1)
        

    # create tasks
    def start_BLE(self):
        asyncio.ensure_future(self.read_BLE(), loop=self.loop)

class RealTimePlot:

    def __init__(self, parent = None):

# Application for managing GUI application
        #self.app = pg.mkQApp("Testplot")
        self.pw = pg.PlotWidget(show=True)
        self.pw.setWindowTitle("Plot1")
        #pw1.setLabel("bottom", " ")
        #pw1.setLabel("left", " ")

# create empty data buffer
        self.curve = self.pw.plot(pen="b")
        self.data= np.zeros(100)

        loop = asyncio.get_event_loop()
        self.ble_worker = BLEWorker(address, loop)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_BLE()
        
        self.latest_data = 0.0

    def read_data(self, new_data):
        self.latest_data = new_data

    def update(self):
        self.data = np.roll(self.data, -1)
        self.data[-1] = self.latest_data
        self.curve.setData(self.data) # update


if __name__ == "__main__":

    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    plot = RealTimePlot()
    
    with loop:
        loop.run_forever()

    #sys.exit(plot.app.exec())

