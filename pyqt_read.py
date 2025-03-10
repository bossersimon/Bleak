
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
import sys

address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"


class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(float)

    def __init__(self, address):
        super().__init__()
        self.address = address
        self.loop = None

# called when new data is received
    async def notification_handler(self, sender, data):
        x, y, z = struct.unpack('<fff', data)
        print(f"received data: {x, y, z}")
        self.data_received.emit(x)
    
    async def read_BLE(self):
        # Connect to ESP
        async with BleakClient(address) as client:
            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)

            # keep script running
            while True:
                await asyncio.sleep(0.1)  # Prevents blocking, allows other tasks to run

    def run(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.read_BLE())


class RealTimePlot:

    def __init__(self):

# Application for managing GUI application
        self.app = pg.mkQApp("Testplot")
        self.pw1 = pg.PlotWidget(show=True)
        self.pw1.setWindowTitle("Plot1")
        #pw1.setLabel("bottom", " ")
        #pw1.setLabel("left", " ")

# create empty data buffer
        self.curve = self.pw1.plot(pen="b")
        self.data= np.zeros(100)

# timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update) 
        self.timer.start(50) # 50ms update

        self.thread = QtCore.QThread()
        self.ble_worker = BLEWorker(address)
        self.ble_worker.moveToThread(self.thread)

        self.ble_worker.data_received.connect(self.read_data)

        self.thread.started.connect(self.ble_worker.run)
        self.thread.start()
        
        self.latest_data = 0.0

    def read_data(self, new_data):
        self.latest_data = new_data

    def update(self):
        self.data = np.roll(self.data, -1)
        self.data[-1] = self.latest_data
        self.curve.setData(self.data) # update



#async def main():

if __name__ == "__main__":
    plot = RealTimePlot()
    sys.exit(plot.app.exec())

#asyncio.run(main())

