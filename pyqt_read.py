
# pyqt_read.py

import asyncio
import struct
from bleak import BleakClient
#import csv
#from multiprocessing import Process, Queue
#import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication
import sys
import qasync

address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"
BIAS_UUID = "97b28d55-f227-4568-885a-4db649a8e9fd"

accel_data = np.array([0,0,0])
gyro_data = np.array([0,0,0])
biases = []

class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(float)

    def __init__(self, address, loop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address
       # self.client = None

# called when new data is received
    async def notification_handler(self, sender, data):
        ax, ay, az, gx, gy, gz = convert_to_float(struct.unpack('>hhhhhh', data))

        #print(f"received data: {ax, ay, az, gx, gy, gz}")
        self.data_received.emit(ax)
    
    async def read_BLE(self):
        # Connect to ESP
        async with BleakClient(self.address) as client:

            # for bias and scale correction
            param_data = await client.read_gatt_char(BIAS_UUID)
            params = list(int.from_bytes(param_data[i:i+2], 'little', signed=True) / 100 for i in range(0, len(param_data), 2))
            print("Adjustment values:", params)

            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)

            while True:
                await asyncio.sleep(0.1)
        

    # create tasks
    def start_BLE(self):
        asyncio.run_coroutine_threadsafe(self.read_BLE(), self.loop)  # Delayed execution
        #asyncio.create_task(self.read_BLE())

class RealTimePlot:

    def __init__(self, loop):

        self.pw = pg.PlotWidget()
        self.pw.setWindowTitle("Plot1")
        #pw1.setLabel("bottom", " ")
        #pw1.setLabel("left", " ")
        self.pw.show()

# create empty data buffer
        self.curve = self.pw.plot(pen="b")
        self.data= np.zeros(100)

#        loop = asyncio.get_running_loop()
        self.loop = loop
        self.ble_worker = BLEWorker(address, loop)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_BLE()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        
        self.latest_data = 0.0

    def read_data(self, new_data):
        self.latest_data = new_data

    def update(self):
        self.data = np.roll(self.data, -1)
        self.data[-1] = self.latest_data
        self.curve.setData(self.data) # update

def convert_to_float(ax, ay, az, gx, gy, gz):
    ax = ax / acc_divider - biases[0] # swap with bit shifts
    ay = ay / acc_divider - biases[1]
    az = az / acc_divider - biases[2]
    gx = gx / gyro_divider - biases[0]
    gy = gy / gyro_divider - biases[1]
    gz = gz / gyro_divider - biases[2]

    return ax, ay, az, gx, gy, gz


if __name__ == "__main__":

# Application for managing GUI application
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    plot = RealTimePlot(loop)
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_BLE)
    
    with loop:
        loop.run_forever()

    #sys.exit(plot.app.exec())

