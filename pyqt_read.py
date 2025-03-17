
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
PARAMS_UUID = "97b28d55-f227-4568-885a-4db649a8e9fd"

accel_data = np.array([0,0,0])
gyro_data = np.array([0,0,0])
acc_divider = 16384
gyro_divider = 131
bias = []

GYRO_SCALES = {
    "250DPS": 0x00,
    "500DPS": 0x08,
    "1000DPS": 0x10,
    "2000DPS": 0x18
}

ACC_SCALES = {
    "2G":  0x00,
    "4G":  0x08,
    "8G":  0x10,
    "16G": 0x18
}

class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(float)

    def __init__(self, address, loop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address
       # self.client = None

# called when new data is received
    async def notification_handler(self, sender, data):
        ax, ay, az, gx, gy, gz = convert_to_float(*struct.unpack('>hhhhhh', data))

        print(f"received data: {ax, ay, az, gx, gy, gz}")
        self.data_received.emit(ax)
    
    async def read_BLE(self):
        # Connect to ESP
        async with BleakClient(self.address) as client:

            #a_scale = ACC_SCALES["2G"]
            #g_scale = GYRO_SCALES["250DPS"]

            # send scale parameters to ESP
            #scales = bytes([a_scale, g_scale], 'big')
            #await client.write_gatt_char(PARAMS_UUID, scales)

            # for bias and scale correction
            param_data = await client.read_gatt_char(PARAMS_UUID)
            global bias
            bias = list(int.from_bytes(param_data[i:i+2], 'little', signed=True) / 100 for i in range(0, len(param_data), 2))
            print("Adjustment values:", bias)

            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            print("start notify complete")

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
    
    data = np.array([ax, ay, az, gx, gy, gz], dtype=np.int16)
    scaled = data.astype(np.float32) / [16384, 16384, 16384, 128, 128, 128] - bias
    ax, ay, az, gx, gy, gz = scaled
    """
    ax = float(np.int16(ax >> 14)) - bias[0] 
    ay = float(np.int16(ay >> 14)) - bias[1]
    az = float(np.int16(az >> 14)) - bias[2]
    gx = float(np.int16(gx >> 7)) - bias[3]
    gy = float(np.int16(gy >> 7)) - bias[4]
    gz = float(np.int16(gz >> 7)) - bias[5]    
    """
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

