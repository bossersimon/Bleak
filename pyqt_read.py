
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
#from PyQt6.QtWidgets import QApplication
#from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton
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
    data_received = QtCore.pyqtSignal(list)

    def __init__(self, address, loop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address
       # self.client = None

# called when new data is received
    async def notification_handler(self, sender, data):
        # Multiple data points at a time:

#        received = np.array(bytearray(data), dtype=np.byte)
#        received.shape = (len(received)/12, 12)
        
        #received = [struct.unpack('>hhhhhh', data[i:i+12]) for i in range(0,len(data), 12)]
        received = convert_to_float(data)
        print(f"shape received: {np.shape(received)}\n")

#        self.data_received.emit(received)


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
            bias = [int.from_bytes(param_data[i:i+2], 'little', signed=True) / 100 for i in range(0, len(param_data), 2)]

            print("Adjustment values:", bias)

            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            print("start notify complete")

            while True:
                await asyncio.sleep(0.1)

    # create tasks
    def start_BLE(self):
        asyncio.run_coroutine_threadsafe(self.read_BLE(), self.loop)  # Delayed execution
        #asyncio.create_task(self.read_BLE())

class PlotWindow(QWidget):

    def __init__(self, loop):

        super().__init__()

        layout = QGridLayout()  # Create a layout

        self.pw1 = pg.PlotWidget(title="ax")
        self.pw2 = pg.PlotWidget(title="ay")
        self.pw3 = pg.PlotWidget(title="az")
        self.pw4 = pg.PlotWidget(title="gx")
        self.pw5 = pg.PlotWidget(title="gy")
        self.pw6 = pg.PlotWidget(title="gz")
        
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

# create empty data buffer
        self.data1= np.zeros(100)
        self.data2= np.zeros(100)
        self.data3= np.zeros(100)
        self.data4= np.zeros(100)
        self.data5= np.zeros(100)
        self.data6= np.zeros(100)

#        loop = asyncio.get_running_loop()
        self.loop = loop
        self.ble_worker = BLEWorker(address, loop)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_BLE()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        
        self.latest_data = [0,0,0,0,0,0] 

    def read_data(self, new_data):
        # new_data of unknown size
        self.latest_data = new_data

    def update(self):
        self.data1 = np.roll(self.data1, -1)
        self.data2 = np.roll(self.data2, -1)
        self.data3 = np.roll(self.data3, -1)
        self.data4 = np.roll(self.data4, -1)
        self.data5 = np.roll(self.data5, -1)
        self.data6 = np.roll(self.data6, -1)
        self.data1[-1] = self.latest_data[0]
        self.data2[-1] = self.latest_data[1]
        self.data3[-1] = self.latest_data[2]
        self.data4[-1] = self.latest_data[3]
        self.data5[-1] = self.latest_data[4]
        self.data6[-1] = self.latest_data[5]

        self.curve1.setData(self.data1) # update
        self.curve2.setData(self.data2) # update
        self.curve3.setData(self.data3) # update
        self.curve4.setData(self.data4) # update
        self.curve5.setData(self.data5) # update
        self.curve6.setData(self.data6) # update

def convert_to_float(buffer):
    
    #data = np.array(buffer, dtype=np.int16)
    #scaled = [data[i:i+12].astype(np.float32) / [16384, 16384, 16384, 131, 131,131] - bias for i in range(0, len(data), 12)]
    

    data_arr = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
    data_arr = data_arr.reshape(-1,6)
    scaled = (data_arr / [16384,16384,16384,131,131,131]) - bias
    scaled = scaled.T

    return scaled


if __name__ == "__main__":

# Application for managing GUI application
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    plot = PlotWindow(loop)
    plot.show()
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_BLE)
    
    with loop:
        loop.run_forever()

    #sys.exit(plot.app.exec())

