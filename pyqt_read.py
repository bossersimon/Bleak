
# pyqt_read.py

import asyncio
from bleak import BleakClient
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QGridLayout
import sys
import qasync
import csv

address = "DC:1E:D5:1B:E9:FE" # ESP MAC address
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c" # Data characteristic
PARAMS_UUID = "97b28d55-f227-4568-885a-4db649a8e9fd" # Parameter characteristic

# Scale parameters
acc_divider = 16384
gyro_divider = 131
dividers = [acc_divider, acc_divider, acc_divider, gyro_divider, gyro_divider, gyro_divider]
bias_values = [0,0,0]

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

# Manages BLE communication and reads data asynchronously
class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(list)

    def __init__(self, address, loop, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address #ESP MAC

    # Called when new data is received
    async def notification_handler(self, sender, data):
        
        received = convert_to_float(data)
        #print(f"shape received: {np.shape(received)}\n")
        
        self.data_received.emit(received.flatten().tolist()) # emits to read_data()


    async def read_ble(self):
        # Connect to ESP
        async with BleakClient(self.address) as client:

            #a_scale = ACC_SCALES["2G"]
            #g_scale = GYRO_SCALES["250DPS"]

            # send scale parameters to ESP
            #scales = bytes([a_scale, g_scale], 'big')
            #await client.write_gatt_char(PARAMS_UUID, scales)

            # for bias and scale correction
            param_data = await client.read_gatt_char(PARAMS_UUID)
            global bias_values
            bias_values = [int.from_bytes(param_data[i:i+2], 'little', signed=True) / 100 for i in range(0, len(param_data), 2)]

            # print("Adjustment values:", bias_values)

            await client.start_notify(CHARACTERISTIC_UUID, self.notification_handler)
            
            # print("start notify complete")

            while True:
                await asyncio.sleep(0.1)

    # create tasks
    def start_ble(self):
        asyncio.run_coroutine_threadsafe(self.read_ble(), self.loop)  # Submit coroutine to loop 

class PlotWindow(QWidget):

    def __init__(self, loop, file_writer):

        super().__init__()

        layout = QGridLayout()

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

        # create empty data buffers
        self.bufferSize = 500
        self.data1= np.zeros(self.bufferSize)
        self.data2= np.zeros(self.bufferSize)
        self.data3= np.zeros(self.bufferSize)
        self.data4= np.zeros(self.bufferSize)
        self.data5= np.zeros(self.bufferSize)
        self.data6= np.zeros(self.bufferSize)
        self.latest_data = np.empty((6,0)) 

        self.loop = loop
        self.ble_worker = BLEWorker(address, loop)
        self.ble_worker.data_received.connect(self.read_data)
        self.ble_worker.start_ble()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)

        # Recording
        self.recording = np.empty((0,6))
        
        self.writer = file_writer

    def read_data(self, new_data):
        self.latest_data = np.array(new_data).reshape(6,-1)
        self.writer.writerows(self.latest_data.T)

    def update(self):

        if (self.latest_data.size > 0):
            chunk_size = self.latest_data.shape[1]

            # shift old data
            self.data1 = np.roll(self.data1, -chunk_size)
            self.data2 = np.roll(self.data2, -chunk_size)
            self.data3 = np.roll(self.data3, -chunk_size)
            self.data4 = np.roll(self.data4, -chunk_size)
            self.data5 = np.roll(self.data5, -chunk_size)
            self.data6 = np.roll(self.data6, -chunk_size)

            # read new data into data buffers
            self.data1[-chunk_size:] = self.latest_data[0]
            self.data2[-chunk_size:] = self.latest_data[1]
            self.data3[-chunk_size:] = self.latest_data[2]
            self.data4[-chunk_size:] = self.latest_data[3]
            self.data5[-chunk_size:] = self.latest_data[4]
            self.data6[-chunk_size:] = self.latest_data[5]

            # update
            self.curve1.setData(self.data1)
            self.curve2.setData(self.data2) 
            self.curve3.setData(self.data3) 
            self.curve4.setData(self.data4) 
            self.curve5.setData(self.data5)  
            self.curve6.setData(self.data6)

            self.latest_data = np.empty((6, 0))

def convert_to_float(buffer):
    
    data_arr = np.frombuffer(buffer, dtype='>i2').astype(np.float32)
    data_arr = data_arr.reshape(-1,6)
    scaled = data_arr / dividers - bias_values
    scaled = scaled.T

    return scaled

def setup_csv():
    f = open("recording.txt", "a", newline="")
    writer = csv.writer(f)
    return f,writer

if __name__ == "__main__":

    # Application for managing GUI application
    app = QApplication(sys.argv)
    loop = qasync.QEventLoop(app) 
    asyncio.set_event_loop(loop)

    f,writer = setup_csv()
    plot = PlotWindow(loop, writer)
    plot.show()
    QtCore.QTimer.singleShot(0, plot.ble_worker.start_ble) # Ensures GUI is fully initialized (may also work without singleShot)
    
    with loop:
        loop.run_forever()

    #sys.exit(plot.app.exec())

