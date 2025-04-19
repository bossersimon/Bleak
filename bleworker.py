
# BLEWorker class

from pyqtgraph.Qt import QtCore
from bleak import BleakClient
import asyncio


# Manages BLE communication and reads data asynchronously
class BLEWorker(QtCore.QObject):
    data_received = QtCore.pyqtSignal(list)

    def __init__(self, loop, address, parent = None):
        super().__init__()
        self.loop = loop
        self.address = address
        self.client = BleakClient(self.address)

    """
    def set_address(self, address):
        self.address = address
        self.client = BleakClient(self.address)
    """
    # Called when new data is received
    async def notification_handler(self, sender, data):
        print("Notification received")
        received = self.convert_to_float(data)
        #print(f"shape received: {np.shape(received)}\n")
        self.data_received.emit(received.flatten().tolist()) # emits to PlotWindow.read_data()


    async def read_ble(self):
        # Connect to ESP
        async with self.client as client:

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
                print("BLE loop alive")
                await asyncio.sleep(0.1)

    # create tasks
    def start_ble(self):
        asyncio.run_coroutine_threadsafe(self.read_ble(), self.loop)  # Submit coroutine to loop 

    def convert_to_float(self, buffer):
        # Scale and bias correction from raw data
        data_arr = np.frombuffer(buffer, dtype='>i2').astype(np.float32)
        data_arr = data_arr.reshape(-1,6)
        scaled = data_arr / dividers - bias_values
        scaled = scaled.T

        return scaled


