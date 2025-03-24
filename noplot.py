
# noplot.py

import asyncio
import struct
from bleak import BleakClient
import numpy as np
#import qasync

address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"

#x_data_buffer = np.zeros(100)
#y_data_buffer = np.zeros(100)
#z_data_buffer = np.zeros(100)

# called when new data is received
async def notification_handler(sender, data):
#    x, y, z = struct.unpack('<fff', data)
    ax, ay, az, gx, gy, gz = struct.unpack('>hhhhhh', data) # big endian
    print(f"received data: {ax, ay, az, gx, gy, gz}")


async def read_BLE():
    # Connect to ESP
    async with BleakClient(address) as client:
        #mtu = await client.exchange_mtu(517)  # Request max MTU
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        while True:
            await asyncio.sleep(0.01)

async def main():
    ble_task = asyncio.create_task(read_BLE())
    await ble_task

asyncio.run(main())
