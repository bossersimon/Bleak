
# btconnect.py

import asyncio
import struct
from bleak import BleakClient
from collections import deque
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import csv


address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"

    # initial plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter([],[],[])
    # Initialize coordinates
gx, gy, gz = [],[],[]
latest_coordinates = None

async def read_BLE(address):
    #dq = deque([]) 

    global latest_coordinates
    # Connect to ESP
    async with BleakClient(address) as client:
        #with open('data.csv', 'a') as file:
         #   csv_writer = csv.writer(file)

            # read data from ESP, write to csv
        while True:
            data = await client.read_gatt_char(CHARACTERISTIC_UUID)
            if data:
                #value = int.from_bytes(data, byteorder='little')
                x, y, z = struct.unpack('<hhh', data)
                latest_coordinates = (x,y,z)
                print(f"received data: {x, y, z}")
            await asyncio.sleep(0.1)  # Prevents blocking, allows other tasks to run

          #          csv_writer.writerow(gx,gy,gz)
                    #dq.extend([gx, gy, gz]);
                    #dq.popleft()

def init():
    ax.set(xlim=[0, 250], ylim=[0, 250], zlim = [0, 250],\
            xlabel='gx', ylabel='gy', zlabel = 'gz')
    return scatter,

def update(frame):
    # update data
    #data = pd.read_csv('data.csv')
    #data = data.tail(1)
#    x = data[0]
#    y = data[1]
#    z = data[2]
    if latest_coordinates != None:
        x, y, z = latest_coordinates
        gx.append(x)
        gy.append(y)
        gz.append(z)

    #ax.scatter(gx, gy, gz)
    scatter._offsets3d = (gx, gy, gz)
    #scatter(gx ,gy, gz)
    #ax.legend()
    # update plot
    #data = np.stack([x,y,z]).T
    return scatter,


async def main(address):
    ble_task = asyncio.create_task(read_BLE(address))

    # calls the animator
    plt.ion()
    ani = animation.FuncAnimation(fig=fig, func=update, frames = None, init_func = init)

    while True:
        plt.draw()  # Redraw the plot
        plt.pause(0.1)  # Pause briefly to allow BLE reading to run
        await asyncio.sleep(0.01)  # Allow async event loop to process other tasks (non-blocking)

    await ble_task


asyncio.run(main(address))

