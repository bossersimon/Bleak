
# btconnect.py

import asyncio
import struct
from bleak import BleakClient
from collections import deque
import matplotlib.pyplot as plt
#from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import csv
from multiprocessing import Process, Queue

#import time
import numpy as np

address = "28:37:2F:6A:B1:42"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"

    # initial plot 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
scatter = ax.scatter([],[],[])
    # Initialize coordinates
gx, gy, gz = [],[],[]
latest_coordinates = None
dt = 0.01
velocity = np.array([0.0, 0.0, 0.0])
pos = np.array([0.0, 0.0, 0.0])


# called when new data is received
async def notification_handler(sender, data):
    x, y, z = struct.unpack('<fff', data)

    velocity += [x, y, z] * dt
    pos += velocity * dt

    latest_coordinates = (x,y,z)
    print(f"received data: {x, y, z}")
    

async def read_BLE(address):
    #dq = deque([]) 
    global latest_coordinates
    # Connect to ESP
    async with BleakClient(address) as client:
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        # keep script running
        while True:
            await asyncio.sleep(0.1)  # Prevents blocking, allows other tasks to run
                    #dq.extend([gx, gy, gz]);
                    #dq.popleft()

def init():
    ax.set(xlim=[-10, 10], ylim=[-10, 10], zlim = [-10, 10],\
            xlabel='gx', ylabel='gy', zlabel = 'gz')
    return scatter,

def update(frame):

    if latest_coordinates != None:
        x, y, z = latest_coordinates
        gx.append(x)
        gy.append(y)
        gz.append(z)

    scatter._offsets3d = (gx, gy, gz)

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

