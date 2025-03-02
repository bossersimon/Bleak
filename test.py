

import asyncio
from bleak import BleakClient, BleakScanner
import matplotlib.pyplot as plt
from collections import deque

SERVICE_UUID        = "9fc7cd06-b6aa-492d-9991-4d5a433023e5"
CHARACTERISTIC_UUID = "c1756f0e-07c7-49aa-bd64-8494be4f1a1c"


# Data buffer (stores last 100 values)
buffer_size = 100
x_data = deque(maxlen=buffer_size)
y_data = deque(maxlen=buffer_size)
z_data = deque(maxlen=buffer_size)

# Initialize real-time plot
plt.ion()
fig, ax = plt.subplots()
line_x, = ax.plot([], [], label="X-axis")
line_y, = ax.plot([], [], label="Y-axis")
line_z, = ax.plot([], [], label="Z-axis")

ax.set_xlim(0, buffer_size)
ax.set_ylim(-20000, 20000)  # Adjust according to gyro range
ax.legend()

async def notification_handler(sender, data):
    """Handles incoming BLE data from ESP32."""
    # Convert 6-byte data to 3-axis int16 values
    gx = int.from_bytes(data[0:2], byteorder='little', signed=True)
    gy = int.from_bytes(data[2:4], byteorder='little', signed=True)
    gz = int.from_bytes(data[4:6], byteorder='little', signed=True)

    print(f"Gyro Data: X={gx}, Y={gy}, Z={gz}")

    # Update buffer
    x_data.append(gx)
    y_data.append(gy)
    z_data.append(gz)

    # Update plot
    line_x.set_ydata(list(x_data))
    line_y.set_ydata(list(y_data))
    line_z.set_ydata(list(z_data))
    line_x.set_xdata(range(len(x_data)))
    line_y.set_xdata(range(len(y_data)))
    line_z.set_xdata(range(len(z_data)))

    plt.draw()
    plt.pause(0.01)

async def main():
    # Scan for ESP32 BLE device
    devices = await BleakScanner.discover()
    esp_device = next((d for d in devices if "MyESP32" in d.name), None)

    if not esp_device:
        print("ESP32 BLE device not found.")
        return

    print(f"Connecting to {esp_device.address}...")

    # Connect to ESP32 BLE
    async with BleakClient(esp_device.address) as client:
        print("Connected! Subscribing to notifications...")

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        # Keep running until user stops
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("Disconnecting...")
            await client.stop_notify(CHARACTERISTIC_UUID)

# Run the async loop
asyncio.run(main())

"""
async def main():
    stop_event = asyncio.Event()

    # TODO: add something that calls stop_event.set()

    def callback(device, advertising_data):
        # TODO: do something with incoming data
        pass

    async with BleakScanner(callback) as scanner:
        ...
        # Important! Wait for an event to trigger stop, otherwise scanner
        # will stop immediately.
        await stop_event.wait()

    # scanner stops when block exits
    ...

asyncio.run(main())
"""
