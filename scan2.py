
"""
import asyncio
from bleak import BleakScanner

async def scan():
    def detection_callback(device, advertisement_data):
        print(f"Device: {device.address}, Name: {device.name}")

    scanner = BleakScanner()
    scanner.register_detection_callback(detection_callback)

    await scanner.start()
    await asyncio.sleep(10)  # Scan for 10 seconds
    await scanner.stop()

asyncio.run(scan())
"""

import asyncio
from bleak import BleakScanner

async def scan():
    def detection_callback(device, advertisement_data):
        print(f"Device: {device.address}, Name: {device.name or 'Unknown'}")

    scanner = BleakScanner(advertisement_data=True,adapter = "hci0")
    scanner.register_detection_callback(detection_callback)

    await scanner.start()
    await asyncio.sleep(10)  # Scan for 10 seconds
    await scanner.stop()

asyncio.run(scan())
