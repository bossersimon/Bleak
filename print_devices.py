


import asyncio
from bleak import BleakScanner
#import bleak

async def main():
#    print("Bleak version:", bleak.__version__)
    scanner = BleakScanner(adapter="hci0")
    scanner._scanner_kwargs = {"duplicate": True}
    devices = await scanner.discover(timeout=20.0)
    for d in devices:
        print(d)

asyncio.run(main())
