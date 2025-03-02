


import asyncio
from bleak import BleakScanner

async def main():

    scanner = BleakScanner(adapter="hci0", advertisement_data = True)
    scanner._scanner_kwargs = {"duplicate": True}  # Try forcing duplicate scanning
    devices = await scanner.discover()
    #devices = await BleakScanner.discover(20.0, return_adv=True)
    for d in devices:
        print(d)

asyncio.run(main())
