"""Retry only transient Office automation busy responses."""
import time

BUSY_HRESULTS = {-2147418111, -2147417846}


def call_office(action, timeout=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return action()
        except Exception as error:
            if getattr(error, 'hresult', None) not in BUSY_HRESULTS or time.monotonic() >= deadline:
                raise
            import pythoncom
            pythoncom.PumpWaitingMessages()
            time.sleep(0.2)
