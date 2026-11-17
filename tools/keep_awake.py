"""Keep Windows awake while a given command pattern is running (no settings changed).
Usage: python tools/keep_awake.py <substring of the command line to watch>"""
import ctypes
import subprocess
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
pattern = sys.argv[1]


def running() -> bool:
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          f"(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          f"Where-Object {{ $_.CommandLine -match '{pattern}' }}).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out.isdigit() and int(out) > 0


ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(f"keep-awake on while '{pattern}' runs", flush=True)
misses = 0
while misses < 3:                      # tolerate the short gap between two consecutive runs
    time.sleep(60)
    misses = 0 if running() else misses + 1
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print("keep-awake off", flush=True)
