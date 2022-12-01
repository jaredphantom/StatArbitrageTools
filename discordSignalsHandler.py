from subprocess import Popen
import time
from datetime import datetime as dt

minute = 60
hour = minute * 60
day = hour * 24

while True:
    if dt.today().hour == 20:
        while True:
            proc = Popen(["python", "discordSignals.py"])
            time.sleep(day)
            proc.wait()
    time.sleep(minute * 10)
