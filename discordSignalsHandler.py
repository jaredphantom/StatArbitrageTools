from subprocess import call
import time

minute = 60
hour = minute * 60
day = hour * 24

while True:
    call(["python", "discordSignals.py"])
    time.sleep(day)
