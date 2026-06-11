# control_flags.py
import threading

# set()  => paused
# clear() => running
pause_event = threading.Event()

# set()  => request all threads to stop ASAP
stop_event = threading.Event()
