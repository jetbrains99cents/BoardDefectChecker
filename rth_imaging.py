# rth_imaging.py
import imagingcontrol4.pyside6
try:
    from imagingcontrol4pyside6 import DeviceSelectionDialog
    imagingcontrol4.pyside6.DeviceSelectionDialog = DeviceSelectionDialog
    print("DeviceSelectionDialog patched successfully.")
except Exception as e:
    print("Error patching DeviceSelectionDialog:", e)
