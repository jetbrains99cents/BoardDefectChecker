import imagingcontrol4 as ic4
from PySide6.QtCore import QThread, Signal

# Create a Worker class for monitoring camera connection
class CameraWorker(QThread):
    connection_status_changed = Signal(bool)  # Signal to emit connection status
    initialization_done = Signal()  # Signal to indicate initialization is complete

    def __init__(self):
        super().__init__()
        self.last_status = None  # Store the last known connection status

    def run(self):
        """Run the camera monitoring logic."""
        self.initialize_camera()  # Wait for camera initialization
        self.initialization_done.emit()  # Emit signal to indicate readiness
        while True:
            self.check_camera_connection()
            self.sleep(1)  # Check every second

    def check_camera_connection(self):
        """Check the camera connection status."""
        device_list = ic4.DeviceEnum.devices()
        is_camera_connection_lost = len(device_list) == 0
        current_status = not is_camera_connection_lost  # True if connected, False if lost

        # Emit the signal only if the status has changed
        if current_status != self.last_status:
            self.last_status = current_status  # Update the last known status
            status = "lost" if is_camera_connection_lost else "established"
            print(f"Camera connection status: {status}.")
            print(f"Emitting connection status: {current_status}")

            # Emit the signal to update the UI
            self.connection_status_changed.emit(current_status)

    def initialize_camera(self):
        """Initialize the camera library."""
        ic4.Library.init(api_log_level=ic4.LogLevel.INFO, log_targets=ic4.LogTarget.STDERR)

def init_camera_connection_handler():
    """Initialize the camera connection worker."""
    _camera_worker = CameraWorker()
    _camera_worker.start()  # Start the worker thread
    return _camera_worker

if __name__ == "__main__":
    camera_worker = init_camera_connection_handler()
    try:
        # Keep the main thread running while the worker does its job
        while True:
            pass  # Replace with any other main loop logic if needed
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        camera_worker.quit()
        camera_worker.wait()  # Wait for the thread to finish
        ic4.Library.exit()