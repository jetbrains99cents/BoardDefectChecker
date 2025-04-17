import cv2
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QMainWindow, QDialog, QApplication
import imagingcontrol4 as ic4
import os
from datetime import datetime
import time


class CameraInteraction(QThread):
    image_captured = Signal(object)  # Signal to emit when an image is captured
    streaming_started = Signal()  # Signal to indicate streaming has started
    streaming_stopped = Signal()  # Signal to indicate streaming has stopped

    def __init__(self):
        super().__init__()
        self.grabber = None
        self.is_streaming = False
        self.is_device_opened = False
        self.sink = None

    def open_device(self):
        device_list = ic4.DeviceEnum.devices()
        if not device_list:
            print("No devices found!")
            return

        for i, dev in enumerate(device_list):
            print(f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]")

        print(f"Select device [0..{len(device_list) - 1}]: ", end="")
        # selected_index = int(input())
        selected_index = 0
        dev_info = device_list[selected_index]

        # Add delay before opening device
        print("Initializing camera connection...")
        time.sleep(2)

        # Try to open device with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.grabber = ic4.Grabber(dev_info)
                # Set the resolution to 640x480
                # self.grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
                # self.grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)
                break
            except ic4.IC4Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed. Retrying...")
                    time.sleep(2)
                else:
                    raise e

    def select_device(self):
        """Show device selection dialog and select a camera device."""
        self.grabber = ic4.Grabber()

        dlg = ic4.pyside6.DeviceSelectionDialog(self.grabber)

        if dlg.exec() == QDialog.Accepted:  # Use QDialog.Accepted directly
            print("Device selected successfully.")
            # Set the resolution to 640x480
            self.grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
            self.grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)
        else:
            print("No device selected.")
            return

    def start_streaming(self, display):
        """Start streaming images to the provided display widget."""
        if self.grabber is not None and not self.is_streaming:
            self.grabber.stream_setup(None, display)  # Start the stream to the display
            self.is_streaming = True
            self.streaming_started.emit()  # Notify that streaming has started
            print("Streaming started.")
            self.start()  # Start the QThread to run in the background
        else:
            print("Streaming is already in progress or no camera selected.")

    def stop_streaming(self):
        """Stop streaming images from the camera."""
        if self.is_streaming:
            self.grabber.stop_streaming()  # Stop the streaming
            self.is_streaming = False
            self.streaming_stopped.emit()  # Notify that streaming has stopped
            print("Streaming stopped.")

    def run(self):
        """Override the run method for the thread's execution."""
        while self.is_streaming:
            try:
                image = None
                # image = self.grabber.grab_image()  # Continuously grab images
                if image is not None:
                    # print("Image captured signal emitting")
                    self.image_captured.emit(image)  # Emit the image
            except ic4.IC4Exception as e:
                print(f"Error grabbing image in thread: {e}")

        print("Camera interaction thread has stopped.")

    def capture_single_image(self, save_image=False):
        """
        Capture a single image and optionally save it.

        Args:
            save_image (bool): If True, saves the image to the specified directory

        Returns:
            str or None: Full path to the saved image if saved, else None
        """
        try:
            if self.grabber is None:
                print("Error: No camera selected. Try to select then open now")
                self.open_device()

            print("Attempting to capture image...")

            # Ensure the device is opened
            if not self.grabber.is_device_open:
                self.grabber.device_open()  # Open device if not already opened

            # Check if the device is valid
            if not self.grabber.is_device_valid:
                print("Error: Device is not valid. Reinitializing...")
                self.open_device()
                self.sink = ic4.SnapSink()
                self.grabber.stream_setup(self.sink)
                print("Reinitialized the sink and streaming setup.")
                return None

            # Setup streaming if not already streaming
            if not self.grabber.is_streaming:
                self.sink = ic4.SnapSink()
                self.grabber.stream_setup(self.sink)
            else:
                print("We are streaming video")
                # self.sink = ic4.SnapSink()
                # self.grabber.stream_setup(self.sink)
                # return

            # Start acquisition if not already active
            # if not self.grabber.is_acquisition_active:
            #    self.grabber.acquisition_start()  # Start acquisition

            # Capture the image
            print("Try to snap single image now")
            buffer = self.sink.snap_single(1000)  # 1000ms timeout
            print("Image captured successfully")

            # Save the image if requested
            if save_image:
                # Create directory if it doesn't exist
                save_dir = r"C:\BoardDefectChecker\images\raw-images"
                os.makedirs(save_dir, exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"raw_image_{timestamp}.bmp"
                filepath = os.path.join(save_dir, filename)

                # Save the image in BMP format
                buffer.save_as_bmp(filepath)
                print(f"Image saved to: {filepath}")

                # Return the full path to the saved image
                return filepath

            # Stop the stream after capture
            self.grabber.stream_stop()

            # Return None if not saving
            return None

        except Exception as e:
            print(f"Error capturing image: {e}")
            return None

    def capture_multiple_images(self, save_image=False):
        pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IC Imaging Control 4 Python Library - Camera Interaction")
        self.resize(1024, 768)

        # Create the camera interaction object
        self.camera_interaction = CameraInteraction()

        # Select a camera device
        self.camera_interaction.select_device()

        # Create a display widget and set it as the central widget
        self.display_widget = ic4.pyside6.DisplayWidget()
        self.setCentralWidget(self.display_widget)

        # Start streaming to the display widget after device selection
        # self.camera_interaction.start_streaming(self.display_widget.as_display())

        # Connect signals to slots if needed
        self.camera_interaction.image_captured.connect(self.handle_image_captured)

    def handle_image_captured(self, image):
        # Handle the captured image (e.g., display it or process it)
        print("Handling captured image...")


class SnapSinkWorker(QThread):
    image_captured = Signal(object)
    devices_available = Signal(list)

    def __init__(self, grabber=None, parent=None):  # Optional grabber as in current code
        super().__init__(parent)
        self.grabber = grabber
        self.running = False
        self.snap_sink = None
        self.selected_device = None

    def enumerate_devices(self):
        """Enumerate devices and emit their display text without IP checking."""
        try:
            device_list = ic4.DeviceEnum.devices()
            if not device_list:
                print("No devices found during enumeration!")
                self.devices_available.emit([])
                return

            device_info_list = []
            for i, dev in enumerate(device_list):
                display_text = f"[{i}] {dev.model_name} ({dev.serial}) [{dev.interface.display_name}]"
                device_info_list.append((display_text, dev))
                print(f"Found device: {display_text}")

            self.devices_available.emit(device_info_list)
        except Exception as e:
            print(f"Error enumerating devices: {e}")
            self.devices_available.emit([])

    def open_device(self, device_index=None):
        device_list = ic4.DeviceEnum.devices()
        if not device_list:
            print("No devices found!")
            return

        # Use provided index or default to 0
        if device_index is not None and 0 <= device_index < len(device_list):
            self.selected_device = device_list[device_index]
        else:
            print("Invalid or no device index provided, using first device as fallback.")
            self.selected_device = device_list[0] if device_list else None

        if not self.selected_device:
            print("No valid device selected!")
            return

        dev_info = self.selected_device
        print(
            f"Opening device: {dev_info.model_name} (SN: {dev_info.serial}) [Interface: {dev_info.interface.display_name}]")
        print("Initializing camera connection...")
        time.sleep(2)  # Keep the delay from original code

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.grabber:
                    try:
                        self.grabber.device_close()
                        print("Closed previous grabber instance.")
                    except ic4.IC4Exception as e:
                        print(f"Error closing previous grabber: {e}")
                self.grabber = ic4.Grabber(dev_info)
                # No explicit device_open call needed here; Grabber constructor handles it
                print(f"Device opened successfully on attempt {attempt + 1}")
                break
            except ic4.IC4Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2)
                else:
                    print(f"All {max_retries} attempts failed to open device: {e}")
                    raise e

    def pre_process(self):
        if self.grabber is None or self.selected_device is None:
            print("Error: No camera selected or grabber not initialized. Opening device now.")
            self.open_device()

        if not self.grabber.is_device_valid:
            print("Error: Device is not valid. Reinitializing...")
            self.open_device()

        self.snap_sink = ic4.SnapSink()
        try:
            self.grabber.stream_setup(self.snap_sink)
            print("Stream setup completed successfully.")
        except ic4.IC4Exception as e:
            print(f"Error setting up stream: {e}")

    def capture_single_image(self, save_image=False):
        """
        Capture a single image and optionally save it, without stopping the continuous stream.

        Args:
            save_image (bool): If True, saves the image to the specified directory.

        Returns:
            str or None: Full path to the saved image if saved, else None.
        """
        global timeout_ms
        try:
            if self.grabber is None:
                print("Error: No camera selected. Opening device now.")
                self.open_device()

            if not self.grabber.is_device_valid:
                print("Error: Device is not valid. Reinitializing...")
                self.open_device()
                if not self.grabber.is_device_valid:
                    print("Device remains invalid after reinitialization.")
                    return None

            if self.snap_sink is None:
                self.pre_process()

            print("Attempting to capture a single image...")
            # Increase timeout (e.g., to 2000ms or 3000ms)
            timeout_ms = 3000
            buffer = self.snap_sink.snap_single(timeout_ms)
            print("Image captured successfully.")

            if save_image:
                save_dir = r"C:\BoardDefectChecker\images\raw-images"
                os.makedirs(save_dir, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"raw_image_{timestamp}.bmp"
                filepath = os.path.join(save_dir, filename)

                buffer.save_as_bmp(filepath)
                print(f"Image saved to: {filepath}")
                return filepath

            return None
        except Exception as e:
            # Include timeout value in error message if it fails
            print(f"Error capturing image (timeout={timeout_ms}ms): {e}")
            return None

    def run(self):
        self.running = True
        while self.running:
            try:
                if self.snap_sink is None or not self.grabber.is_device_valid:
                    print("Snap sink or device invalid, reinitializing...")
                    self.pre_process()
                frame = self.snap_sink.snap_single(1000)
                if frame is not None:
                    np_frame = frame.numpy_wrap()
                    rgb_frame = cv2.cvtColor(np_frame, cv2.COLOR_BGR2RGB)
                    self.image_captured.emit(rgb_frame)
            except ic4.IC4Exception as e:
                print("Error capturing image via SnapSink:", e)
            self.msleep(30)

    def stop(self):
        print("Snap sink worker is stopping")
        self.running = False
        if self.grabber:
            try:
                self.grabber.stream_stop()
                print("Stream stopped.")
            except ic4.IC4Exception as e:
                print(f"Error stopping stream: {e}")
            try:
                self.grabber.device_close()
                print("Device closed.")
            except ic4.IC4Exception as e:
                print(f"Error closing device: {e}")
            self.grabber = None
        self.snap_sink = None
        self.wait()
        print("Snap sink worker stopped successfully.")


# Example usage:
if __name__ == "__main__":
    from sys import argv

    app = QApplication(argv)
    app.setStyle("fusion")

    with ic4.Library.init_context():
        wnd = MainWindow()
        wnd.show()
        app.exec()
