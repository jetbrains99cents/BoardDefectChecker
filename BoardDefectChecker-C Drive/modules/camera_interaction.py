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
                #self.grabber.device_property_map.set_value(ic4.PropId.WIDTH, 640)
                #self.grabber.device_property_map.set_value(ic4.PropId.HEIGHT, 480)
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
                print("Image captured signal emitting")
                image = self.grabber.grab_image()  # Continuously grab images
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

            # Start acquisition if not already active
            # if not self.grabber.is_acquisition_active:
            #    self.grabber.acquisition_start()  # Start acquisition

            # Capture the image
            buffer = self.sink.snap_single(1000)  # 1000ms timeout
            print("Image captured successfully")

            # Save the image if requested
            if save_image:
                # Create directory if it doesn't exist
                save_dir = r"C:\BoardDefectChecker\images\raw-images"
                os.makedirs(save_dir, exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"raw_image_{timestamp}.jpeg"
                filepath = os.path.join(save_dir, filename)

                # Save the image
                buffer.save_as_jpeg(filepath, quality_pct=100)
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
        self.camera_interaction.start_streaming(self.display_widget.as_display())

        # Connect signals to slots if needed
        self.camera_interaction.image_captured.connect(self.handle_image_captured)

    def handle_image_captured(self, image):
        # Handle the captured image (e.g., display it or process it)
        print("Handling captured image...")


# Example usage:
if __name__ == "__main__":
    from sys import argv

    app = QApplication(argv)
    app.setStyle("fusion")

    with ic4.Library.init_context():
        wnd = MainWindow()
        wnd.show()
        app.exec()
