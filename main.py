import sys
import os
import importlib.util
from PySide6.QtWidgets import QApplication
from main_window import MainWindow
from modules import config
from modules.camera_connection import init_camera_connection_handler
from modules.detection_log import DetectionLogWorker  # Import your DetectionLogWorker

# Print system path and config parameters
print(sys.path)
print("Config parameter list: " + str(config.config_dict))

def load_config_file(file_path):
    """
    Dynamically loads a Python-based configuration file.
    Checks if the file exists and imports it as a module.
    """
    if not os.path.exists(file_path):
        print(f"Error: Config file '{file_path}' does not exist.")
        return None

    try:
        # Dynamically load the Python file as a module
        spec = importlib.util.spec_from_file_location("dynamic_config", file_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)

        # Check if `config_dict` exists in the loaded module
        if hasattr(config_module, "config_dict"):
            config_data = config_module.config_dict
            print(f"Config file '{file_path}' loaded successfully!")
            print("Config content:", config_data)
            return config_data
        else:
            print(f"Error: Config file '{file_path}' does not contain 'config_dict'.")
            return None
    except Exception as e:
        print(f"Error: An unexpected error occurred while loading the config file: {e}")
        return None

def main():
    app = QApplication([])

    # Create the detection log worker without a staff ID
    detection_log_worker = DetectionLogWorker()

    # Create the main window and pass the worker
    window = MainWindow(detection_log_worker)

    # Load the config file
    config_file_path = r"C:\BoardDefectChecker\modules\config.py"  # Replace it with your actual config file path
    config_data = load_config_file(config_file_path)

    # Update the UI with the loaded config data
    if config_data:
        window.load_config(config_data)

    # Initialize camera connection handler
    camera_worker = init_camera_connection_handler()
    camera_worker.connection_status_changed.connect(window.update_connection_status)

    # Connect the signal for logging results
    detection_log_worker.log_result_signal.connect(window.update_log_status)

    # Start the detection log worker
    detection_log_worker.start()

    # Start the application event loop
    #window.show()
    window.showMaximized()

    app.exec()

if __name__ == "__main__":
    main()