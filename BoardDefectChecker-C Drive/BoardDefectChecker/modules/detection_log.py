import logging
import os
import csv
from datetime import datetime
from PySide6.QtCore import QThread, Signal

# Define the hard-coded log directories
OPERATION_LOG_DIR = r'C:\BoardDefectChecker\operation-logs'
DETECTION_LOG_DIR = r'C:\BoardDefectChecker\detection-logs'

# Define the log directory
# LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'operation-logs')
os.makedirs(OPERATION_LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist
os.makedirs(DETECTION_LOG_DIR, exist_ok=True)  # Create directory if it doesn't exist

# Set up logging configuration
DETECTION_LOG_FILE = os.path.join(DETECTION_LOG_DIR, 'detection_log.log')

logging.basicConfig(
    filename=DETECTION_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DetectionLogWorker(QThread):
    log_result_signal = Signal(str)  # Signal to emit when logging is done

    def __init__(self):
        super().__init__()
        self.staff_id = None  # Initialize staff ID as None
        self.detected_part_count = 0
        self.detected_ng_count = 0
        self.detected_ok_count = 0

        self.detection_log_directory = DETECTION_LOG_DIR
        self.operation_log_directory = OPERATION_LOG_DIR
        # Define the path for the detection-logs directory
        #self.detection_log_directory = os.path.join(DETECTION_LOG_DIR, 'detection-logs')
        #os.makedirs(self.detection_log_directory, exist_ok=True)  # Create the directory if it doesn't exist

    def set_staff_id(self, staff_id):
        """Set the staff ID dynamically."""
        self.staff_id = staff_id
        self.reset_counts()

    def get_staff_id(self):
        """Get the current staff ID."""
        return self.staff_id

    def log_detection_result(self, part_serial_number, detection_result):
        """Log the detection result in a separate thread."""
        if self.staff_id is None:
            logging.error("Staff ID not set. Cannot log detection result.")
            return

        self.detected_part_count += 1

        # Determine file name based on the staff ID, part serial number, and detection result
        file_name = f"{self.staff_id}_{part_serial_number}_{detection_result}.csv"
        file_path = os.path.join(self.operation_log_directory, file_name)

        # Prepare the content to log
        detection_time = datetime.now().strftime('%Y%m%d%H%M%S')  # Unified format
        log_entry = [
            self.detected_part_count,  # Index
            self.staff_id,
            part_serial_number,
            detection_result,
            detection_time
        ]

        # Write to CSV file
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(log_entry)

        # Log the action
        logging.info(f"Logged detection result: {log_entry}")

        # Update counts based on the detection result
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

        # Emit a signal indicating the logging is done
        self.log_result_signal.emit(f"Logged: {log_entry}")

    def reset_counts(self):
        """Reset the detected part counts for a new working section."""
        self.detected_part_count = 0
        self.detected_ng_count = 0
        self.detected_ok_count = 0
        logging.info("Counts have been reset for a new working section.")

    def run(self):
        """Run the thread's activity here if needed."""
        pass

# Example usage
if __name__ == "__main__":
    worker = DetectionLogWorker()
    worker.set_staff_id("John Doe")  # Set staff ID for demonstration

    # Simulate logging some detection results
    worker.start()  # Start the thread
    worker.log_detection_result(part_serial_number="123456", detection_result="OK")
    worker.log_detection_result(part_serial_number="123457", detection_result="NG")

    # Reset counts for a new working section
    worker.reset_counts()