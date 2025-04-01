import logging
import os
import csv
import base64
import zlib
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
    log_result_signal = Signal(str, int, int, int)  # filepath, total_count, ng_count, ok_count

    def __init__(self):
        super().__init__()
        self.staff_id = None  # Initialize staff ID as None
        self.detected_part_count = 0
        self.detected_ng_count = 0
        self.detected_ok_count = 0

        self.detection_log_directory = DETECTION_LOG_DIR
        self.operation_log_directory = OPERATION_LOG_DIR
        # Define the path for the detection-logs directory
        # self.detection_log_directory = os.path.join(DETECTION_LOG_DIR, 'detection-logs')
        # os.makedirs(self.detection_log_directory, exist_ok=True)  # Create the directory if it doesn't exist

    def set_staff_id(self, staff_id):
        """Set the staff ID dynamically."""
        self.staff_id = staff_id
        self.reset_counts()

    def get_staff_id(self):
        """Get the current staff ID."""
        return self.staff_id

    def set_log_directory(self, directory):
        """Set the operation log directory dynamically."""
        self.operation_log_directory = directory
        os.makedirs(self.operation_log_directory, exist_ok=True)  # Ensure the directory exists
        logging.info(f"[DetectionLogWorker] Operation log directory updated to: {self.operation_log_directory}")

    def log_detection_result(self, part_serial_number, detection_result, jpeg_image_path, compress=False):
        """Log the detection result along with a JPEG image (optionally compressed and Base64-encoded) into a CSV file."""
        if self.staff_id is None:
            logging.error("Staff ID not set. Cannot log detection result.")
            return

        self.detected_part_count += 1

        # Determine file name based on staff ID, part serial number, and detection result
        file_name = f"{self.staff_id}_{part_serial_number}_{detection_result}.csv"
        file_path = os.path.join(self.operation_log_directory, file_name)

        # Get current detection time in unified format
        detection_time = datetime.now().strftime('%Y%m%d%H%M%S')

        # Open the JPEG image file in binary mode and read its contents
        try:
            with open(jpeg_image_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            logging.error(f"Failed to read image file {jpeg_image_path}: {e}")
            return

        print(f"Original image size: {len(image_data)} bytes")

        # Optionally compress the image data using zlib
        if compress:
            compressed_data = zlib.compress(image_data, level=9)
            print(f"Compressed image size: {len(compressed_data)} bytes")
            data_to_encode = compressed_data
        else:
            data_to_encode = image_data

        # Convert the (compressed or raw) data to a Base64-encoded string
        base64_image = base64.b64encode(data_to_encode).decode('utf-8')
        print(f"Base64 encoded size: {len(base64_image)} characters")

        # Prepare the CSV log entry, appending the Base64 image after the detection time.
        log_entry = [
            self.detected_part_count,  # Index
            self.staff_id,
            part_serial_number,
            detection_result,
            detection_time,
            base64_image
        ]

        # Append the log entry to the CSV file.
        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(log_entry)
            logging.info(f"Logged detection result: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write to CSV file {file_path}: {e}")
            return

        # Update counts based on the detection result.
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

        # Emit a signal indicating the logging is complete with all counts
        self.log_result_signal.emit(
            f"Logged: {file_path}",
            self.detected_part_count,
            self.detected_ng_count,
            self.detected_ok_count
        )

    def log_token_fpc_detection_result(self, part_serial_number, detection_result, bmp_image_path):
        if self.staff_id is None:
            logging.error("Staff ID not set. Cannot log detection result.")
            return

        self.detected_part_count += 1

        file_name = f"{self.staff_id}_{part_serial_number}_{detection_result}.csv"
        file_path = os.path.join(self.operation_log_directory, file_name)
        detection_time = datetime.now().strftime('%Y%m%d%H%M%S')

        try:
            with open(bmp_image_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            logging.error(f"Failed to read BMP image file {bmp_image_path}: {e}")
            return

        print(f"Original BMP image size: {len(image_data)} bytes")
        base64_image = base64.b64encode(image_data).decode('utf-8')
        print(f"Base64 encoded size: {len(base64_image)} characters")

        log_entry = [
            self.detected_part_count,
            self.staff_id,
            part_serial_number,
            detection_result,
            detection_time,
            base64_image
        ]

        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL)
                writer.writerow(log_entry)
            logging.info(f"Logged Token FPC detection result: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write to CSV file {file_path}: {e}")
            return

        # Integrity verification
        decoded_image_data = base64.b64decode(base64_image)
        debug_output_path = os.path.join(self.operation_log_directory, f"debug_{part_serial_number}_{detection_result}.bmp")
        try:
            with open(debug_output_path, 'wb') as f:
                f.write(decoded_image_data)
            print(f"Debug: Saved decoded BMP to {debug_output_path}, size: {len(decoded_image_data)} bytes")
            if len(decoded_image_data) == len(image_data):
                print("Debug: Base64 encoding/decoding integrity verified")
            else:
                print(f"Debug: Integrity check failed! Original size: {len(image_data)}, Decoded size: {len(decoded_image_data)}")
            os.remove(debug_output_path)
            print(f"Debug: Deleted decoded BMP file {debug_output_path}")
        except Exception as e:
            logging.error(f"Failed to handle debug BMP file {debug_output_path}: {e}")

        # Update counts and emit signal with all counts
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

        self.log_result_signal.emit(
            f"Logged: {file_path}",
            self.detected_part_count,
            self.detected_ng_count,
            self.detected_ok_count
        )

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
