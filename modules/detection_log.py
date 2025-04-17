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

# Create directories if they don't exist
os.makedirs(OPERATION_LOG_DIR, exist_ok=True)
os.makedirs(DETECTION_LOG_DIR, exist_ok=True)

# Set up logging configuration
DETECTION_LOG_FILE = os.path.join(DETECTION_LOG_DIR, 'detection_log.log')

logging.basicConfig(
    filename=DETECTION_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class DetectionLogWorker(QThread):
    log_result_signal = Signal(str, int, int, int)  # log_message, total_count, ng_count, ok_count

    def __init__(self):
        super().__init__()
        self.staff_id = None  # Initialize staff ID as None
        self.detected_part_count = 0
        self.detected_ng_count = 0
        self.detected_ok_count = 0

        self.detection_log_directory = DETECTION_LOG_DIR
        self.operation_log_directory = OPERATION_LOG_DIR

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

        # Always increment counters
        self.detected_part_count += 1
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

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

        # Always increment counters
        self.detected_part_count += 1
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

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
        debug_output_path = os.path.join(self.operation_log_directory,
                                         f"debug_{part_serial_number}_{detection_result}.bmp")
        try:
            with open(debug_output_path, 'wb') as f:
                f.write(decoded_image_data)
            print(f"Debug: Saved decoded BMP to {debug_output_path}, size: {len(decoded_image_data)} bytes")
            if len(decoded_image_data) == len(image_data):
                print("Debug: Base64 encoding/decoding integrity verified")
            else:
                print(
                    f"Debug: Integrity check failed! Original size: {len(image_data)}, Decoded size: {len(decoded_image_data)}")
            os.remove(debug_output_path)
            print(f"Debug: Deleted decoded BMP file {debug_output_path}")
        except Exception as e:
            logging.error(f"Failed to handle debug BMP file {debug_output_path}: {e}")

        # Emit a signal with all counts
        self.log_result_signal.emit(
            f"Logged: {file_path}",
            self.detected_part_count,
            self.detected_ng_count,
            self.detected_ok_count
        )

    def log_bezel_pwb_detection_result(self, part_serial_number, left_result, right_result, left_image_path,
                                       right_image_path):
        if self.staff_id is None:
            logging.error("Staff ID not set. Cannot log detection result.")
            return

        # Always increment counters
        self.detected_part_count += 1

        # Overall result
        detection_result = "OK" if left_result == "OK" and right_result == "OK" else "NG"
        file_name = f"{self.staff_id}_{part_serial_number}_{detection_result}"
        file_path = os.path.join(self.operation_log_directory, f"{file_name}.csv")
        detection_time = datetime.now().strftime('%Y%m%d%H%M%S')

        # Update NG/OK counts immediately after determining detection_result
        if detection_result.lower() == "ng":
            self.detected_ng_count += 1
        elif detection_result.lower() == "ok":
            self.detected_ok_count += 1

        # Convert left image to Base64
        try:
            with open(left_image_path, 'rb') as f:
                left_image_data = f.read()
            left_base64_image = base64.b64encode(left_image_data).decode('utf-8')
            print(f"Left image Base64 encoded size: {len(left_base64_image)} characters")
        except Exception as e:
            logging.error(f"Failed to read left image file {left_image_path}: {e}")
            left_base64_image = "N/A"

        # Convert right image to Base64
        try:
            with open(right_image_path, 'rb') as f:
                right_image_data = f.read()
            right_base64_image = base64.b64encode(right_image_data).decode('utf-8')
            print(f"Right image Base64 encoded size: {len(right_base64_image)} characters")
        except Exception as e:
            logging.error(f"Failed to read right image file {right_image_path}: {e}")
            right_base64_image = "N/A"

        # Save the left Base64 data to a separate CSV file (raw Base64 string only)
        left_base64_filename = f"{file_name}_left_capture.csv"
        left_base64_save_path = os.path.join(self.operation_log_directory, left_base64_filename)
        try:
            with open(left_base64_save_path, mode='a', encoding='utf-8') as file:
                # Write the raw Base64 string directly, no header, no quotes
                file.write(left_base64_image + '\n')
            print(f"[DetectionLogWorker] Saved left Base64 data to {left_base64_save_path}")
        except Exception as e:
            logging.error(f"Failed to save left Base64 data to {left_base64_save_path}: {e}")
            left_base64_save_path = "N/A"

        # Save the right Base64 data to a separate CSV file (raw Base64 string only)
        right_base64_filename = f"{file_name}_right_capture.csv"
        right_base64_save_path = os.path.join(self.operation_log_directory, right_base64_filename)
        try:
            with open(right_base64_save_path, mode='a', encoding='utf-8') as file:
                # Write the raw Base64 string directly, no header, no quotes
                file.write(right_base64_image + '\n')
            print(f"[DetectionLogWorker] Saved right Base64 data to {right_base64_save_path}")
        except Exception as e:
            logging.error(f"Failed to save right Base64 data to {right_base64_save_path}: {e}")
            right_base64_save_path = "N/A"

        # Log entry for the main CSV file (unchanged)
        log_entry = [
            self.detected_part_count,
            self.staff_id,
            part_serial_number,
            detection_result,
            left_result,
            right_result,
            detection_time,
            left_base64_save_path,  # Store the path to the left Base64 CSV file
            right_base64_save_path  # Store the path to the right Base64 CSV file
        ]

        try:
            with open(file_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file, quoting=csv.QUOTE_ALL)
                # Write header if file is new
                if os.stat(file_path).st_size == 0:
                    writer.writerow(["Index", "Staff ID", "Part Serial Number", "Detection Result",
                                     "Left Result", "Right Result", "Timestamp",
                                     "Left Base64 CSV Path", "Right Base64 CSV Path"])
                writer.writerow(log_entry)
            logging.info(f"Logged Bezel-PWB detection result: {file_path}")
        except Exception as e:
            logging.error(f"Failed to write to CSV file {file_path}: {e}")
            return

        self.log_result_signal.emit(
            f"Logged: {file_path}",
            self.detected_part_count,
            self.detected_ng_count,
            self.detected_ok_count
        )

    def log_result(self, log_message, part_number, staff_id, final_result, defect_reason,
                   left_result, right_result, left_time, right_time, is_counter_turned_on):
        """Log a general detection result with detailed information."""
        if self.staff_id is None:
            logging.error("Staff ID not set. Cannot log result.")
            return

        # Always increment counters
        self.detected_part_count += 1
        if final_result.lower() == "ng":
            self.detected_ng_count += 1
        elif final_result.lower() == "ok":
            self.detected_ok_count += 1

        # Emit the updated counts
        self.log_result_signal.emit(
            log_message, self.detected_part_count, self.detected_ng_count, self.detected_ok_count
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
