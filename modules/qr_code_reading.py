import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex
import os
import re
from datetime import datetime
from typing import Optional

# --- OpenCV is assumed to be available ---
# print("[QR Module] Using OpenCV QRCodeDetector.") # Moved inside classes

class QRCodeReader:
    """
    A class to load an image and decode a QR code using OpenCV from a static file.
    """

    def __init__(self, image_path: str, verbose: bool = True): # Added verbose flag
        """
        Initialize the QRCodeReader with the path to the image file.

        :param image_path: Path to the image file containing the QR code.
        :param verbose: If True, print initialization messages.
        """
        self.image_path = image_path
        self.verbose = verbose # Store verbose flag
        # Initialize OpenCV detector
        self.detector = cv2.QRCodeDetector()
        if self.verbose:
            print("[QRCodeReader] Initialized with OpenCV QRCodeDetector.")


    def load_image(self):
        """
        Loads the image from the provided file path.

        :return: The loaded image (as NumPy array).
        :raises ValueError: If the image cannot be loaded.
        """
        if not os.path.exists(self.image_path):
             raise ValueError(f"Image file not found: {self.image_path}")
        image = cv2.imread(self.image_path)
        if image is None:
            raise ValueError(f"Unable to open image using OpenCV: {self.image_path}")
        return image

    def decode_qr(self):
        """
        Detects and decodes the QR code from the image file using OpenCV.

        :return: A tuple (data, bbox, rectified_image) where:
                 - data: the decoded text (empty string if none found),
                 - bbox: bounding box points,
                 - rectified_image: binarized QR image.
        """
        image = self.load_image()
        # Use OpenCV detector
        data, bbox, rectified_image = self.detector.detectAndDecode(image)
        # Keep runtime prints for now, or add verbose check here too if needed
        # if data:
        #      print(f"[QRCodeReader OpenCV] Decoded: {data}")
        # else:
        #      print("[QRCodeReader OpenCV] No QR code found.")
        return data, bbox, rectified_image

    def get_qr_data(self):
        """
        Retrieves the decoded QR code data from the image file.

        :return: The decoded QR code text.
        :raises ValueError: If no QR code is detected.
        """
        data, _, _ = self.decode_qr()
        if data:
            return data
        else:
            raise ValueError("No QR code detected in the image.")


# --- QR Scanner Worker Thread using OpenCV ---
class QRScannerWorker(QThread):
    """
    Worker thread to continuously scan video frames for QR codes using OpenCV QRCodeDetector.
    """
    qr_code_found = Signal(str, object, str)

    def __init__(self, parent=None, verbose: bool = True): # Added verbose flag
        super().__init__(parent)
        self.running = False
        self.current_frame_left = None
        self.current_frame_right = None
        self._lock = QMutex()
        self.verbose = verbose # Store verbose flag
        # Initialize OpenCV detector for the worker
        self.detector = cv2.QRCodeDetector()
        if self.verbose:
            print("[QR Worker] Initialized with OpenCV QRCodeDetector.")


    def set_frame(self, frame: np.ndarray, source: str):
        """Update the frame to be processed (thread-safe)."""
        self._lock.lock()
        try:
            # Store the frame directly (OpenCV works with numpy array)
            if source == 'left':
                # Keep frame in RGB format as received
                self.current_frame_left = frame
            elif source == 'right':
                self.current_frame_right = frame
        finally:
            self._lock.unlock()

    def run(self):
        """Continuously process the latest frame for QR codes using OpenCV."""
        if self.verbose: print("[QR Worker] Started (Using OpenCV).") # Control start message
        self.running = True
        last_processed_frame_left = None
        last_processed_frame_right = None
        frame_process_count = 0 # Debug counter

        while self.running:
            current_left = None
            current_right = None
            processed_this_cycle = False # Flag to check if any work was done

            # Safely get current frames
            self._lock.lock()
            try:
                current_left = self.current_frame_left
                current_right = self.current_frame_right
            finally:
                self._lock.unlock()

            # --- Process Left Frame if New ---
            if current_left is not None and current_left is not last_processed_frame_left:
                frame_to_process = current_left.copy()
                source = 'left'
                last_processed_frame_left = current_left # Update last processed immediately
                processed_this_cycle = True
                frame_process_count += 1
                # Control runtime prints with verbose flag
                if self.verbose and frame_process_count % 10 == 1:
                     print(f"[QR Worker] Processing frame #{frame_process_count} from {source}...")

                try:
                    if frame_to_process.size > 0 and frame_to_process.shape[0] > 0 and frame_to_process.shape[1] > 0:
                        if len(frame_to_process.shape) == 3 and frame_to_process.shape[2] == 3:
                            cv_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2GRAY)
                        else:
                            cv_frame = frame_to_process

                        data, bbox, _ = self.detector.detectAndDecode(cv_frame)

                        if data:
                            if self.verbose: print(f"[QR Worker] SUCCESS: QR Code found by {source} camera (OpenCV): {data}")
                            self.qr_code_found.emit(data, frame_to_process, source)
                            # Keep running
                        elif bbox is not None and self.verbose and frame_process_count % 20 == 1:
                            print(f"[QR Worker] INFO: Potential QR code detected by {source} (bbox found), but failed to decode.")
                    # else:
                    #      if self.verbose: print("[QR Worker] Skipping empty left frame.")

                except cv2.error as e:
                     if "Assertion failed" not in str(e) and self.verbose: print(f"[QR Worker] OpenCV error (Left): {e}")
                except Exception as e:
                    if self.verbose: print(f"[QR Worker] Error processing left frame: {e}")

            # --- Process Right Frame if New (Independent Check) ---
            if current_right is not None and current_right is not last_processed_frame_right:
                frame_to_process = current_right.copy()
                source = 'right'
                last_processed_frame_right = current_right # Update last processed immediately
                processed_this_cycle = True
                if self.verbose: print(f"[QR Worker] Processing frame from {source}...") # Log every attempt for right if verbose

                try:
                    if frame_to_process.size > 0 and frame_to_process.shape[0] > 0 and frame_to_process.shape[1] > 0:
                        if len(frame_to_process.shape) == 3 and frame_to_process.shape[2] == 3:
                            cv_frame = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2GRAY)
                        else:
                            cv_frame = frame_to_process

                        data, bbox, _ = self.detector.detectAndDecode(cv_frame)

                        if data:
                            if self.verbose: print(f"[QR Worker] SUCCESS: QR Code found by {source} camera (OpenCV): {data}")
                            self.qr_code_found.emit(data, frame_to_process, source)
                            # Keep running
                        elif bbox is not None and self.verbose: # Log if bbox found but no data
                            print(f"[QR Worker] INFO: Potential QR code detected by {source} (bbox found), but failed to decode.")
                    # else:
                    #      if self.verbose: print("[QR Worker] Skipping empty right frame.")

                except cv2.error as e:
                     if "Assertion failed" not in str(e) and self.verbose: print(f"[QR Worker] OpenCV error (Right): {e}")
                except Exception as e:
                    if self.verbose: print(f"[QR Worker] Error processing right frame: {e}")

            # Sleep only if no work was done in this cycle to avoid unnecessary delay
            if not processed_this_cycle:
                self.msleep(50) # Shorter sleep if idle
            else:
                self.msleep(10) # Very short sleep if work was done, allows faster processing

        if self.verbose: print("[QR Worker] Stopped.") # Control stop message


    def stop(self):
        """Stop the worker thread."""
        if self.verbose: print("[QR Worker] Stopping...")
        self.running = False
        self.wait(500)
        if self.isRunning():
            if self.verbose: print("[QR Worker] Warning: Thread did not stop gracefully, terminating.")
            self.terminate()
        if self.verbose: print("[QR Worker] Stop sequence complete.")


# --- Standalone function to save QR code image (UPDATED SIGNATURE) ---
def save_qr_code_image(frame: np.ndarray, source: str, qr_data: str, base_save_dir: str, folder_prefix: str, verbose: bool = True) -> Optional[str]: # Added verbose
    """
    Saves the frame containing the QR code to a specific directory, using a dynamic folder prefix.

    Args:
        frame (np.ndarray): The image frame (expected in RGB format from worker signal).
        source (str): The source identifier (e.g., 'left', 'right').
        qr_data (str): The decoded QR code data (used for filename).
        base_save_dir (str): The root directory to save QR images into (e.g., "C:\\...\\qr-codes").
        folder_prefix (str): The prefix for the dated subdirectory (e.g., "bezel-pwb-position").
        verbose (bool): If True, print saving messages.

    Returns:
        Optional[str]: The full path to the saved image, or None on failure.
    """
    if frame is None:
        if verbose: print("[Error][Save QR Func] Cannot save None frame.")
        return None
    if not base_save_dir:
        if verbose: print("[Error][Save QR Func] Base save directory not provided.")
        return None
    if not folder_prefix:
        if verbose: print("[Error][Save QR Func] Folder prefix not provided.")
        return None # Require a prefix

    try:
        # Ensure the base directory exists
        os.makedirs(base_save_dir, exist_ok=True)

        # Create dated subdirectory with dynamic prefix
        date_str = datetime.now().strftime("%d-%m-%Y")
        sub_dir_name = f"{folder_prefix}-{date_str}" # Use the passed prefix
        save_dir = os.path.join(base_save_dir, sub_dir_name)
        os.makedirs(save_dir, exist_ok=True)

        # Construct filename
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Basic sanitization for filename from QR data
        sanitized_qr_data = re.sub(r'[\\/*?:"<>|]', '_', qr_data)[:50] # Limit length
        filename = f"qr_code_{source}_{sanitized_qr_data}_{time_str}.png" # Use png
        save_path = os.path.join(save_dir, filename)

        # Convert RGB frame (from signal) to BGR for cv2.imwrite
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame # Assume compatible if not RGB (e.g., grayscale)

        # Save the image
        success = cv2.imwrite(save_path, bgr_frame)
        if success:
            if verbose: print(f"[Save QR Func] Saved QR code image to: {save_path}")
            return save_path
        else:
            if verbose: print(f"[Error][Save QR Func] cv2.imwrite failed for path: {save_path}")
            return None
    except Exception as e:
        if verbose: print(f"[Error][Save QR Func] Exception during saving: {e}")
        return None


# Example usage (for QRCodeReader):
if __name__ == "__main__":
    # Path to the QR code BMP file.
    image_path = r"D:\Working\BoardDefectChecker\resources\qr_code.bmp"
    # Test verbose flag for QRCodeReader
    reader_verbose = QRCodeReader(image_path, verbose=False)
    reader_quiet = QRCodeReader(image_path, verbose=False)
    try:
        print("\nTesting verbose reader:")
        qr_content_v, _, _ = reader_verbose.decode_qr()
        print(f"Decoded (verbose): {qr_content_v}")

        print("\nTesting quiet reader:")
        qr_content_q, _, _ = reader_quiet.decode_qr()
        print(f"Decoded (quiet): {qr_content_q}")

    except Exception as e:
        print("Error reading QR from file:", e)

    # Note: QRScannerWorker is intended to be used within a Qt application
