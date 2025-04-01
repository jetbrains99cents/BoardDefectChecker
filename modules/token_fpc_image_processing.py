import os
import cv2
import numpy as np
import datetime
import matplotlib.pyplot as plt

class TokenFPCImageProcessor:
    """
    Specialized image-processing class for the Token FPC tab.
    Moves over the logic for histograms, binarizing images, etc.
    that used to be in 'image_processing.py' but only relates to Token FPC.
    """
    def __init__(self, config_dict=None):
        self.config_dict = config_dict if config_dict else {}
        self.is_images_shown = False  # Tied to checkBox_8, for example

        # Directory paths for saving images will be loaded from config.
        self.raw_images_directory_path = ""
        self.resized_images_directory_path = ""
        self.binary_images_directory_path = ""
        self.edge_images_directory_path = ""
        self.ng_images_directory_path = ""
        self.ok_images_directory_path = ""

    def load_config(self, config_data):
        """Load or parse any token-FPC-specific config info."""
        self.config_dict = config_data
        # Store directory paths from config
        self.raw_images_directory_path = config_data.get("raw-images-directory-path", "")
        self.resized_images_directory_path = config_data.get("resized-images-directory-path", "")
        self.binary_images_directory_path = config_data.get("binary-images-directory-path", "")
        self.edge_images_directory_path = config_data.get("edge-images-directory-path", "")
        self.ok_images_directory_path = config_data.get("ok-images-directory-path", "")
        self.ng_images_directory_path = config_data.get("ng-images-directory-path", "")
        print("[Debug] TokenFPCImageProcessing: config loaded.")

    def save_raw_image(self, raw_image, original_image_path, show_image=False):
        """
        Saves the raw image into a subdirectory whose name is prefixed with "token-fpc-"
        followed by the current date (dd-mm-yyyy). For example: "token-fpc-23-03-2025".
        Returns the full path of the saved image.
        """
        # Get current date string in dd-mm-yyyy format
        date_str = datetime.datetime.now().strftime("%d-%m-%Y")
        # Build the subdirectory name with prefix
        sub_dir = f"token-fpc-{date_str}"
        # Use the raw images base directory from config
        if self.raw_images_directory_path:
            save_dir = os.path.join(self.raw_images_directory_path, sub_dir)
        else:
            # Fallback to current directory if not set
            save_dir = sub_dir
        os.makedirs(save_dir, exist_ok=True)
        # Build new filename: e.g. raw_image_20250313_115731.bmp
        base_name = os.path.basename(original_image_path)
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = os.path.splitext(base_name)[1]
        new_filename = f"raw_image_{time_str}{ext}"
        saved_path = os.path.join(save_dir, new_filename)
        # Save the image
        cv2.imwrite(saved_path, raw_image)
        if show_image:
            cv2.imshow("Saved Raw Image", cv2.resize(raw_image, (640, 480), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
        print(f"[Debug] Raw image saved to: {saved_path}")
        return saved_path

    def apply_blur_filters(self, raw_image_path, show_image=False,
                           apply_median_blur=True, median_kernel_size=7,
                           apply_gaussian_blur=True, gaussian_kernel_size=3, gaussian_sigma=0.5):
        """
        Applies median and/or Gaussian blur filters to an image.
        Returns the blurred image and the path where it is saved in the resized images directory.
        """
        if not os.path.exists(raw_image_path):
            print(f"[Error] Image not found: {raw_image_path}")
            return None, None

        image = cv2.imread(raw_image_path)
        if image is None:
            print("[Error] Failed to load image for blur filtering.")
            return None, None

        blurred_image = image.copy()

        if apply_median_blur:
            ksize = median_kernel_size if median_kernel_size % 2 == 1 else median_kernel_size + 1
            blurred_image = cv2.medianBlur(blurred_image, ksize)
            if show_image or self.is_images_shown:
                cv2.imshow("TokenFPC Median Blur", cv2.resize(blurred_image, (640, 480), interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

        if apply_gaussian_blur:
            ksize = gaussian_kernel_size if gaussian_kernel_size % 2 == 1 else gaussian_kernel_size + 1
            blurred_image = cv2.GaussianBlur(blurred_image, (ksize, ksize), gaussian_sigma)
            if show_image or self.is_images_shown:
                cv2.imshow("TokenFPC Gaussian Blur", cv2.resize(blurred_image, (640, 480), interpolation=cv2.INTER_AREA))
                cv2.waitKey(1)

        # Save the blurred image into the resized images directory (with dated subdir)
        if self.resized_images_directory_path:
            date_str = datetime.datetime.now().strftime("%d-%m-%Y")
            sub_dir = f"token-fpc-{date_str}"
            save_dir = os.path.join(self.resized_images_directory_path, sub_dir)
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.basename(raw_image_path)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"blurred_{time_str}{os.path.splitext(base_name)[1]}"
            out_path = os.path.join(save_dir, new_filename)
            cv2.imwrite(out_path, blurred_image)
            print(f"[Debug] TokenFPC blur filters result saved to: {out_path}")
        else:
            out_path = None

        return blurred_image, out_path

    def convert_to_binary_single_threshold(self, input_image, threshold=150, is_invert=False, show_image=False):
        """
        Converts an image to binary using a single threshold.
        Returns the binary image and (optionally) the saved path.
        """
        if input_image is None:
            print("[Error] Input image is None in convert_to_binary_single_threshold.")
            return None, None

        # Convert to grayscale if necessary
        if len(input_image.shape) == 3:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = input_image

        thresh_type = cv2.THRESH_BINARY_INV if is_invert else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, threshold, 255, thresh_type)

        if show_image or self.is_images_shown:
            cv2.imshow("TokenFPC Binary", cv2.resize(binary, (640, 480), interpolation=cv2.INTER_AREA))
            cv2.waitKey(1)

        # Save the binary image into the binary images directory (with dated subdir)
        if self.binary_images_directory_path:
            date_str = datetime.datetime.now().strftime("%d-%m-%Y")
            sub_dir = f"token-fpc-{date_str}"
            save_dir = os.path.join(self.binary_images_directory_path, sub_dir)
            os.makedirs(save_dir, exist_ok=True)
            time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"binary_{time_str}.png"
            out_path = os.path.join(save_dir, filename)
            cv2.imwrite(out_path, binary)
            print(f"[Debug] TokenFPC binary image saved to: {out_path}")
        else:
            out_path = None

        return binary, out_path

    def analyze_histogram(self, image, show_plot=False):
        """
        Analyzes the histogram of the input image.
        Optionally displays a matplotlib plot.
        Returns a dictionary with histogram statistics.
        """
        if image is None:
            return

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        mean_val = np.mean(gray)

        if show_plot or self.is_images_shown:
            plt.figure(figsize=(10, 6))
            plt.plot(hist, color='blue')
            plt.title("TokenFPC Histogram")
            plt.xlabel("Pixel Intensity (0-255)")
            plt.ylabel("Frequency")
            plt.show()
            cv2.waitKey(1)

        print(f"[Debug] TokenFPC histogram: mean={mean_val:.1f}")
        return {"mean": mean_val, "hist": hist}

    def is_image_full_color(self, image, color=(255, 255, 255)):
        """
        Checks if the image is entirely the specified RGB color.
        """
        if image is None:
            return False
        if len(image.shape) != 3 or image.shape[2] != 3:
            print("[Error] The image must be a 3-channel RGB image.")
            return False

        mask = (image == color)
        return np.all(mask)

    # You can add additional specialized methods here as needed.
