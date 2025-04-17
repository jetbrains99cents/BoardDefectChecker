import os
import cv2
from datetime import datetime
from modules.ai_models import BezelPWBPositionSegmenter


class BezelPWBPositionImageProcessor:
    def __init__(self, config_dict=None):
        self.config_dict = config_dict if config_dict else {}
        self.is_images_shown = False
        self.raw_images_directory_path = r"C:\BoardDefectChecker\images\raw-images"
        self.processed_images_directory_path = r"C:\BoardDefectChecker\ai-outputs"
        os.makedirs(self.raw_images_directory_path, exist_ok=True)
        os.makedirs(self.processed_images_directory_path, exist_ok=True)
        print("[BezelPWBPositionImageProcessor] Initialized.")

    def load_config(self, config_data=None):
        if config_data:
            self.config_dict = config_data
            self.raw_images_directory_path = config_data.get("raw-images-directory-path",
                                                             self.raw_images_directory_path)
            self.processed_images_directory_path = config_data.get("processed-images-directory-path",
                                                                   self.processed_images_directory_path)
            print("[Debug] BezelPWBPositionImageProcessor: Config loaded from data.")
            return config_data
        else:
            print("[Debug] BezelPWBPositionImageProcessor: No config data provided, using defaults.")
            return None

    def save_raw_image(self, raw_image, camera_id, original_image_path, show_image=False):
        if raw_image is None:
            print("[Error][Processor] Cannot save None image.")
            return None
        date_str = datetime.now().strftime("%d-%m-%Y")
        sub_dir = f"bezel-pwb-position-{date_str}"
        base_save_dir = self.raw_images_directory_path
        save_dir = os.path.join(base_save_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(original_image_path)[1] if original_image_path else ".png"
        new_filename = f"raw_{camera_id}_{time_str}{ext}"
        saved_path = os.path.join(save_dir, new_filename)
        try:
            cv2.imwrite(saved_path, raw_image)
            print(f"[Info][Processor] Raw {camera_id} saved: {saved_path}")
        except Exception as e:
            print(f"[Error][Processor] Failed to save raw image to {saved_path}: {e}")
            return None
        if show_image or self.is_images_shown:
            try:
                window_title = f"Raw {camera_id} Image - Press SPACE to close"
                display_img = cv2.resize(raw_image, (640, 480), interpolation=cv2.INTER_AREA)
                cv2.imshow(window_title, display_img)
                cv2.waitKey(0)
                cv2.destroyWindow(window_title)
            except Exception as e:
                print(f"[Error][Processor] Failed to display image {window_title}: {e}")
        return saved_path

    def save_processed_image(self, image, side, suffix=""):
        if image is None:
            print(f"[Error][Processor] Cannot save None processed image for {side} {suffix}")
            return None
        date_str = datetime.now().strftime("%d-%m-%Y")
        subdir = f"bezel-pwb-position-{date_str}"
        save_dir = os.path.join(self.processed_images_directory_path, subdir)
        os.makedirs(save_dir, exist_ok=True)
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{side}_processed_bezel_pwb_position_{time_str}{suffix}.png"
        save_path = os.path.join(save_dir, filename)
        try:
            cv2.imwrite(save_path, image)
            print(f"[Info][Processor] Saved processed image: {save_path}")
            return save_path
        except Exception as e:
            print(f"[Error][Processor] Failed to save processed image to {save_path}: {e}")
            return None

    def binarize_image(self, image):
        if image is None:
            print("[Error][Processor] Cannot binarize None image.")
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def process_image(self, bezel_pwb_ai_model, unused_model, left_input_path, right_input_path,
                      visualize_all_masks=False, label_color=(255, 0, 0)):
        """
        Process left and right images using BezelPWBPositionSegmenter, either in evaluation or test mode.

        Args:
            bezel_pwb_ai_model: Instance of BezelPWBPositionSegmenter.
            unused_model: Placeholder for unused model.
            left_input_path: Path to the left input image.
            right_input_path: Path to the right input image.
            visualize_all_masks: If True, visualize all masks (not used directly here but passed for consistency).
            label_color: Color for visualizations.

        Returns:
            Tuple: (left_vis, right_vis, left_pwb_vis, right_pwb_vis, left_result, right_result, left_total_time, right_total_time, final_result, defect_reason)
                - In evaluation mode: Includes OK/NG results and reasons.
                - In test mode: Results and reasons are None, focusing on visualization.
        """
        left_image = cv2.imread(left_input_path)
        right_image = cv2.imread(right_input_path)
        if left_image is None or right_image is None:
            print("[Error][Processor] Failed to load one or both input images.")
            return None, None, None, None, "NG", "NG", 0.0, 0.0, "NG", "Image load failed"

        left_binary = self.binarize_image(left_image)
        right_binary = self.binarize_image(right_image)
        if left_binary is None or right_binary is None:
            print("[Error][Processor] Binarization failed for one or both images.")
            return None, None, None, None, "NG", "NG", 0.0, 0.0, "NG", "Binarization failed"

        # Only display raw images at 1280x800 when is_images_shown is True
        if self.is_images_shown:
            window_title = "Raw Left Image - Press SPACE to close"
            cv2.imshow(window_title, cv2.resize(left_image, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            window_title = "Raw Right Image - Press SPACE to close"
            cv2.imshow(window_title, cv2.resize(right_image, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)

            window_title = "Left Binary Image - Press SPACE to close"
            cv2.imshow(window_title, cv2.resize(left_binary, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            window_title = "Right Binary Image - Press SPACE to close"
            cv2.imshow(window_title, cv2.resize(right_binary, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)

        start_time = datetime.now()

        # Set max_masks based on is_images_shown: None (all masks) if checked, another one if unchecked
        max_masks = 50 if self.is_images_shown else 50
        test_mode = True
        edge_threshold = 5
        sort_by_area = True

        # Process left image
        left_mask_img, left_table_img, left_time, left_result, left_reason = bezel_pwb_ai_model.process_image(
            left_image,
            max_masks=max_masks,
            label_color=label_color,
            test_mode=test_mode,
            max_masks_to_show=max_masks,
            edge_threshold=edge_threshold,
            sort_by_area=sort_by_area,
            window_title="Left Processed Image (Mask + BB) - Press SPACE to close",
            is_image_shown=self.is_images_shown
        )

        # Process right image
        right_mask_img, right_table_img, right_time, right_result, right_reason = bezel_pwb_ai_model.process_image(
            right_image,
            max_masks=max_masks,
            label_color=label_color,
            test_mode=test_mode,
            max_masks_to_show=max_masks,
            edge_threshold=edge_threshold,
            sort_by_area=sort_by_area,
            window_title="Right Processed Image (Mask + BB) - Press SPACE to close",
            is_image_shown=self.is_images_shown
        )

        # Save both mask and table images
        left_mask_path = self.save_processed_image(left_mask_img, "left", "_mask") if left_mask_img is not None else None
        left_table_path = self.save_processed_image(left_table_img, "left", "_table") if left_table_img is not None else None
        right_mask_path = self.save_processed_image(right_mask_img, "right", "_mask") if right_mask_img is not None else None
        right_table_path = self.save_processed_image(right_table_img, "right", "_table") if right_table_img is not None else None

        left_pwb_vis, right_pwb_vis = None, None
        left_pwb_time, right_pwb_time = 0.0, 0.0
        left_pwb_result, right_pwb_result = "OK", "OK"
        left_pwb_reason, right_pwb_reason = "Skipped", "Skipped"

        left_total_time = left_time
        right_total_time = right_time
        final_result = "OK" if left_result == "OK" and right_result == "OK" else "NG"
        defect_reason = ""
        if final_result == "NG":
            reasons = []
            if left_result != "OK":
                reasons.append(f"Left: {left_reason}")
            if right_result != "OK":
                reasons.append(f"Right: {right_reason}")
            defect_reason = "; ".join(reasons)

        return (left_mask_img, right_mask_img, left_pwb_vis, right_pwb_vis,
                left_result, right_result, left_total_time, right_total_time, final_result, defect_reason)