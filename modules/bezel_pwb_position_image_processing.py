import os
import cv2
from datetime import datetime
from modules.ai_models import BezelPWBPositionSegmenter
import traceback  # Keep if used elsewhere for error printing


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

    def process_image(self, bezel_pwb_ai_model, unused_model, left_input_path, right_input_path, is_pwb_check_enabled,
                      **kwargs):  # Use kwargs to potentially catch unused params if needed
        """
        Process left and right images using BezelPWBPositionSegmenter.
        Passes relevant parameters to the AI model's process_image method.
        Returns results including the main visualization and the annotated PWB image.

        Args:
            bezel_pwb_ai_model: Instance of BezelPWBPositionSegmenter.
            unused_model: Placeholder for unused model.
            left_input_path: Path to the left input image.
            right_input_path: Path to the right input image.
            **kwargs: Allows passing optional parameters compatible with BezelPWBPositionSegmenter.process_image

        Returns:
            Tuple: (left_mask_img, right_mask_img, left_annotated_pwb_img, right_annotated_pwb_img,
                    left_result, right_result, left_total_time, right_total_time, final_result, defect_reason)
                   The PWB images will be None if the check was disabled or failed.
        """
        left_image = cv2.imread(left_input_path)
        right_image = cv2.imread(right_input_path)
        if left_image is None or right_image is None:
            print("[Error][Processor] Failed to load one or both input images.")
            # Return None for images, NG status, 0 time, and reason
            return None, None, None, None, "NG", "NG", 0.0, 0.0, "NG", "Image load failed"

        # --- Binarization and Raw Image Display (Optional) ---
        left_binary = self.binarize_image(left_image)
        right_binary = self.binarize_image(right_image)
        # ... (optional display logic for raw/binary remains the same) ...
        if self.is_images_shown:
            # Display raw images
            window_title_l_raw = "Raw Left Image - Press SPACE to close"
            cv2.imshow(window_title_l_raw, cv2.resize(left_image, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title_l_raw)

            window_title_r_raw = "Raw Right Image - Press SPACE to close"
            cv2.imshow(window_title_r_raw, cv2.resize(right_image, (1280, 800), interpolation=cv2.INTER_AREA))
            cv2.waitKey(0)
            cv2.destroyWindow(window_title_r_raw)

            # Display binary images
            if left_binary is not None:
                window_title_l_bin = "Left Binary Image - Press SPACE to close"
                cv2.imshow(window_title_l_bin, cv2.resize(left_binary, (1280, 800), interpolation=cv2.INTER_AREA))
                cv2.waitKey(0)
                cv2.destroyWindow(window_title_l_bin)
            if right_binary is not None:
                window_title_r_bin = "Right Binary Image - Press SPACE to close"
                cv2.imshow(window_title_r_bin, cv2.resize(right_binary, (1280, 800), interpolation=cv2.INTER_AREA))
                cv2.waitKey(0)
                cv2.destroyWindow(window_title_r_bin)
        # --- End Binarization and Raw Image Display ---

        # --- Prepare parameters for the AI model call ---
        # Extract relevant parameters from kwargs or use defaults
        # These should match the parameters expected by BezelPWBPositionSegmenter.process_image
        ai_params = {
            "max_masks": kwargs.get("max_masks", 100),
            "test_mode": kwargs.get("test_mode", False),  # Typically False in production
            "max_masks_to_show": kwargs.get("max_masks_to_show", 100),  # Match previous logic
            "edge_threshold": kwargs.get("edge_threshold", 50),
            "enable_mask_iou_filter": kwargs.get("enable_mask_iou_filter", False),
            "iou_threshold": kwargs.get("iou_threshold", 0.9),
            "enable_relative_position_check": kwargs.get("enable_relative_position_check", False),
            "relative_position_pairs_to_check": kwargs.get("relative_position_pairs_to_check", ['bezel-stamped_mark']),
            "enable_containment_check": kwargs.get("enable_containment_check", True),
            "containment_reference_type": kwargs.get("containment_reference_type", "bezel"),
            "containment_target_type": kwargs.get("containment_target_type", "stamped_mark"),
            "enable_bbox_nms": kwargs.get("enable_bbox_nms", True),
            "bbox_nms_iou_threshold": kwargs.get("bbox_nms_iou_threshold", 0.1),
            "bbox_nms_target_types": kwargs.get("bbox_nms_target_types", ["bezel", "copper_mark", "stamped_mark"]),
            "sort_by_area": kwargs.get("sort_by_area", True),
            "is_image_shown": self.is_images_shown,  # Use processor's flag
            "draw_param_table": kwargs.get("draw_param_table", False),  # Don't draw table by default
            "draw_distance_lines": kwargs.get("draw_distance_lines", False),
            "draw_relative_positions": kwargs.get("draw_relative_positions", False),
            "draw_min_rect_bbox": kwargs.get("draw_min_rect_bbox", True),
            "enable_pwb_check": is_pwb_check_enabled
        }
        # --- End Parameter Preparation ---

        # --- Process Left Image ---
        try:
            print(f"[Processor] Calling AI model for LEFT image: {left_input_path}")
            # MODIFIED: Unpack 5 return values
            left_mask_img, left_annotated_pwb_img, left_time, left_result, left_reason = bezel_pwb_ai_model.process_image(
                left_image,
                window_title="Left Processed Image - Press SPACE to close",  # Pass window title
                **ai_params  # Pass prepared parameters
            )
            print(f"[Processor] LEFT AI Result: Status={left_result}, Reason={left_reason}, Time={left_time:.3f}s")
        except Exception as e_left:
            print(f"[Error][Processor] Exception during LEFT AI processing: {e_left}")
            traceback.print_exc()
            left_mask_img, left_annotated_pwb_img, left_time, left_result, left_reason = None, None, 0.0, "NG", f"Processing Error: {e_left}"
        # --- End Process Left Image ---

        # --- Process Right Image ---
        try:
            print(f"[Processor] Calling AI model for RIGHT image: {right_input_path}")
            # MODIFIED: Unpack 5 return values
            right_mask_img, right_annotated_pwb_img, right_time, right_result, right_reason = bezel_pwb_ai_model.process_image(
                right_image,
                window_title="Right Processed Image - Press SPACE to close",  # Pass window title
                **ai_params  # Pass prepared parameters
            )
            print(f"[Processor] RIGHT AI Result: Status={right_result}, Reason={right_reason}, Time={right_time:.3f}s")
        except Exception as e_right:
            print(f"[Error][Processor] Exception during RIGHT AI processing: {e_right}")
            traceback.print_exc()
            right_mask_img, right_annotated_pwb_img, right_time, right_result, right_reason = None, None, 0.0, "NG", f"Processing Error: {e_right}"
        # --- End Process Right Image ---

        # --- Save Processed Images (Mask Images Only) ---
        # REMOVED: Saving of table images
        left_mask_path = self.save_processed_image(left_mask_img, "left",
                                                   "_mask") if left_mask_img is not None else None
        right_mask_path = self.save_processed_image(right_mask_img, "right",
                                                    "_mask") if right_mask_img is not None else None
        # Save annotated PWB images separately if needed (optional)
        left_pwb_path = self.save_processed_image(left_annotated_pwb_img, "left",
                                                  "_pwb_annotated") if left_annotated_pwb_img is not None else None
        right_pwb_path = self.save_processed_image(right_annotated_pwb_img, "right",
                                                   "_pwb_annotated") if right_annotated_pwb_img is not None else None
        # --- End Save Processed Images ---

        # --- Determine Final Result ---
        left_total_time = left_time  # Use the computation time directly
        right_total_time = right_time  # Use the computation time directly
        final_result = "OK" if left_result == "OK" and right_result == "OK" else "NG"
        defect_reason = ""
        if final_result == "NG":
            reasons = []
            if left_result != "OK":
                reasons.append(f"Left: {left_reason}")
            if right_result != "OK":
                reasons.append(f"Right: {right_reason}")
            defect_reason = "; ".join(reasons) if reasons else "NG reason unknown"
        # --- End Determine Final Result ---

        # --- MODIFIED RETURN ---
        # Return the 10 values expected by the worker/UI
        return (left_mask_img, right_mask_img, left_annotated_pwb_img, right_annotated_pwb_img,
                left_result, right_result, left_total_time, right_total_time, final_result, defect_reason)
