import os
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import gc
import sys  # Keep sys if used elsewhere for path manipulation
import traceback  # Keep if used elsewhere for error printing
from typing import List, Dict, Tuple, Any, Optional

# Import segment_anything (ensure library is installed if needed)
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    print("[Warning] Failed to import 'segment_anything'.")


    # Define dummy classes if needed and Pillow unavailable
    class sam_model_registry:
        pass


    class SamAutomaticMaskGenerator:
        pass

# Import FastSAM (ensure library is installed)
try:
    from fastsam import FastSAM
except ImportError:
    print("[FATAL ERROR] Failed to import FastSAM. Ensure the 'fastsam' library is installed.")


    # *** ADDED DUMMY CLASS ***
    class FastSAM:
        pass  # Define dummy if import fails
    # sys.exit(1) # Consider if exit is still appropriate

import cv2
from datetime import datetime
import time
from abc import ABC, abstractmethod

# --- Import project modules using 'from modules.' ---
try:
    from modules.image_processing import ImageProcessor
except ImportError:
    print("[Warning] Could not import ImageProcessor from modules.")


    class ImageProcessor:
        pass  # Define dummy class if not found

try:
    from modules.token_fpc_image_processing import TokenFPCImageProcessor
except ImportError:
    print("[Warning] Could not import TokenFPCImageProcessor from modules.")


    class TokenFPCImageProcessor:
        pass  # Define dummy class if not found

try:
    from modules.rotation_invariant_checking import RotationInvariantAOIChecker  # Make sure this import is correct

    print("[Info] Successfully imported RotationInvariantAOIChecker from modules.")
except ImportError as e:
    print(f"[FATAL ERROR] Failed to import RotationInvariantAOIChecker from modules: {e}")


    # *** ADDED DUMMY CLASS ***
    class RotationInvariantAOIChecker:  # Define dummy if import fails
        def __init__(self, config): pass

        def test_masks(self, *args, **kwargs): return []

        def evaluate(self, *args, **kwargs): return False, "Checker Not Loaded", {}
    # sys.exit(1) # Consider if exit is still appropriate
# --- End project module imports ---


# Import PIL (Pillow)
try:
    from PIL import Image, ImageDraw, ImageFont

    _pillow_available_global = True  # Define flag for conditional use later
except ImportError:
    _pillow_available_global = False
    print("[Warning] Pillow library not found. Text rendering might be basic.")


    # Define dummy classes if Pillow is missing
    class ImageFont:
        @staticmethod
        def truetype(font, size): return None


    class Image:
        pass


    class ImageDraw:
        pass


class BalanceAnalyzer:
    """Analyzes horizontal balance of segmented objects"""

    def analyze_balance(self, mask):
        """
        Analyze balance using multiple metrics
        Returns dict with balance scores and analysis
        """
        height, width = mask.shape
        mask_uint8 = mask.astype(np.uint8)

        # Find centroid
        moments = cv2.moments(mask_uint8)
        if moments['m00'] == 0:
            return None

        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']

        # Mass Distribution Score
        left_mass = cv2.countNonZero(mask_uint8[:, :int(width / 2)])
        right_mass = cv2.countNonZero(mask_uint8[:, int(width / 2):])
        mass_ratio = min(left_mass, right_mass) / max(left_mass, right_mass) if max(left_mass, right_mass) > 0 else 1
        mass_bias = 'left' if left_mass > right_mass else 'right'

        # Center of Mass Offset
        com_offset = abs(center_x - width / 2) / (width / 2)
        lean_direction = 'left' if center_x < width / 2 else 'right'

        # Row-wise Balance Analysis
        row_biases = []
        for y in range(height):
            row = mask_uint8[y, :]
            left_sum = np.sum(row[:int(width / 2)])
            right_sum = np.sum(row[int(width / 2):])
            if left_sum + right_sum > 0:
                bias = (left_sum - right_sum) / (left_sum + right_sum)
                row_biases.append(bias)

        row_bias_mean = np.mean(row_biases) if row_biases else 0
        row_bias_std = np.std(row_biases) if row_biases else 0

        # Edge Distribution
        edges = cv2.Canny(mask_uint8, 100, 200)
        left_edges = cv2.countNonZero(edges[:, :int(width / 2)])
        right_edges = cv2.countNonZero(edges[:, int(width / 2):])
        edge_ratio = min(left_edges, right_edges) / max(left_edges, right_edges) if max(left_edges,
                                                                                        right_edges) > 0 else 1

        # Combine scores
        balance_score = (
                0.3 * mass_ratio +
                0.3 * (1 - com_offset) +
                0.2 * (1 - abs(row_bias_mean)) +
                0.2 * edge_ratio
        )

        return {
            'score': balance_score,
            'is_balanced': balance_score > 0.85,
            'mass_ratio': mass_ratio,
            'mass_bias': mass_bias,
            'com_offset': com_offset,
            'lean_direction': lean_direction,
            'row_bias_mean': row_bias_mean,
            'row_bias_std': row_bias_std,
            'edge_ratio': edge_ratio,
            'center': (center_x, center_y)
        }


class ImageSegmenter(ABC):
    """Abstract base class for image segmentation"""

    @abstractmethod
    def __init__(self, model_path="models/"):
        self.device = "cpu"
        self.model_path = model_path

    @abstractmethod
    def _load_model(self):
        """Load the segmentation model"""
        pass

    # @abstractmethod
    # def process_image(self, input_path, output_path, pin_count):
    #    """Process a single image and save the result"""
    #    pass

    # @abstractmethod
    # def visualize_masks(self, image, masks):
    #    """Visualize segmentation masks"""
    #    pass

    # @abstractmethod
    # def draw_bounding_boxes(self, image, masks, output_path):
    #    """Draw bounding boxes with dimensions on the image"""
    #    pass

    def draw_skeletons(self, image, masks, output_path):
        """Draw skeletons of segmented regions"""
        pass

    def filter_max_width_mask(self, masks, image_shape, edge_threshold=5):
        """Filter and return the mask with maximum width away from edges"""
        pass


class SymmetryAnalysisMixin:
    def analyze_symmetry(self, mask, skeleton):
        """
        Analyze symmetry of a mask using skeleton-based centroid
        Returns: dict with symmetry scores and analysis
        """
        height, width = mask.shape

        # Find skeleton points and calculate centroid
        y_indices, x_indices = np.where(skeleton)
        if len(y_indices) == 0:
            return None

        # Calculate centroid from skeleton
        center_y = np.mean(y_indices)
        center_x = np.mean(x_indices)

        # Convert mask to uint8 for OpenCV operations
        mask_uint8 = mask.astype(np.uint8)

        # 1. Area-based symmetry using skeleton centroid
        left_area = cv2.countNonZero(mask_uint8[:, :int(center_x)])
        right_area = cv2.countNonZero(mask_uint8[:, int(center_x):])
        area_score = abs(left_area - right_area) / (left_area + right_area) if (left_area + right_area) > 0 else 1.0

        # 2. Contour-based comparison around skeleton centroid
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            contour = contours[0]
            left_points = []
            right_points = []

            for point in contour[:, 0, :]:
                if point[0] < center_x:
                    # Mirror points around skeleton centroid
                    left_points.append((center_x - point[0], point[1] - center_y))
                else:
                    right_points.append((point[0] - center_x, point[1] - center_y))

            if left_points and right_points:
                # Convert points to contour format
                left_contour = np.array(left_points).reshape((-1, 1, 2)).astype(np.float32)
                right_contour = np.array(right_points).reshape((-1, 1, 2)).astype(np.float32)

                try:
                    contour_score = cv2.matchShapes(left_contour, right_contour, cv2.CONTOURS_MATCH_I2, 0)
                except cv2.error:
                    contour_score = 1.0
            else:
                contour_score = 1.0
        else:
            contour_score = 1.0

        # 3. Pixel distribution analysis relative to skeleton centroid
        distribution_scores = []
        for y in range(height):
            row = mask_uint8[y, :]
            # Calculate distribution around skeleton centroid
            left_sum = np.sum(row[:int(center_x)])
            right_sum = np.sum(row[int(center_x):])

            if left_sum + right_sum > 0:
                score = abs(left_sum - right_sum) / (left_sum + right_sum)
                distribution_scores.append(score)

        distribution_score = np.mean(distribution_scores) if distribution_scores else 1.0

        # 4. Skeleton symmetry score
        left_skeleton = skeleton[:, :int(center_x)]
        right_skeleton = skeleton[:, int(center_x):]
        left_skel_points = np.sum(left_skeleton)
        right_skel_points = np.sum(right_skeleton)
        skeleton_score = abs(left_skel_points - right_skel_points) / (left_skel_points + right_skel_points) if (
                                                                                                                       left_skel_points + right_skel_points) > 0 else 1.0

        # Combine scores with skeleton weight
        final_score = (
                0.3 * area_score +
                0.3 * contour_score +
                0.2 * distribution_score +
                0.2 * skeleton_score  # Added skeleton symmetry weight
        )

        return {
            'score': final_score,
            'center': (center_x, center_y),
            'is_symmetric': final_score < 0.15,
            'area_score': area_score,
            'contour_score': contour_score,
            'distribution_score': distribution_score,
            'skeleton_score': skeleton_score,  # New score
            'skeleton_center': (center_x, center_y)  # Explicitly show skeleton centroid
        }


class BalanceAnalysisMixin:
    def analyze_balance(self, mask):
        """Analyze balance metrics only"""
        height, width = mask.shape
        mask_uint8 = mask.astype(np.uint8)

        # Find centroid
        moments = cv2.moments(mask_uint8)
        if moments['m00'] == 0:
            return None

        center_x = moments['m10'] / moments['m00']
        center_y = moments['m01'] / moments['m00']

        # Mass Distribution Score
        left_mass = cv2.countNonZero(mask_uint8[:, :int(width / 2)])
        right_mass = cv2.countNonZero(mask_uint8[:, int(width / 2):])
        mass_ratio = min(left_mass, right_mass) / max(left_mass, right_mass) if max(left_mass, right_mass) > 0 else 1
        mass_bias = 'left' if left_mass > right_mass else 'right'

        # Center of Mass Offset
        com_offset = abs(center_x - width / 2) / (width / 2)
        lean_direction = 'left' if center_x < width / 2 else 'right'

        # Row-wise Balance Analysis
        row_biases = []
        for y in range(height):
            row = mask_uint8[y, :]
            left_sum = np.sum(row[:int(width / 2)])
            right_sum = np.sum(row[int(width / 2):])
            if left_sum + right_sum > 0:
                bias = (left_sum - right_sum) / (left_sum + right_sum)
                row_biases.append(bias)

        row_bias_mean = np.mean(row_biases) if row_biases else 0
        row_bias_std = np.std(row_biases) if row_biases else 0

        # Edge Distribution
        edges = cv2.Canny(mask_uint8, 100, 200)
        left_edges = cv2.countNonZero(edges[:, :int(width / 2)])
        right_edges = cv2.countNonZero(edges[:, int(width / 2):])
        edge_ratio = min(left_edges, right_edges) / max(left_edges, right_edges) if max(left_edges,
                                                                                        right_edges) > 0 else 1

        # Combine scores
        balance_score = (
                0.3 * mass_ratio +
                0.3 * (1 - com_offset) +
                0.2 * (1 - abs(row_bias_mean)) +
                0.2 * edge_ratio
        )

        return {
            'score': balance_score,
            'is_balanced': balance_score > 0.85,
            'mass_ratio': mass_ratio,
            'mass_bias': mass_bias,
            'com_offset': com_offset,
            'lean_direction': lean_direction,
            'row_bias_mean': row_bias_mean,
            'row_bias_std': row_bias_std,
            'edge_ratio': edge_ratio,
            'center': (center_x, center_y)
        }

    def draw_balance(self, image, mask, balance_info, output_path):
        """Draw balance visualization only"""
        if balance_info is None:
            return

        balance_image = image.copy()
        height, width = image.shape[:2]
        center_x, center_y = balance_info['center']

        # Draw center and COM lines
        cv2.line(balance_image,
                 (int(width / 2), 0),
                 (int(width / 2), height),
                 (0, 255, 255), 1)  # Yellow center line

        cv2.line(balance_image,
                 (int(center_x), 0),
                 (int(center_x), height),
                 (255, 0, 0), 2)  # Red COM line

        # Color sides
        left_mask = mask.copy()
        right_mask = mask.copy()
        left_mask[:, int(width / 2):] = 0
        right_mask[:, :int(width / 2)] = 0

        if balance_info['mass_bias'] == 'left':
            balance_image[left_mask > 0] = [0, 0, 255]  # Red for heavier
            balance_image[right_mask > 0] = [0, 255, 0]  # Green for lighter
        else:
            balance_image[left_mask > 0] = [0, 255, 0]
            balance_image[right_mask > 0] = [0, 0, 255]

        # Draw metrics text
        text_lines = [
            f"Balance Score: {balance_info['score']:.3f}",
            f"Status: {'Balanced' if balance_info['is_balanced'] else 'Unbalanced'}",
            f"Mass Ratio: {balance_info['mass_ratio']:.3f}",
            f"Lean: {balance_info['lean_direction']} ({balance_info['com_offset']:.3f})",
            f"Row Bias: {balance_info['row_bias_mean']:.3f} ± {balance_info['row_bias_std']:.3f}",
            f"Edge Ratio: {balance_info['edge_ratio']:.3f}"
        ]

        self._draw_text(balance_image, text_lines, center_x, balance_info['is_balanced'])

        cv2.imwrite(output_path, balance_image)
        print(f"Saved balance analysis to {output_path}")

    def _draw_text(self, image, text_lines, center_x, is_balanced):
        """Helper method for drawing text"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        y_offset = 30
        color = (0, 255, 0) if is_balanced else (0, 0, 255)

        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_x = int(center_x - text_size[0] / 2)

            cv2.putText(image, line,
                        (text_x, y_offset),
                        font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(image, line,
                        (text_x, y_offset),
                        font, font_scale, color, thickness)
            y_offset += 20

    def analyze_and_draw_balance(self, image, mask, output_path):
        """Convenience method that combines analysis and drawing"""
        balance_info = self.analyze_balance(mask)
        if balance_info:
            self.draw_balance(image, mask, balance_info, output_path)
        return balance_info


class SAMImageSegmenter(ImageSegmenter, SymmetryAnalysisMixin, BalanceAnalysisMixin):
    def __init__(self, model_type="vit_b", model_path="../ai-models/"):
        pass

    def _load_model(self):
        pass

    def _create_mask_generator(self):
        pass

    def load_image(self, image_path):
        pass

    def generate_masks(self, image):
        pass

    def visualize_masks(self, image, masks):
        pass

    def process_image(self, input_path, output_path, visualize_all_masks=False):
        pass

    def draw_bounding_boxes(self, image, masks, output_path):
        pass

    def draw_skeletons(self, image, masks, output_path):
        pass

    def draw_max_width_box(self, image, masks, output_path, edge_threshold=5):
        pass

    def filter_max_width_mask(self, masks, image_shape, edge_threshold=5):
        pass


class SmallFPCFastImageSegmenter(ImageSegmenter):
    def __init__(self, model_type="x", model_path="../ai-models/", angle_difference_threshold=0.5):
        super().__init__(model_path)
        model_file = f"FastSAM-{model_type}.pt"
        self.model_path = os.path.join(model_path, model_file)

        # Initialize image processing worker
        self.image_processor = ImageProcessor()

        # To store current mask of fpc or connector
        self.current_result_mask = None

        # To store box sizes
        self.box_sizes = []  # Initialize a list to store box sizes

        # Store current box size
        self.current_box_size = (0, 0)  # Initialize current box size variable

        # Time model loading
        start_time = time.time()
        self.model = self._load_model()
        load_time = time.time() - start_time

        # Flags
        self.is_image_shown = False
        self.angle_difference_threshold = angle_difference_threshold  # Set angle difference threshold

        print(f"Loaded FastSAM-{model_type} model from {self.model_path}")
        print(f"Model loading time: {load_time:.2f} seconds")

    def _load_model(self):
        """Load the FastSAM model"""
        model = FastSAM(self.model_path)
        return model

    def visualize_masks(self, image, masks):
        """Visualize segmentation masks"""
        result_image = image

        # Assign single mask to current result mask
        if len(masks) == 1:
            self.current_result_mask = masks[0]

        # Convert masks to numpy if they're on GPU
        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        if hasattr(masks, 'numpy'):
            masks = masks.numpy()

        # Handle each mask
        for i in range(len(masks)):
            color = np.random.rand(3) * 255
            mask_area = masks[i].astype(bool)
            result_image[mask_area] = result_image[mask_area] * 0.5 + color * 0.5

        # Show the final image if the flag is set
        if self.is_image_shown:
            cv2.imshow("Segmented Image", result_image)
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows()  # Close the window after key press

        return result_image

    def process_image(self, input_path, output_path, pin_count, visualize_all_masks=False, input_image_type='fpc'):
        """
        Process a single image and save the result with optional mask visualization.

        Args:
            input_path (str): Path to the input image to be processed.
            output_path (str): Path to save the output image.
            visualize_all_masks (bool): If True, visualize all masks. If False, only visualize the mask with the maximum width.
            input_image_type (str): Type of part to be processed; can be 'fpc' or 'connector'.

        Returns:
            Tuple: A tuple containing:
                - result (ndarray or None): The processed image with visualizations or None if no suitable mask was found.
                - output_path (str or None): The path to the saved output image or None if no suitable mask was found.
                - angle_difference (float or None): The angle difference calculated from the skeleton visualization or None if not applicable.
                - skeleton_image (ndarray or None): The image with skeleton overlay or None if not applicable.
                - skeleton_output_path (str or None): The path to the saved skeleton visualization or None if not applicable.

        Raises:
            Exception: If there is an error during image processing or mask generation.

        Notes:
            - This method uses a model to generate masks and then visualizes them based on the specified criteria.
            - It calculates bounding boxes and can save visualizations for both all masks and the maximum width mask.
            - The method times the processing duration and prints it to the console for performance monitoring.
            :param pin_count: Pin count of the model
        """

        print(f"Processing {input_path}...")

        # Time the entire processing
        start_time = time.time()

        # Load image
        image = cv2.imread(input_path)

        confidence = None
        iou = None
        min_width = None
        max_width = None
        min_height = None
        max_height = None
        is_edge_check = True
        if input_image_type == 'connector':
            print('Processing connector image')
            confidence = 0.2
            if pin_count == 12:
                # For 12 pins model
                iou = 0.7
                min_width = 650
                max_width = 900
                min_height = 120
                max_height = 700
            else:
                # For 10 pins model
                iou = 0.7
                min_width = 500
                max_width = 720
                min_height = 120
                max_height = 300

            is_edge_check = False
        elif input_image_type == 'fpc':
            print('Processing fpc image')
            confidence = 0.2
            if pin_count == 12:
                iou = 0.9
                min_width = 610
                max_width = 860
                min_height = 245
                # max_height = 290
                # max_height = float('inf')
                max_height = 380
            else:
                iou = 0.9
                min_width = 550
                max_width = 860
                min_height = 245
                # max_height = 290
                # max_height = float('inf')
                max_height = 380
            is_edge_check = True

        # Generate everything results
        everything_results = self.model(
            input_path,
            device=self.device,
            retina_masks=True,
            imgsz=1024,
            conf=confidence,
            iou=iou,
        )

        # Check if everything_results is None
        if everything_results is None:
            print("Error: Model returned no results.")
            return None, None, None, None, None, None

        if is_edge_check:
            print("Turn on edge check")
        else:
            print("Turn off edge check")

        # Get masks data
        masks = everything_results[0].masks.data

        if visualize_all_masks:
            # Create base output path without extension
            base_output_path, _ = os.path.splitext(output_path)
            bbox_output_path = f"{base_output_path}.png"

            # Save visualization of all masks
            result_image = self.visualize_masks(image, masks)
            # cv2.imwrite(f"{base_output_path}_all_masks.png", result)

            # Draw bounding boxes on all masks
            self.draw_bounding_boxes(result_image, masks, bbox_output_path)

            # Save skeleton visualization for all masks
            # skeleton_output = f"{base_output_path}_all_skeletons.png"
            # skeleton_output = f"{base_output_path}.png"
            # angle_difference, skeleton_image, skeleton_output_path = self.draw_skeletons(result, masks, skeleton_output)
            angle_difference, skeleton_image, skeleton_output_path = None, None, None
            if input_image_type == 'connector':
                return result_image, masks, bbox_output_path, None
            # Return outputs for further processing
            return result_image, output_path, angle_difference, skeleton_image, skeleton_output_path

        else:
            # Filter for max width mask
            selected_mask, box_coords = self.filter_max_width_mask(masks, image.shape, min_width=min_width,
                                                                   max_width=max_width, min_height=min_height,
                                                                   max_height=max_height,
                                                                   input_image_type=input_image_type,
                                                                   check_edges=is_edge_check, debug=False)

            if selected_mask is not None:
                # Create base output path without extension
                base_output_path, _ = os.path.splitext(output_path)
                output_path = f"{base_output_path}.png"

                # Create single mask list for visualization
                single_mask = [selected_mask]  # For FastSAM, just use the mask directly

                # Save regular segmentation
                result_image = self.visualize_masks(image, single_mask)
                # cv2.imwrite(f"{base_output_path}.png", result)

                # Draw bounding boxes on the selected mask
                self.draw_bounding_boxes(result_image, single_mask, output_path)

                # Save skeleton visualization
                # skeleton_output = f"{base_output_path}_skeleton.png"
                # skeleton_output = output_path
                if input_image_type == 'fpc':
                    angle_difference, skeleton_image, skeleton_output_path = self.draw_skeletons(
                        result_image, single_mask, output_path
                    )

                    process_time = time.time() - start_time
                    print(f"Processing time for fpc: {process_time:.2f} seconds")
                    # Return outputs for further processing
                    return result_image, output_path, angle_difference, skeleton_image, skeleton_output_path, "{:.2f}".format(
                        process_time)

                if input_image_type == 'connector':
                    process_time = time.time() - start_time
                    print(f"Processing time for connector: {process_time:.2f} seconds")
                    return result_image, single_mask, output_path, "{:.2f}".format(process_time)
            else:
                if input_image_type == 'fpc':
                    # Try to search for the closest matched one
                    print("Finding max width mask with edge check returns nothing. Try again without checking")
                    # Filter for max width mask
                    selected_mask, box_coords = self.filter_max_width_mask(masks, image.shape, min_width=min_width,
                                                                           max_width=max_width, min_height=min_height,
                                                                           max_height=500, check_edges=False,
                                                                           debug=False)

                    if selected_mask is not None:
                        # Create base output path without extension
                        base_output_path, _ = os.path.splitext(output_path)
                        output_path = f"{base_output_path}.png"

                        # Create single mask list for visualization
                        single_mask = [selected_mask]  # For FastSAM, just use the mask directly

                        # Save regular segmentation
                        result_image = self.visualize_masks(image, single_mask)
                        # cv2.imwrite(f"{base_output_path}.png", result)

                        # Draw bounding boxes on the selected mask
                        self.draw_bounding_boxes(result_image, single_mask, output_path)

                        # Save skeleton visualization
                        # skeleton_output = f"{base_output_path}_skeleton.png"
                        # skeleton_output = output_path
                        angle_difference, skeleton_image, skeleton_output_path = self.draw_skeletons(
                            result_image, single_mask, output_path
                        )

                        process_time = time.time() - start_time
                        print(f"Processing time: {process_time:.2f} seconds")

                        # Return outputs for further processing
                        return result_image, output_path, angle_difference, skeleton_image, skeleton_output_path, "{:.2f}".format(
                            process_time)

                    print("No suitable masks found away from edges")
                if input_image_type == 'connector':
                    print("No selected mask found for connector image. Return entirely None")
                    return None, None, None, None

        process_time = time.time() - start_time
        print(f"Processing time: {process_time:.2f} seconds")
        print("No selected mask found. Return entirely None")

        return None, None, None, None, None, "{:.2f}".format(process_time)  # Return None if no mask found

    def draw_bounding_boxes(self, image, masks, output_path):
        """Draw bounding boxes with dimensions on the image and record sizes.

        Args:
            image: The input image to draw on.
            masks: The masks to be processed.
            output_path: Path to save the output image.
        """
        result_image = image  # Create a copy of the image to draw on

        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        if hasattr(masks, 'numpy'):
            masks = masks.numpy()

        for i in range(len(masks)):
            mask = masks[i]
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            # Regular bounding box
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            w, h = x2 - x1, y2 - y1

            # Check if the box meets the width and height criteria
            # if w < min_width or w > max_width or h < min_height or h > max_height:
            #    print(
            #        f"Mask {i}: Skipped - Width: {w}, Height: {h} (min: {min_width}, max: {max_width}, min height: {min_height}, max height: {max_height})")
            #    continue  # Skip this mask if it doesn't meet the criteria

            # Assign current box size
            self.current_box_size = (w, h)

            # Record box size
            self.box_sizes.append((w, h))

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare and draw text for bounding box
            bbox_text = f"BBox: x:{x1}, y:{y1}, w:{w}, h:{h}"
            bbox_font = cv2.FONT_HERSHEY_SIMPLEX
            bbox_font_scale = 0.5
            bbox_thickness = 1
            bbox_text_size = cv2.getTextSize(bbox_text, bbox_font, bbox_font_scale, bbox_thickness)[0]
            bbox_text_x = x1 + (w - bbox_text_size[0]) // 2
            # Set Y position to be half the distance from the top of the bounding box
            bbox_text_y = y1 + (h + bbox_text_size[1]) // 2 - (y2 - y1) // 4

            cv2.putText(result_image, bbox_text, (bbox_text_x, bbox_text_y), bbox_font, bbox_font_scale, (0, 0, 0),
                        bbox_thickness + 1)
            cv2.putText(result_image, bbox_text, (bbox_text_x, bbox_text_y), bbox_font, bbox_font_scale,
                        (255, 255, 255), bbox_thickness)

        # Save results
        cv2.imwrite(output_path, result_image)  # Save the image
        print(f"Saved bounding box visualization to {output_path}")

    def plot_box_size_statistics(self):
        """Plot statistics for box height and width distributions using points"""
        if not self.box_sizes:
            print("No box sizes recorded.")
            return

        widths, heights = zip(*self.box_sizes)  # Unzip the recorded sizes

        # Create a figure with two subplots
        plt.figure(figsize=(12, 6))

        # Plot width distribution as points
        plt.subplot(1, 2, 1)
        unique_widths = sorted(set(widths))  # Get unique widths
        width_counts = [widths.count(w) for w in unique_widths]  # Count occurrences
        plt.scatter(unique_widths, width_counts, color='blue', alpha=0.7, label='Width')
        plt.title('Width Distribution')
        plt.xlabel('Width')
        plt.ylabel('Frequency')

        # Annotate each point with its value
        for w, count in zip(unique_widths, width_counts):
            plt.annotate(f'{count}', (w, count), textcoords="offset points", xytext=(0, 10), ha='center')

        # Plot height distribution as points
        plt.subplot(1, 2, 2)
        unique_heights = sorted(set(heights))  # Get unique heights
        height_counts = [heights.count(h) for h in unique_heights]  # Count occurrences
        plt.scatter(unique_heights, height_counts, color='green', alpha=0.7, label='Height')
        plt.title('Height Distribution')
        plt.xlabel('Height')
        plt.ylabel('Frequency')

        # Annotate each point with its value
        for h, count in zip(unique_heights, height_counts):
            plt.annotate(f'{count}', (h, count), textcoords="offset points", xytext=(0, 10), ha='center')

        plt.tight_layout()
        plt.show()

    def draw_skeletons(self, image, masks, output_path, debug=False):
        """Draw skeletons and direction analysis, returning angle difference and output image path.

        Args:
            image: The input image to draw on.
            masks: The masks to be processed.
            output_path: Path to save the output image.
            debug: If True, print debug information; otherwise, suppress prints.

        Returns:
            tuple: (angle_difference, skeleton_image, output_path)
        """
        from skimage.morphology import skeletonize

        skeleton_image = image
        angle_difference = 0  # Initialize angle_difference

        # Convert masks if needed
        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        if hasattr(masks, 'numpy'):
            masks = masks.numpy()

        if debug:
            print(f"Processing {len(masks)} masks...")  # Print number of masks

        for i in range(len(masks)):
            mask = masks[i].astype(bool)
            skeleton = skeletonize(mask)
            skeleton_image[skeleton] = [255, 0, 0]  # Assign red color to the skeleton

            # Calculate moments for the mask
            mask_uint8 = mask.astype(np.uint8) * 255  # Convert to uint8 for moment calculation
            moments = cv2.moments(mask_uint8)

            if moments['m00'] == 0:
                continue  # Skip if the mask is empty

            # Calculate centroid from moments
            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            # Get the coordinates of the mask
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0:
                continue

            # Prepare points for PCA
            points = np.column_stack((x_indices, y_indices))  # x,y format

            # Perform PCA to find the main direction
            mean = np.mean(points, axis=0)
            centered_points = points - mean
            cov_matrix = np.cov(centered_points.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

            # The direction of the first eigenvector is the main orientation
            main_direction = eigenvectors[:, 0]
            angle_difference = np.degrees(
                np.arctan2(main_direction[1], main_direction[0]))  # Angle difference in degrees

            if debug:
                print(f"Angle difference for mask {i}: {angle_difference:.2f} degrees")

            # Draw arrows at fixed angles: 0, 90, 180, 270 degrees
            arrow_length = min(image.shape[:2]) // 4
            fixed_actual_angles = [0, 90, 180, 270]  # Fixed expected angles
            actual_angles = [angle_difference, angle_difference + 90, angle_difference + 180, angle_difference + 270]

            for index, arr_angle in enumerate(fixed_actual_angles):
                # Calculate the end coordinates based on the angle difference
                rad_angle = np.radians(float(actual_angles[index]))  # Use actual_angles for calculations
                end_x = int(center_x + arrow_length * np.cos(rad_angle))
                end_y = int(center_y + arrow_length * np.sin(rad_angle))

                # Initialize arrow color with a default value
                arrow_color = (0, 255, 255)  # Default to yellow

                # Calculate the angle difference from the expected direction
                angle_diff = abs(arr_angle - actual_angles[index])  # Difference in degrees

                if debug:
                    print(f"Angle difference for arrow at {arr_angle:.2f} degrees: {angle_diff:.2f} degrees")

                # Check if the angle difference exceeds 0.5 degrees
                if angle_diff > self.angle_difference_threshold:
                    arrow_color = (0, 0, 255)  # Change to red if it exceeds the offset

                cv2.arrowedLine(skeleton_image,
                                (int(center_x), int(center_y)),
                                (end_x, end_y),
                                arrow_color, 3, tipLength=0.3)

                if debug:
                    print(
                        f"Drawing arrow at angle: {arr_angle:.2f} degrees with color: {'red' if arrow_color == (0, 0, 255) else 'yellow'}")

                # Add angle label
                label_x = int(center_x + (arrow_length + 30) * np.cos(rad_angle))
                label_y = int(center_y + (arrow_length + 30) * np.sin(rad_angle))
                display_angle = float(actual_angles[index] % 360)  # Use actual_angles for display
                if display_angle > 180:
                    display_angle -= 360

                # Use "deg" instead of "°" to avoid encoding issues
                text = f"{display_angle:.1f} deg"  # Display angle to one decimal place
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(skeleton_image, text,
                            (label_x, label_y),
                            font, 0.7, (0, 0, 0), 3)
                cv2.putText(skeleton_image, text,
                            (label_x, label_y),
                            font, 0.7, arrow_color, 1)

            if debug:
                print(f"Finished processing mask {i}")

        cv2.imwrite(output_path, skeleton_image)
        print(f"Saved skeleton visualization to {output_path}")

        # Show the final skeleton image if the flag is set
        if self.is_image_shown:
            cv2.imshow("Skeleton Image", skeleton_image)
            cv2.waitKey(0)  # Wait indefinitely for a key press
            cv2.destroyAllWindows()  # Close the window after key press

        return angle_difference, skeleton_image, output_path  # Return angle_difference, output image, and path

    def draw_label(self, image, output_path, label, color):
        """Draw label on the top left corner of the image"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        position = (10, text_size[1] + 10)  # Position with some padding
        cv2.putText(image, label, position, font, font_scale, color, thickness)
        # Save image
        cv2.imwrite(output_path, image)
        print("Output path: " + output_path)

    def draw_max_width_box(self, image, masks, output_path, edge_threshold=5):
        """
        Draw only the box with maximum width that's not near image edges
        Args:
            image: Input image
            masks: Segmentation masks
            output_path: Where to save the result
            edge_threshold: Minimum distance from image edges (default 5 pixels)
        """
        result_image = image.copy()
        height, width = image.shape[:2]
        max_width = 0
        selected_box = None
        selected_mask = None

        # Convert masks if needed (for FastSAM)
        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        if hasattr(masks, 'numpy'):
            masks = masks.numpy()

        # Process each mask
        for i in range(len(masks)):
            # Get mask
            mask = masks[i].astype(bool) if isinstance(masks, np.ndarray) else masks[i]['segmentation']
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            # Get bounding box
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            w = x2 - x1

            # Check if box is near edges
            if (x1 < edge_threshold or  # Too close to left
                    y1 < edge_threshold or  # Too close to top
                    x2 > width - edge_threshold or  # Too close to right
                    y2 > height - edge_threshold):  # Too close to bottom
                continue

            # Update if this is the widest box so far
            if w > max_width:
                max_width = w
                selected_box = (x1, y1, x2, y2)
                selected_mask = mask

        # Draw the selected box if found
        if selected_box is not None:
            x1, y1, x2, y2 = selected_box
            w = x2 - x1
            h = y2 - y1

            # Draw the mask with random color
            color = np.random.rand(3) * 255
            mask_area = selected_mask
            result_image[mask_area] = result_image[mask_area] * 0.5 + color * 0.5

            # Draw rectangle
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw text
            text = f"x:{x1},y:{y1}\nw:{w},h:{h}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1

            # Calculate text position
            text_size = cv2.getTextSize(text.replace('\n', ' '), font, font_scale, thickness)[0]
            text_x = x1 + (w - text_size[0]) // 2
            text_y = y1 + (h + text_size[1]) // 2

            # Draw text with background
            for j, line in enumerate(text.split('\n')):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                y = text_y + j * text_size[1]
                cv2.putText(result_image, line, (text_x, y), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(result_image, line, (text_x, y), font, font_scale, (255, 255, 255), thickness)

            print(f"Found max width box: width={w}, height={h}, at (x={x1}, y={y1})")
        else:
            print("No suitable boxes found away from edges")

        cv2.imwrite(output_path, result_image)
        print(f"Saved max width box visualization to {output_path}")

    def filter_max_width_mask(self, masks, image_shape, edge_threshold=5, min_width=float('inf'),
                              max_width=float('inf'), min_height=float('inf'), max_height=float('inf'),
                              input_image_type='fpc', check_edges=True, debug=False):
        """
        Filter and return the mask with maximum width within specified range away from edges.

        Args:
            masks: Input masks to filter.
            image_shape: Shape of the image (height, width).
            edge_threshold: Minimum distance from the edges.
            min_width: Minimum width of the mask to consider.
            max_width: Maximum width of the mask to consider.
            min_height: Minimum height of the mask to consider.
            max_height: Maximum height of the mask to consider.
            check_edges (bool): If True, check for proximity to image edges; otherwise, skip edge checks.
            debug: If True, print debug information; otherwise, suppress prints.

        Returns:
            tuple: (selected_mask, box_coords) or (None, None) if no suitable mask found.
        """
        height, width = image_shape[:2]
        print(f"Image width and height: {width}, {height}")
        selected_mask = None
        selected_box = None
        max_width_found = 0

        # Convert masks if needed
        if hasattr(masks, 'cpu'):
            masks = masks.cpu()
        if hasattr(masks, 'numpy'):
            masks = masks.numpy()

        if debug:
            print(f"Processing {len(masks)} masks...")  # Print number of masks

        for i in range(len(masks)):
            # FastSAM masks are already boolean arrays
            mask = masks[i].astype(bool)
            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue

            # Get bounding box
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            w = x2 - x1
            h = y2 - y1  # Calculate height

            if debug:
                print(
                    f"Mask {i}: Width = {w}, Height = {h}, X1 = {x1}, Y1 = {y1}, X2 = {x2}, Y2 = {y2}")  # Print width and height for each box

            # Check if box satisfies width and height criteria
            if not (min_width <= w <= max_width):
                if debug:
                    print(f"Mask {i}: Width out of range, skipping.")  # Print if width is out of range
                continue

            if not (min_height <= h <= max_height):
                if debug:
                    print(f"Mask {i}: Height out of range, skipping.")  # Print if height is out of range
                continue

            # Check if the bounding box is near the image edges if edge checks are enabled
            if check_edges:
                if x1 <= edge_threshold:  # Check if the left edge is too close to the left boundary
                    if debug:
                        print(f"Mask {i}: Too close to the left edge, skipping.")
                    continue  # Skip this mask

                if y1 <= edge_threshold:  # Check if the top edge is too close to the top boundary
                    if debug:
                        print(f"Mask {i}: Too close to the top edge, skipping.")
                    continue  # Skip this mask

                if x2 >= width - edge_threshold:  # Check if the right edge is too close to the right boundary
                    if debug:
                        print(f"Mask {i}: Too close to the right edge, skipping.")
                    continue  # Skip this mask

                if y2 >= height - edge_threshold:  # Check if the bottom edge is too close to the bottom boundary
                    if debug:
                        print(f"Mask {i}: Too close to the bottom edge, skipping.")
                    continue  # Skip this mask

            if input_image_type == 'connector':
                if y1 <= 20:
                    print(f"Mask {i}: Very close to top edge, skipping")
                    continue

            # Update if this is the widest box so far
            if w > max_width_found:
                max_width_found = w
                selected_box = (x1, y1, x2, y2)
                selected_mask = mask
                if debug:
                    print(f"Mask {i}: Selected as widest box with Width = {w}")  # Print if selected

        if selected_mask is None and debug:
            print("No suitable mask found.")  # Print if no mask is selected

        return selected_mask, selected_box

    def check_connector_lock_defect(self, is_image_shown, extracted_connector_image, connector_result_mask,
                                    expected_pin_count,
                                    expected_min_top_left_pixel_density, expected_max_top_left_pixel_density,
                                    expected_min_top_right_pixel_density, expected_max_top_right_pixel_density,
                                    left_offset, right_offset):
        self.image_processor.is_images_shown = is_image_shown
        if self.image_processor.check_connector_lock_defect(extracted_connector_image, connector_result_mask,
                                                            expected_pin_count, expected_min_top_left_pixel_density,
                                                            expected_max_top_left_pixel_density,
                                                            expected_min_top_right_pixel_density,
                                                            expected_max_top_right_pixel_density,
                                                            left_offset, right_offset):
            print("Connector Lock Defect Check: OK")
            return True
        else:
            print("Connector Lock Defect Check: NG")
            return False

    def check_fpc_lead_balance(self, image, output_path, angle_difference, lower_width_threshold,
                               higher_width_threshold, lower_height_threshold, higher_height_threshold,
                               mask, edge_threshold=10):
        """Check if the FPC lead is balanced based on angle difference, box size thresholds, and edge proximity."""

        # Angle difference offset: 0.3 degree
        angle_diff_offset = 0.1

        # Check angle difference
        if abs(angle_difference + angle_diff_offset) > self.angle_difference_threshold:
            failure_cause = f"Angle difference: {angle_difference:.2f} degrees"
            print(f"FPC lead is not balanced with {failure_cause}")
            label = f"NG - {failure_cause}"
            label_color = (0, 0, 255)  # Red color for NG
            self.draw_label(image, output_path, label, label_color)
            return False, output_path  # Return False and output image path

        print(f"FPC lead is balanced with angle difference: {angle_difference:.2f} degrees")

        # Check box size using current box size
        width, height = self.current_box_size  # Use current box size variable
        if not (lower_width_threshold <= width <= higher_width_threshold):
            failure_cause = f"Box width {width} out of range ({lower_width_threshold}, {higher_width_threshold})"
            print(failure_cause)  # Print the failure cause directly
            label = f"NG - {failure_cause}"
            label_color = (0, 0, 255)  # Red color for NG
            self.draw_label(image, output_path, label, label_color)
            return False, output_path  # Width check failed

        if not (lower_height_threshold <= height <= higher_height_threshold):
            failure_cause = f"Box height {height} out of range ({lower_height_threshold}, {higher_height_threshold})"
            print(failure_cause)  # Print the failure cause directly
            label = f"NG - {failure_cause}"
            label_color = (0, 0, 255)  # Red color for NG
            self.draw_label(image, output_path, label, label_color)
            return False, output_path  # Height check failed

        # Get coordinates from the mask to check proximity to top and bottom edges
        y_indices, x_indices = np.where(mask)  # Get the indices of the mask
        if len(y_indices) == 0 or len(x_indices) == 0:
            print("Mask is empty, cannot determine edge proximity.")
            return False, output_path  # Return if the mask is empty

        # Calculate the bounding box from the mask
        y1, y2 = np.min(y_indices), np.max(y_indices)
        image_height = image.shape[0]  # Get the height of the image

        # Check proximity to top and bottom edges
        if y1 <= edge_threshold:
            failure_cause = f"Box is too close to the top edge ({y1} <= {edge_threshold})"
            print(failure_cause)
            label = f"NG - {failure_cause}"
            label_color = (0, 0, 255)  # Red color for NG
            self.draw_label(image, output_path, label, label_color)
            return False, output_path  # Top edge check failed

        if (image_height - y2) <= edge_threshold:
            failure_cause = f"Box is too close to the bottom edge ({image_height - y2} <= {edge_threshold})"
            print(failure_cause)
            label = f"NG - {failure_cause}"
            label_color = (0, 0, 255)  # Red color for NG
            self.draw_label(image, output_path, label, label_color)
            return False, output_path  # Bottom edge check failed

        print(f"Box size is balanced with width: {width} and height: {height}")
        label = "OK"
        label_color = (255, 0, 0)  # Blue color for OK
        self.draw_label(image, output_path, label, label_color)

        return True, output_path  # All checks passed

    def extract_connector_masked_image(self, image, mask, display_masked_image=False):
        """
        Extract the area from the input image corresponding to the bounding box of the given mask.

        Args:
            image (ndarray): The input image from which to extract the masked area.
            mask (ndarray): A boolean mask indicating the area to extract.
            display_masked_image (bool): If True, display the extracted image area; defaults to False.

        Returns:
            ndarray: The cropped image corresponding to the bounding box of the mask.
        """
        # Ensure the mask is a boolean array
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Get the coordinates of the non-zero (True) mask pixels
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0 or len(y_indices) == 0:
            # If the mask is empty or has no True values, return None or an empty image
            print("Mask is empty. No bounding box to extract.")
            return None

        # Calculate the bounding box (x1, y1, x2, y2)
        x1, x2 = np.min(x_indices), np.max(x_indices)
        y1, y2 = np.min(y_indices), np.max(y_indices)

        # Crop the image using the bounding box
        cropped_image = image[y1:y2 + 1, x1:x2 + 1]

        # Optionally display the cropped image
        if display_masked_image:
            cv2.imshow("Extracted Area", cropped_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return cropped_image


class TokenFPCFastImageSegmenter(ImageSegmenter):
    """
    Specialized FastSAM-based image segmentation class for inspecting Token FPC parts.
    Processes 1440x1080 images, visualizing masks and drawing bounding boxes.
    When visualize_all_masks=False, only processes masks within predefined mark areas.
    Only bounding box images are saved to C:\\BoardDefectChecker\\ai-outputs\\token-fpc-dd-mm-yyyy\\,
    with a label in the top-right corner showing processing times.
    """

    def __init__(self, model_type="x", model_path="../ai-models/", fpc_two_marks_height_diff=9,
                 output_directory=r"C:\BoardDefectChecker\ai-outputs", is_image_shown=False,
                 left_right_offset=50, connector_offset=30):
        super().__init__(model_path)
        model_file = f"FastSAM-{model_type}.pt"
        self.model_path = os.path.join(model_path, model_file)

        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.is_image_shown = is_image_shown
        self.fpc_two_marks_height_diff = fpc_two_marks_height_diff

        self.predefined_areas = {
            "left_fpc_mark": (182, 324, 340, 168),
            "right_fpc_mark": (886, 324, 340, 168),
            "connector_mark": (0, 636, 1440, 312)
        }

        self.predefined_mask_sizes = {
            "left_fpc_mark": (236, 39),
            "right_fpc_mark": (236, 39),
            "connector_mark": (1092, 119)
        }

        self.left_right_offset = left_right_offset
        self.connector_offset = connector_offset
        self.fpc_two_marks_height_diff = fpc_two_marks_height_diff

        self.size_ranges = {
            "left_fpc_mark": (
                self.predefined_mask_sizes["left_fpc_mark"][0] - self.left_right_offset,
                self.predefined_mask_sizes["left_fpc_mark"][0] + self.left_right_offset,
                self.predefined_mask_sizes["left_fpc_mark"][1] - self.left_right_offset,
                self.predefined_mask_sizes["left_fpc_mark"][1] + self.left_right_offset
            ),
            "right_fpc_mark": (
                self.predefined_mask_sizes["right_fpc_mark"][0] - self.left_right_offset,
                self.predefined_mask_sizes["right_fpc_mark"][0] + self.left_right_offset,
                self.predefined_mask_sizes["right_fpc_mark"][1] - self.left_right_offset,
                self.predefined_mask_sizes["right_fpc_mark"][1] + self.left_right_offset
            ),
            "connector_mark": (
                self.predefined_mask_sizes["connector_mark"][0] - self.connector_offset,
                self.predefined_mask_sizes["connector_mark"][0] + self.connector_offset,
                self.predefined_mask_sizes["connector_mark"][1] - self.connector_offset,
                self.predefined_mask_sizes["connector_mark"][1] + self.connector_offset
            )
        }

        start_time = time.time()
        self.model = self._load_model()
        load_time = time.time() - start_time

        print(f"Loaded FastSAM-{model_type} model from {self.model_path}")
        print(f"Model loading time: {load_time:.2f} seconds")
        print(
            f"Mask size ranges with offsets - Left/Right: ±{self.left_right_offset}, Connector: ±{self.connector_offset}")
        print(f"Left FPC mask: w={self.size_ranges['left_fpc_mark'][0]}-{self.size_ranges['left_fpc_mark'][1]}, "
              f"h={self.size_ranges['left_fpc_mark'][2]}-{self.size_ranges['left_fpc_mark'][3]}")
        print(f"Right FPC mask: w={self.size_ranges['right_fpc_mark'][0]}-{self.size_ranges['right_fpc_mark'][1]}, "
              f"h={self.size_ranges['right_fpc_mark'][2]}-{self.size_ranges['right_fpc_mark'][3]}")
        print(f"Connector mask: w={self.size_ranges['connector_mark'][0]}-{self.size_ranges['connector_mark'][1]}, "
              f"h={self.size_ranges['connector_mark'][2]}-{self.size_ranges['connector_mark'][3]}")

    def _load_model(self):
        """Load the FastSAM model."""
        return FastSAM(self.model_path)

    def _generate_output_path(self, prefix):
        """Generate a unique output path for saving images."""
        timestamp = datetime.now()
        date_str = timestamp.strftime("%d-%m-%Y")
        subdir = os.path.join(self.output_directory, f"token-fpc-{date_str}")
        os.makedirs(subdir, exist_ok=True)
        filename = f"{prefix}_{timestamp.strftime('%Y%m%d_%H%M%S')}.bmp"
        return os.path.join(subdir, filename)

    def _filter_masks_in_areas(self, masks):
        """
        Filter masks to only those within predefined areas and matching size ranges.
        """
        filtered_masks = {}
        for name, (x, y, w, h) in self.predefined_areas.items():
            min_w, max_w, min_h, max_h = self.size_ranges[name]
            largest_mask = None
            max_area = 0

            for mask in masks:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                mask_x1, mask_y1 = np.min(x_indices), np.min(y_indices)
                mask_x2, mask_y2 = np.max(x_indices), np.max(y_indices)
                mask_w = mask_x2 - mask_x1 + 1
                mask_h = mask_y2 - mask_y1 + 1

                if (mask_x1 >= x and mask_x2 <= x + w and mask_y1 >= y and mask_y2 <= y + h and
                        min_w <= mask_w <= max_w and min_h <= mask_h <= max_h):
                    area_size = np.sum(mask)
                    if area_size > max_area:
                        max_area = area_size
                        largest_mask = mask

            if largest_mask is not None:
                filtered_masks[name] = largest_mask

        return filtered_masks

    def visualize_masks(self, image, masks, visualize_all_masks=True, all_masks=None):
        """
        Visualize segmentation masks on the image with yellow rectangles for predefined areas.
        """
        result_image = image.copy()

        for area in self.predefined_areas.values():
            x, y, w, h = area
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        if visualize_all_masks and all_masks is not None:
            for mask in all_masks:
                color = np.random.rand(3) * 255
                mask_area = mask.astype(bool)
                result_image[mask_area] = result_image[mask_area] * 0.5 + color * 0.5
        else:
            for mask in masks.values():
                if mask is not None:
                    color = np.array([0, 255, 0])  # Green for filtered masks
                    mask_area = mask.astype(bool)
                    result_image[mask_area] = result_image[mask_area] * 0.5 + color * 0.5

        if self.is_image_shown:
            resized_img = cv2.resize(result_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow("Visualized Masks", resized_img)
            key = cv2.waitKey(0)
            if key == 32:
                cv2.destroyWindow("Visualized Masks")

        return result_image

    def draw_bounding_boxes(self, image, masks, visualize_all_masks=True, all_masks=None,
                            font_scale=1, ai_time=0.0, non_ai_time=0.0, total_time=0.0,
                            label_color=(255, 0, 0), check_result=None):
        """
        Draw green bounding boxes with size labels around masks, plus a time label and Check result/Reason with light gray background.
        """
        result_image = image.copy()

        # Draw yellow rectangles for predefined areas
        for area in self.predefined_areas.values():
            x, y, w, h = area
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Store connector info for distance arrows
        connector_info = None
        if "connector_mark" in masks and masks["connector_mark"] is not None:
            conn_y_indices, conn_x_indices = np.where(masks["connector_mark"])
            conn_x1, conn_y1 = np.min(conn_x_indices), np.min(conn_y_indices)
            conn_x2, conn_y2 = np.max(conn_x_indices), np.max(conn_y_indices)
            connector_info = (conn_x1, conn_y1, conn_x2, conn_y2)

        # Draw bounding boxes
        if visualize_all_masks and all_masks is not None:
            for mask in all_masks:
                y_indices, x_indices = np.where(mask)
                if len(y_indices) == 0 or len(x_indices) == 0:
                    continue
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                w, h = x2 - x1 + 1, y2 - y1 + 1
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"X: {x1}, Y: {y1}, W: {w}, H: {h}"
                label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                label_x = x1 + (w - label_size[0]) // 2
                label_y = y1 - 10 - baseline
                cv2.rectangle(result_image, (label_x - 5, label_y - label_size[1] - 5),
                              (label_x + label_size[0] + 5, label_y + baseline + 5), (255, 255, 255), -1)
                cv2.putText(result_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)
        else:
            for name, mask in masks.items():
                if mask is not None:
                    y_indices, x_indices = np.where(mask)
                    x1, y1 = np.min(x_indices), np.min(y_indices)
                    x2, y2 = np.max(x_indices), np.max(y_indices)
                    w, h = x2 - x1 + 1, y2 - y1 + 1
                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"X: {x1}, Y: {y1}, W: {w}, H: {h}"
                    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                    label_x = x1 + (w - label_size[0]) // 2
                    label_y = y1 - 10 - baseline
                    cv2.rectangle(result_image, (label_x - 5, label_y - label_size[1] - 5),
                                  (label_x + label_size[0] + 5, label_y + baseline + 5), (255, 255, 255), -1)
                    cv2.putText(result_image, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)

                    if name in ["left_fpc_mark", "right_fpc_mark"] and connector_info:
                        mark_center_x = x1 + w // 2
                        mark_bottom_y = y2
                        conn_x1, conn_y1, _, _ = connector_info
                        connector_top_y = conn_y1
                        arrow_start = (mark_center_x, mark_bottom_y)
                        arrow_end = (mark_center_x, connector_top_y)
                        cv2.arrowedLine(result_image, arrow_start, arrow_end, (255, 0, 255), 2, tipLength=0.05)
                        cv2.arrowedLine(result_image, arrow_end, arrow_start, (255, 0, 255), 2, tipLength=0.05)
                        distance = abs(mark_bottom_y - connector_top_y)
                        distance_label = f"{distance}"
                        arrow_center_y = (mark_bottom_y + connector_top_y) // 2
                        distance_size, _ = cv2.getTextSize(distance_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                        distance_x = mark_center_x + (10 if name == "right_fpc_mark" else -distance_size[0] - 10)
                        distance_y = arrow_center_y + distance_size[1] // 2
                        cv2.rectangle(result_image, (distance_x - 5, distance_y - distance_size[1] - 5),
                                      (distance_x + distance_size[0] + 5, distance_y + 5), (255, 255, 255), -1)
                        cv2.putText(result_image, distance_label, (distance_x, distance_y), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (255, 0, 255), 2)

        # Add Check result/Reason label at top-left with light gray background
        if check_result:
            status, reason = check_result
            status_color = (255, 0, 0) if status == "OK" else (0, 0, 255)  # Blue for OK, Red for NG
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_size = 0.75  # Font size remains 0.75
            thickness = 2
            line_spacing = 15  # Matches smaller font

            # Text to display
            check_result_text = f"Check result: {status}"
            reason_text = f"Reason: {reason}"

            # Calculate text sizes
            check_size, baseline = cv2.getTextSize(check_result_text, font, font_size, thickness)
            reason_size, _ = cv2.getTextSize(reason_text, font, font_size, thickness)
            max_width = max(check_size[0], reason_size[0])
            total_height = check_size[1] + reason_size[1] + line_spacing
            padding = 10
            x_left = padding
            y_top = padding

            # Draw light gray background rectangle (same as time label)
            cv2.rectangle(result_image, (x_left - padding, y_top - padding),
                          (x_left + max_width + padding, y_top + total_height + padding), (200, 200, 200), -1)

            # Draw text
            cv2.putText(result_image, check_result_text, (x_left, y_top + check_size[1]), font, font_size, status_color,
                        thickness)
            cv2.putText(result_image, reason_text, (x_left, y_top + check_size[1] + line_spacing + reason_size[1]),
                        font, font_size, status_color, thickness)

        # Draw time label at top-right
        label_lines = [
            f"AI inference time: {ai_time:.2f}s",
            f"Without AI time: {non_ai_time:.2f}s",
            f"Total time: {total_time:.2f}s"
        ]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (255, 0, 0)  # Blue
        bg_color = (200, 200, 200)  # Light gray
        thickness = 1
        line_spacing = int(20 * font_scale) + 5

        text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in label_lines]
        max_width = max(size[0] for size in text_sizes)
        total_height = sum(size[1] for size in text_sizes) + (len(label_lines) - 1) * line_spacing
        padding = 10
        x_right = result_image.shape[1] - max_width - padding
        y_top = padding

        cv2.rectangle(result_image, (x_right - padding, y_top - padding),
                      (x_right + max_width + padding, y_top + total_height + padding), bg_color, -1)

        for i, (line, (w, h)) in enumerate(zip(label_lines, text_sizes)):
            y_pos = y_top + i * (h + line_spacing) + h
            cv2.putText(result_image, line, (x_right, y_pos), font, font_scale, text_color, thickness)

        bb_path = self._generate_output_path("bounding_boxes")
        cv2.imwrite(bb_path, result_image)
        print(f"[Debug] Bounding boxes image saved to: {bb_path}")

        if self.is_image_shown:
            resized_img = cv2.resize(result_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow("Bounding Boxes", resized_img)
            key = cv2.waitKey(0)
            if key == 32:
                cv2.destroyWindow("Bounding Boxes")

        return result_image, bb_path

    def check_connector_lock(self, masks):
        """Check connector lock defect. Returns (bool, reason)."""
        if "connector_mark" not in masks or masks["connector_mark"] is None:
            return False, "Connector mask not found"
        y_indices, _ = np.where(masks["connector_mark"])
        h = np.max(y_indices) - np.min(y_indices) + 1
        if h < 130:  # No offset here
            return False, f"Connector height {h} < 134"
        return True, "Connector OK"

    def check_token_fpc_balance(self, masks):
        """Check token FPC balance using top edge Y-values and distance using bottom edge Y-values. Returns (bool, reason)."""
        left_mask = masks.get("left_fpc_mark")
        right_mask = masks.get("right_fpc_mark")
        connector_mask = masks.get("connector_mark")

        if left_mask is None or right_mask is None:
            return False, "Left or right FPC mask not found"
        if connector_mask is None:
            return False, "Connector mask not found for distance check"

        # Get bounding box coordinates for left FPC mark
        left_y_indices, left_x_indices = np.where(left_mask)
        left_y1 = np.min(left_y_indices)  # Top edge
        left_y2 = np.max(left_y_indices)  # Bottom edge

        # Get bounding box coordinates for right FPC mark
        right_y_indices, right_x_indices = np.where(right_mask)
        right_y1 = np.min(right_y_indices)  # Top edge
        right_y2 = np.max(right_y_indices)  # Bottom edge

        # Get bounding box coordinates for connector mark
        conn_y_indices, conn_x_indices = np.where(connector_mask)
        conn_y1 = np.min(conn_y_indices)  # Top edge of connector
        conn_y2 = np.max(conn_y_indices)  # Bottom edge

        # Use top Y of bounding boxes for balance check (y1 is the top edge)
        left_top_y = left_y1
        right_top_y = right_y1

        # Check balance between left and right FPC marks using top edges
        if abs(left_top_y - right_top_y) > self.fpc_two_marks_height_diff:
            return False, f"Unbalanced FPC (Left Top Y: {left_top_y}, Right Top Y: {right_top_y}, Threshold: {self.fpc_two_marks_height_diff})"

        # Use bottom Y of bounding boxes for distance check (y2 is the bottom edge)
        left_bottom_y = left_y2
        right_bottom_y = right_y2
        connector_top_y = conn_y1

        # Check distance from FPC marks (bottom edge) to connector (top edge)
        left_distance = abs(left_bottom_y - connector_top_y)
        right_distance = abs(right_bottom_y - connector_top_y)
        max_distance = 284 + 16  # Only positive offset
        if left_distance > max_distance or right_distance > max_distance:
            return False, f"Distance to connector exceeds {max_distance} (Left: {left_distance}, Right: {right_distance})"

        return True, "FPC balance OK"

    def process_image(self, input_path, visualize_all_masks=False, font_scale=1, label_color=(255, 0, 0)):
        print(f"[Debug] Processing image: {input_path} ...")
        start_time = time.time()

        # Non-AI time: Loading image
        image_load_start = time.time()
        image = cv2.imread(input_path)
        if image is None:
            print("[Error] Could not load image.")
            return None, None, None, None, None
        image_load_time = time.time() - image_load_start

        if image.shape[:2] != (1080, 1440):
            print(f"[Warning] Image size is {image.shape[:2]}, expected (1080, 1440).")

        # AI inference time
        ai_start = time.time()
        result = self.model(input_path, device="cpu", retina_masks=True, imgsz=896, conf=0.4, iou=0.8)
        ai_time = time.time() - ai_start
        if result is None or len(result) == 0:
            print("[Error] Model returned no results.")
            return None, None, None, None, None

        # Prepare masks and flags
        all_masks = [mask.numpy() for mask in result[0].masks.data]
        filtered_masks = self._filter_masks_in_areas(all_masks) if not visualize_all_masks else {}
        has_left_fpc = "left_fpc_mark" in filtered_masks and filtered_masks["left_fpc_mark"] is not None
        has_right_fpc = "right_fpc_mark" in filtered_masks and filtered_masks["right_fpc_mark"] is not None
        has_connector = "connector_mark" in filtered_masks and filtered_masks["connector_mark"] is not None

        # Defect checks
        connector_ok, connector_reason = self.check_connector_lock(filtered_masks)
        fpc_ok, fpc_reason = self.check_token_fpc_balance(filtered_masks)
        overall_ok = connector_ok and fpc_ok
        overall_reason = "All OK" if overall_ok else (connector_reason if not connector_ok else fpc_reason)
        detection_result = "OK" if overall_ok else "NG"
        defect_reason = None if overall_ok else overall_reason

        # Non-AI time: Visualization and drawing
        viz_start = time.time()
        visualized_image = self.visualize_masks(image, filtered_masks, visualize_all_masks, all_masks)
        non_ai_time = image_load_time + (time.time() - viz_start)
        total_time = ai_time + non_ai_time

        # Draw bounding boxes with check result
        bb_image, bb_path = self.draw_bounding_boxes(
            visualized_image, filtered_masks, visualize_all_masks, all_masks, font_scale, ai_time, non_ai_time,
            total_time, label_color, check_result=(detection_result, overall_reason)
        )

        # Debug prints
        print(
            f"[Debug] Left FPC found: {has_left_fpc}, Right FPC found: {has_right_fpc}, Connector found: {has_connector}")
        print(f"[Debug] Connector check: {connector_ok}, Reason: {connector_reason}")
        print(f"[Debug] FPC balance check: {fpc_ok}, Reason: {fpc_reason}")
        print(f"[Debug] AI inference time: {ai_time:.2f} seconds")
        print(f"[Debug] Without AI time: {non_ai_time:.2f} seconds")
        print(f"[Debug] Total processing time: {total_time:.2f} seconds")

        # Return processing_time as float, not formatted string
        return visualized_image, bb_path, total_time, detection_result, defect_reason


# --- Base Class for FastSAM Segmenters ---
class BaseFastSamSegmenter:
    """Base class providing FastSAM model loading and inference."""

    def __init__(self, model_type="x", model_path="ai-models/", imgsz=1024, conf=0.2, iou=0.9):
        """
        Initializes the BaseFastSamSegmenter.

        Args:
            model_type (str): The type of FastSAM model (e.g., "x", "s").
            model_path (str): Path to the directory containing model files.
            imgsz (int): Image size for inference.
            conf (float): Confidence threshold for inference.
            iou (float): IoU threshold for inference.
        """
        self.device = "cpu"  # Semicolon removed
        model_file = f"FastSAM-{model_type}.pt"  # Semicolon removed
        self.model_full_path = os.path.join(model_path, model_file)
        self.imgsz = imgsz  # Semicolon removed
        self.conf = conf  # Semicolon removed
        self.iou = iou  # Semicolon removed
        self.is_image_shown = False  # Semicolon removed
        self.model = None
        start_time = time.time()  # Semicolon removed
        self.model = self._load_model()  # Semicolon removed
        load_time = time.time() - start_time
        if self.model:
            print(f"[{self.__class__.__name__} Base] Model loaded in {load_time:.2f}s. Path: {self.model_full_path}")
        else:
            print(
                f"[ERROR][{self.__class__.__name__} Base] FAILED load model: {self.model_full_path}")  # Consider raising error

    def _load_model(self):
        """Loads the FastSAM model from the specified path."""
        try:
            if os.path.exists(self.model_full_path):
                model = FastSAM(self.model_full_path)  # Semicolon removed
                return model
            else:
                print(f"[Error] Model file not found: {self.model_full_path}")  # Semicolon removed
                return None
        except NameError:
            print("[Error] FastSAM class not found.")  # Semicolon removed
            return None
        except Exception as e:
            print(f"[Error] FastSAM load exception: {e}")  # Semicolon removed
            traceback.print_exc()  # Semicolon removed
            return None

    def run_inference(self, image_path_or_array) -> List[np.ndarray]:
        """
        Runs FastSAM inference on an image.

        Args:
            image_path_or_array: Path to the image file or a NumPy array representing the image.

        Returns:
            List[np.ndarray]: A list of segmentation masks as NumPy arrays (uint8), or an empty list on failure.
        """
        if self.model is None: print("[Error] Inference fail: Model not loaded."); return []
        try:
            results = self.model(image_path_or_array, device=self.device, retina_masks=True, imgsz=self.imgsz,
                                 conf=self.conf, iou=self.iou)
            # Check the structure of the results carefully
            if results and isinstance(results, list) and len(results) > 0 and hasattr(results[0], 'masks') and results[
                0].masks is not None and hasattr(results[0].masks, 'data'):
                masks_data = results[0].masks.data
                if masks_data is None: print("[Warning] Inference returned None for masks data."); return []
                # Convert to NumPy array
                masks_np = masks_data.cpu().numpy() if hasattr(masks_data, 'cpu') else np.array(masks_data)
                if not isinstance(masks_np, np.ndarray): print(
                    "[Warning] Failed to convert masks to NumPy array."); return []

                # Process masks based on dimensions
                processed_masks = []
                if masks_np.ndim == 3:  # Batch of masks [N, H, W]
                    for mask in masks_np:
                        if mask.ndim == 2:  # Ensure individual mask is 2D
                            processed_masks.append(mask.astype(np.uint8))
                        else:
                            print(f"[Warning] Skipping mask with unexpected dimensions: {mask.ndim}")
                elif masks_np.ndim == 2:  # Single mask [H, W]
                    processed_masks.append(masks_np.astype(np.uint8))
                else:
                    print(f"[Warning] Unexpected mask dimensions from model: {masks_np.ndim}")
                return processed_masks
            else:
                # Handle cases where the expected structure isn't found
                print("[Warning] No valid masks found in inference results structure.")  # Semicolon removed
                return []
        except Exception as e:
            print(f"[Error] Inference exception: {e}\n{traceback.format_exc()}")  # Semicolon removed
            return []


# --- End Base Class ---


# --- Combined Bezel/PWB Segmenter (UPDATED SECTIONS) ---
class BezelPWBPositionSegmenter(BaseFastSamSegmenter):
    """
    Segmenter specifically for Bezel/PWB position checking.
    Uses FastSAM and RotationInvariantAOIChecker.
    Performs a two-stage check: Bezel position first, then optional PWB position.
    Evaluates using counts defined in the loaded configuration.
    Allows configuration of edge threshold for evaluation.
    Includes logic to avoid overlapping numbered circles in visualizations.
    Allows choosing between axis-aligned and min-rotated bounding box drawing.
    Uses different colors for bounding boxes based on object type.
    Provides options to toggle detailed visualizations (table, distance, relative pos).
    Draws text labels with backgrounds for better visibility.
    Separates drawing computation time from display/wait time.
    Includes PWB measurement visualization.
    """

    def __init__(self, model_type="x", model_path="ai-models/", output_dir=r"C:\BoardDefectChecker\ai-outputs"):
        """Initializes the BezelPWBPositionSegmenter."""
        super().__init__(model_type=model_type, model_path=model_path, imgsz=896, conf=0.2, iou=0.9)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # --- CONFIGURATION STRING (NEEDS MANUAL UPDATE from learning tool) ---
        # *** USER ACTION REQUIRED: Manually add the "internal_geometry_checks" section below ***
        # Example structure:
        # "internal_geometry_checks": {
        #   "pwb_check": {
        #     "target_object_type": "stamped_mark",
        #     "binarization_threshold": 128,
        #     "morph_kernel_size": [3, 3],
        #     "distance_range_mm": [1.5, 2.5],
        #     "pixel_size_mm": 0.00345 # Added pixel size here for clarity
        #   }
        # }
        self.rotation_invariant_checking_config = '''
                 {
                   "target_objects": {
                     "bezel": {
                       "expected_evaluation_count": 1,
                       "total_samples_labeled": 190,
                       "feature_ranges": {
                         "area": [8809, 43065],
                         "aspect_ratio": [0.119, 0.281],
                         "larger_dim": [306.286, 677.683],
                         "smaller_dim": [39.846, 160.561],
                         "perimeter": [668.071, 1832.17]
                       }
                     },
                     "copper_mark": {
                       "expected_evaluation_count": 2,
                       "total_samples_labeled": 279,
                       "feature_ranges": {
                         "area": [1571.25, 3438.75],
                         "aspect_ratio": [0.187, 0.331],
                         "larger_dim": [99.805, 118.609],
                         "smaller_dim": [19.673, 36.411],
                         "perimeter": [233.518, 301.283]
                       }
                     },
                     "stamped_mark": {
                       "expected_evaluation_count": 1,
                       "total_samples_labeled": 162,
                       "feature_ranges": {
                         "area": [2903.6, 8375.4],
                         "aspect_ratio": [0.323, 0.788],
                         "larger_dim": [91.8, 124.2],
                         "smaller_dim": [33.52, 87.48],
                         "perimeter": [246.678, 478.76]
                       }
                     }
                   },
                   "distance_constraints": {
                     "bezel-copper_mark": {"range": [28.56, 326.02], "mean": 150.825, "stddev": 49.539, "count": 337},
                     "bezel-stamped_mark": {"range": [484.646, 858.299], "mean": 675.628, "stddev": 40.174, "count": 190},
                     "copper_mark-stamped_mark": {"range": [583.675, 1021.73], "mean": 789.546, "stddev": 74.239, "count": 279}
                   },
                   "relative_position_constraints": {
                     "bezel-copper_mark": {"dx_range": [-272.308, 229.892], "dy_range": [-385.202, 205.832], "mean_dx": -21.208, "stddev_dx": 83.7, "mean_dy": -89.685, "stddev_dy": 98.506, "count": 337},
                     "bezel-stamped_mark": {"dx_range": [-1231.696, 1229.154], "dy_range": [-913.083, 1592.667], "mean_dx": -1.271, "stddev_dx": 410.142, "mean_dy": 339.792, "stddev_dy": 417.625, "count": 190},
                     "copper_mark-stamped_mark": {"dx_range": [-1586.16, 1175.537], "dy_range": [-1368.152, 1916.537], "mean_dx": -205.311, "stddev_dx": 460.283, "mean_dy": 274.192, "stddev_dy": 547.448, "count": 279},
                     "copper_mark-copper_mark": {"dx_range": [-171.428, 417.941], "dy_range": [-213.522, 144.613], "mean_dx": 123.256, "stddev_dx": 98.228, "mean_dy": -34.454, "stddev_dy": 59.689, "count": 117},
                     "bezel-bezel": {"dx_range": [-21.59, 20.046], "dy_range": [-148.591, 59.343], "mean_dx": -0.772, "stddev_dx": 6.939, "mean_dy": -44.624, "stddev_dy": 34.656, "count": 32}
                   },
                   "overlap_rules": [
                     {"objects": ["copper_mark", "stamped_mark"], "mode": "absolute"},
                     {"objects": ["bezel", "stamped_mark"], "mode": "absolute"},
                     {"objects": ["bezel", "copper_mark"], "mode": "absolute"},
                     {"objects": ["bezel", "bezel"], "mode": "absolute"},
                     {"objects": ["copper_mark", "copper_mark"], "mode": "absolute"},
                     {"objects": ["stamped_mark", "stamped_mark"], "mode": "absolute"}
                   ],
                   "internal_geometry_checks": {
                     "pwb_check": {
                       "target_object_type": "stamped_mark",
                       "binarization_threshold": 128,
                       "binary_inverted": false,
                       "erode_iterations": 0,
                       "dilate_iterations": 0,
                       "morph_kernel_size": [3, 3],
                       "enable_smoothing": false,
                       "column_edge_ignore_ratio": 0.15,
                       "min_column_height_ratio": 0.5,
                       "min_distance_mm": 0.6,
                       "max_distance_mm": 1.4,
                       "calibration_ref_height_mm": 0.425,
                       "calibration_ref_pixels": 30,
                       "additional_height_pixels": 10
                     }
                   }
                 }
                 '''
        # --- End CONFIGURATION STRING ---

        config_dict = {}
        try:
            config_dict = json.loads(self.rotation_invariant_checking_config)
            print(f"[{self.__class__.__name__}] Parsed configuration JSON.")
        except json.JSONDecodeError as e:
            print(f"[ERROR][{self.__class__.__name__}] Failed parse config JSON: {e}. Using empty default.")
            config_dict = {"target_objects": {}}
        except Exception as e_cfg:
            print(f"[ERROR][{self.__class__.__name__}] Unexpected error loading config: {e_cfg}. Using empty default.")
            config_dict = {"target_objects": {}}

        # Initialize the checker
        try:
            if 'RotationInvariantAOIChecker' in globals() and not hasattr(RotationInvariantAOIChecker, 'pass'):
                # Pass the full config_dict, the checker should parse internal_geometry_checks if needed
                self.rotation_invariant_checker = RotationInvariantAOIChecker(config_dict)
                print(f"[{self.__class__.__name__}] Checker initialized.")
            else:
                print("[ERROR] RotationInvariantAOIChecker class not available or is a dummy.")
                self.rotation_invariant_checker = None  # Set to None if dummy
        except Exception as e:
            print(f"[FATAL] Failed init checker: {e}")
            self.rotation_invariant_checker = None  # Set to None on error

        # --- Font Configuration ---
        self.font_path = "C:/Windows/Fonts/segoeui.ttf"
        self.font_size_large = 18
        self.font_size_small = 14
        self.font_large = None
        self.font_small = None
        global _pillow_available_global  # Assuming this global flag exists
        if _pillow_available_global:
            try:
                from PIL import ImageFont  # Import locally if needed
                self.font_large = ImageFont.truetype(self.font_path, self.font_size_large)
                self.font_small = ImageFont.truetype(self.font_path, self.font_size_small)
                print(f"[Info] Loaded font: {self.font_path}")
            except Exception as e:
                print(f"[{self.__class__.__name__}] Warn: Font load error: {e}.")
                self.font_large = None
                self.font_small = None
        else:
            print(f"[{self.__class__.__name__}] Info: Pillow not available.")
        # --- End Font Configuration ---

        print(f"[{self.__class__.__name__}] Initialized. Output Dir: {self.output_dir}")
        # --- End Initialization ---

    def load_blank_image(self, target_shape):
        """Loads a blank background image."""
        blank_image_path = r"C:\BoardDefectChecker\resources\blank.png"
        blank_image = cv2.imread(blank_image_path)
        target_h, target_w = target_shape[0], target_shape[1]
        if blank_image is None:
            print(f"[Error] Failed load blank image. Returning black.")
            target_shape_3ch = (target_h, target_w, 3) if len(target_shape) == 2 else target_shape
            return np.zeros(target_shape_3ch, dtype=np.uint8)
        try:
            return cv2.resize(blank_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        except Exception as resize_err:
            print(f"[Error] Failed resize blank image: {resize_err}")
            target_shape_3ch = (target_h, target_w, 3) if len(target_shape) == 2 else target_shape
            return np.zeros(target_shape_3ch, dtype=np.uint8)

    def visualize_masks(self, image, masks, **kwargs):
        """Visualizes masks by overlaying them with random colors."""
        if image is None: return None
        result_image = image.copy()
        processed_masks = [m for m in masks if isinstance(m, np.ndarray) and m.ndim == 2]
        if not processed_masks: return result_image

        visualize_all_masks = kwargs.get('visualize_all_masks', False)

        if visualize_all_masks:
            for mask_area_uint8 in processed_masks:
                mask_area = mask_area_uint8.astype(bool)
                color = np.random.randint(0, 256, size=3, dtype=np.uint8)
                try:
                    overlay = np.zeros_like(result_image)
                    overlay[mask_area] = color
                    result_image = cv2.addWeighted(result_image, 1.0, overlay, 0.5, 0)
                except (IndexError, ValueError) as e:
                    print(f"[Warning] Skipping mask overlay: {e}")

        return result_image

    def _find_non_overlapping_position(self, center_x, center_y, radius, placed_circles, max_attempts=8,
                                       step_scale=1.5):
        """Helper to find non-overlapping position for drawing circles."""
        current_pos = (center_x, center_y)
        # Check initial position
        is_overlapping = any(
            (current_pos[0] - px) ** 2 + (current_pos[1] - py) ** 2 < (radius + pr) ** 2
            for px, py, pr in placed_circles
        )
        if not is_overlapping:
            return current_pos

        # Try offsets if initial position overlaps
        step = int(radius * step_scale)
        # Spiral-like offsets outwards
        offsets = [(0, -step), (step, -step), (step, 0), (step, step), (0, step), (-step, step), (-step, 0),
                   (-step, -step)]

        for attempt in range(max_attempts):
            dx, dy = offsets[attempt % len(offsets)]
            next_x = center_x + dx
            next_y = center_y + dy
            is_overlapping = any(
                (next_x - px) ** 2 + (next_y - py) ** 2 < (radius + pr) ** 2
                for px, py, pr in placed_circles
            )
            if not is_overlapping:
                return (next_x, next_y)

        # Return original position if no non-overlapping spot found after attempts
        return (center_x, center_y)

    # --- draw_bounding_boxes METHOD (UPDATED SIGNATURE AND LOGIC) ---
    def draw_bounding_boxes(self, image: np.ndarray,
                            classified_masks: Dict[str, List[Dict]],
                            final_status: str,  # Overall status (OK/NG)
                            # --- New Parameter ---
                            pwb_measurement_info: Optional[Dict] = None,  # Info from PWB check
                            relative_position_pairs_to_check: Optional[List[str]] = None,
                            draw_param_table: bool = False,
                            draw_distance_lines: bool = False,
                            draw_relative_positions: bool = False,
                            draw_min_rect_bbox: bool = True
                            ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Draws bounding boxes (axis-aligned or min-rotated) with type-specific colors,
        object type text (with background), numbered centroids, final status (with background),
        and optionally: parameter table, distance lines, relative position vectors,
        and PWB measurement annotation.
        *** Does NOT display the image. ***
        """
        if image is None:
            print("[Error][draw_bounding_boxes] Input image is None.")
            return None, None
        try:
            height, width = image.shape[:2]
        except Exception as shape_err:
            print(f"[Error][draw_bounding_boxes] Cannot get shape: {shape_err}")
            return None, None
        mask_image = image.copy()
        table_image_np = None
        if draw_param_table:
            table_image_np = np.full((height, width, 3), (150, 150, 150), dtype=np.uint8)

        BBOX_COLOR_MAP = {
            "bezel": (255, 0, 0),  # Blue
            "copper_mark": (0, 255, 0),  # Green
            "stamped_mark": (0, 0, 255),  # Red
            "Unknown": (0, 255, 255)  # Yellow for fallback
        }
        grey_color_bgr = (150, 150, 150)
        black_color_bgr = (0, 0, 0)
        black_color_rgb = (0, 0, 0)  # For Pillow

        # Table Setup (only if drawing table)
        table_image_pil = None
        table_draw = None
        pillow_ready = False
        if draw_param_table:
            block_width = 200
            block_height = 150
            line_spacing_pixels = 22
            margin = 20
            cell_margin = 5
            max_display_masks = 30
            global _pillow_available_global
            if _pillow_available_global and hasattr(self, 'font_large') and hasattr(self,
                                                                                    'font_small') and self.font_large and self.font_small:
                try:
                    from PIL import Image, ImageDraw  # Re-import locally if needed
                    table_image_pil = Image.fromarray(cv2.cvtColor(table_image_np, cv2.COLOR_BGR2RGB))
                    table_draw = ImageDraw.Draw(table_image_pil)
                    pillow_ready = True
                except Exception as pil_e:
                    print(f"[Warning] Pillow table setup failed: {pil_e}.")
                    pillow_ready = False

        # Prepare Data
        displayed_blocks = 0
        all_mask_info_with_type = []
        object_info_map = {}
        if classified_masks and isinstance(classified_masks, dict):
            for obj_type, masks_list in classified_masks.items():
                object_info_map[obj_type] = []
                if not isinstance(masks_list, list): continue
                for mask_info in masks_list:
                    if draw_param_table and displayed_blocks >= max_display_masks: break
                    if isinstance(mask_info, dict) and 'features' in mask_info:
                        mask_info_copy = mask_info.copy()
                        mask_info_copy["type"] = obj_type
                        all_mask_info_with_type.append(mask_info_copy)
                        if draw_param_table: displayed_blocks += 1
                        features = mask_info['features']
                        cx = features.get("centroid_x")
                        cy = features.get("centroid_y")
                        angle = features.get("angle")
                        if cx is not None and cy is not None:
                            object_info_map[obj_type].append(
                                {'centroid': (int(cx), int(cy)), 'angle': angle, 'features': features})
                if draw_param_table and displayed_blocks >= max_display_masks: break

        # Draw BBoxes/MinRects, Object Type Text, Numbers, Table Cells
        placed_circles_info: List[Tuple[int, int, int]] = []
        circle_radius = 15
        blocks_per_column = max(1,
                                (height - 2 * margin) // block_height) if draw_param_table and block_height > 0 else 1
        text_color_bgr = (255, 255, 0)
        text_bg_color_bgr = (0, 0, 0)
        text_font_scale = 0.5
        text_thickness = 1
        text_padding = 2

        # --- Store info needed for PWB annotation ---
        pwb_target_draw_info = None

        for idx, mask_data in enumerate(all_mask_info_with_type, 1):
            mask = mask_data.get("mask")
            obj_type = mask_data.get("type", "Unknown")
            features = mask_data.get("features")
            if mask is None or features is None: continue
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0: continue

            bbox_color = BBOX_COLOR_MAP.get(obj_type, BBOX_COLOR_MAP["Unknown"])

            try:  # Draw BBox/MinRect, Type Text, and Circle
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                text_anchor_x, text_anchor_y = x1, y1
                min_rect_vertices = features.get("min_rect_vertices")

                # --- Draw Bounding Box OR Min Rotated Rect ---
                if draw_min_rect_bbox and min_rect_vertices is not None:
                    points = np.array(min_rect_vertices, dtype=np.int32)
                    if len(points) == 4:
                        cv2.polylines(mask_image, [points], isClosed=True, color=bbox_color, thickness=2)
                        top_vertex_index = np.argmin(points[:, 1])
                        text_anchor_x = points[top_vertex_index, 0]
                        text_anchor_y = points[top_vertex_index, 1]
                    else:
                        cv2.rectangle(mask_image, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)),
                                      bbox_color, 2)
                else:
                    cv2.rectangle(mask_image, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)),
                                  bbox_color, 2)

                # --- Draw Object Type Text ---
                type_text = obj_type
                (tw_type, th_type), baseline_type = cv2.getTextSize(type_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                    text_font_scale, text_thickness)
                text_x_type = max(0, text_anchor_x)
                text_y_type = max(th_type + text_padding, text_anchor_y - text_padding - baseline_type)
                cv2.rectangle(mask_image, (text_x_type - text_padding, text_y_type - th_type - text_padding),
                              (text_x_type + tw_type + text_padding, text_y_type + baseline_type + text_padding),
                              text_bg_color_bgr, -1)
                cv2.putText(mask_image, type_text, (text_x_type, text_y_type), cv2.FONT_HERSHEY_SIMPLEX,
                            text_font_scale, text_color_bgr, text_thickness, cv2.LINE_AA)

                # --- Draw Numbered Circle ---
                initial_centroid_x = int(features.get("centroid_x", x1 + (x2 - x1) // 2))
                initial_centroid_y = int(features.get("centroid_y", y1 + (y2 - y1) // 2))
                final_cx, final_cy = self._find_non_overlapping_position(initial_centroid_x, initial_centroid_y,
                                                                         circle_radius, placed_circles_info)
                placed_circles_info.append((final_cx, final_cy, circle_radius))
                final_cx = max(0, min(width - 1, final_cx))
                final_cy = max(0, min(height - 1, final_cy))
                cv2.circle(mask_image, (final_cx, final_cy), circle_radius, grey_color_bgr, -1)
                cv2.circle(mask_image, (final_cx, final_cy), circle_radius, black_color_bgr, 1)
                number_text = str(idx)
                (tw_num, th_num), _ = cv2.getTextSize(number_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.putText(mask_image, number_text, (final_cx - tw_num // 2, final_cy + th_num // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color_bgr, 1, cv2.LINE_AA)

                # --- Store info for PWB annotation ---
                if pwb_measurement_info and obj_type == pwb_measurement_info.get("target_object_type"):
                    pwb_target_draw_info = {"centroid_x": int(features.get("centroid_x", x1 + (x2 - x1) // 2)),
                                            "bbox_y2": y2, "bbox_x1": x1, "bbox_x2": x2}

            except Exception as mask_draw_err:
                print(f"[Warning] Error drawing bbox/circle for mask #{idx}: {mask_draw_err}")
                continue

            # --- Draw Table Cell ---
            if draw_param_table and idx <= max_display_masks:
                try:
                    col_idx = (idx - 1) // blocks_per_column
                    row_idx = (idx - 1) % blocks_per_column
                    block_x = margin + col_idx * block_width
                    block_y = margin + row_idx * block_height
                    if block_x + block_width > width or block_y + block_height > height: continue
                    text_lines = [f"{idx}: {obj_type}", f" Area: {features.get('area', 'N/A'):.0f}",
                                  f" AR: {features.get('aspect_ratio', 'N/A'):.2f}",
                                  f" LDim: {features.get('larger_dim', 'N/A'):.1f}",
                                  f" SDim: {features.get('smaller_dim', 'N/A'):.1f}",
                                  f" Perim: {features.get('perimeter', 'N/A'):.1f}",
                                  f" Angle: {features.get('angle', 'N/A'):.1f}"]
                    if pillow_ready and table_draw is not None and self.font_large and self.font_small:
                        current_y = block_y + cell_margin
                        table_draw.text((block_x + cell_margin, current_y), text_lines[0], font=self.font_large,
                                        fill=black_color_rgb)
                        current_y += line_spacing_pixels
                        for line in text_lines[1:]:
                            table_draw.text((block_x + cell_margin, current_y), line, font=self.font_small,
                                            fill=black_color_rgb)
                            current_y += line_spacing_pixels
                    else:  # OpenCV Fallback
                        current_y = block_y + cell_margin + 18
                        cv2.putText(table_image_np, text_lines[0], (block_x + cell_margin, current_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color_bgr, 1, cv2.LINE_AA)
                        current_y += line_spacing_pixels
                        for line in text_lines[1:]:
                            cv2.putText(table_image_np, line, (block_x + cell_margin, current_y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, black_color_bgr, 1, cv2.LINE_AA)
                            current_y += line_spacing_pixels
                except Exception as table_draw_err:
                    print(f"[Error Table] Failed draw cell #{idx}: {table_draw_err}")
                    traceback.print_exc()

        # Draw Distance Lines (Only if enabled)
        if draw_distance_lines:
            print("[Info] Drawing distance constraints...")
            distance_line_color = (255, 255, 0)
            distance_text_color = (255, 255, 0)
            distance_font_scale = 0.5
            distance_font_thickness = 1
            dist_constraints = {}
            drawn_dist_count = 0
            if hasattr(self, 'rotation_invariant_checker') and hasattr(self.rotation_invariant_checker,
                                                                       'distance_constraints'):
                dist_constraints = self.rotation_invariant_checker.distance_constraints
            for pair_key, constraint_data in dist_constraints.items():
                try:
                    obj_types = pair_key.split('-')
                    if len(obj_types) != 2: continue
                    type1, type2 = obj_types[0], obj_types[1]
                    infos1 = object_info_map.get(type1, [])
                    infos2 = object_info_map.get(type2, [])
                    if not infos1 or not infos2: continue
                    for info1 in infos1:
                        c1 = info1['centroid']
                        for info2 in infos2:
                            if info1 is info2: continue
                            c2 = info2['centroid']
                            dist = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2)
                            dist_text = f"{dist:.1f}"
                            cv2.line(mask_image, c1, c2, distance_line_color, distance_font_thickness)
                            mid_x = (c1[0] + c2[0]) // 2
                            mid_y = (c1[1] + c2[1]) // 2
                            (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, distance_font_scale,
                                                          distance_font_thickness)
                            text_x = mid_x - tw // 2
                            text_y = mid_y - th // 2 - 5
                            cv2.putText(mask_image, dist_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                                        distance_font_scale, distance_text_color, distance_font_thickness, cv2.LINE_AA)
                            drawn_dist_count += 1
                except Exception as dist_draw_err:
                    print(f"[Warning] Error drawing distance for '{pair_key}': {dist_draw_err}")
            print(f"[Debug] Drawn {drawn_dist_count} distance lines.")
        else:
            print("[Info] Drawing distance lines disabled.")

        # Draw Relative Position Visualization (Only if enabled)
        if draw_relative_positions:
            print("[Info] Drawing relative position constraints...")
            relpos_constraints = {}
            drawn_relpos_count = 0
            if hasattr(self, 'rotation_invariant_checker') and hasattr(self.rotation_invariant_checker,
                                                                       'relative_position_constraints'):
                relpos_constraints = self.rotation_invariant_checker.relative_position_constraints
            visualize_pairs_set = None
            if relative_position_pairs_to_check is not None:
                visualize_pairs_set = set("-".join(sorted(p.split('-'))) for p in relative_position_pairs_to_check)
                print(f"  - Only visualizing relative position for pairs: {visualize_pairs_set}")
            else:
                print("  - Visualizing all defined relative position pairs.")
            relpos_orig_color = (0, 165, 255)
            relpos_rot_color = (255, 0, 255)
            relpos_axis_color = (100, 100, 100)
            relpos_text_color = (255, 0, 255)
            relpos_font_scale = 0.4
            relpos_thickness = 1
            axis_length = 30
            for pair_key, constraint_data in relpos_constraints.items():
                try:
                    obj_types = pair_key.split('-')
                    if len(obj_types) != 2: continue
                    sorted_pair_key = "-".join(sorted(obj_types))
                    should_skip = visualize_pairs_set is not None and sorted_pair_key not in visualize_pairs_set
                    if should_skip: continue
                    type1, type2 = obj_types[0], obj_types[1]
                    infos1 = object_info_map.get(type1, [])
                    infos2 = object_info_map.get(type2, [])
                    if not infos1 or not infos2: continue
                    for info1 in infos1:
                        c1 = info1['centroid']
                        a1_deg = info1.get('angle')
                        if a1_deg is None: continue
                        a1_rad = math.radians(a1_deg)
                        cos_neg_a1 = math.cos(-a1_rad)
                        sin_neg_a1 = math.sin(-a1_rad)
                        cos_pos_a1 = math.cos(a1_rad)
                        sin_pos_a1 = math.sin(a1_rad)
                        x_axis_end_x = c1[0] + int(axis_length * cos_pos_a1)
                        x_axis_end_y = c1[1] + int(axis_length * sin_pos_a1)
                        cv2.line(mask_image, c1, (x_axis_end_x, x_axis_end_y), relpos_axis_color, relpos_thickness)
                        cv2.putText(mask_image, "dx'", (x_axis_end_x + 5, x_axis_end_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    relpos_font_scale * 0.8, relpos_axis_color, 1)
                        y_axis_end_x = c1[0] - int(axis_length * sin_pos_a1)
                        y_axis_end_y = c1[1] + int(axis_length * cos_pos_a1)
                        cv2.line(mask_image, c1, (y_axis_end_x, y_axis_end_y), relpos_axis_color, relpos_thickness)
                        cv2.putText(mask_image, "dy'", (y_axis_end_x + 5, y_axis_end_y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    relpos_font_scale * 0.8, relpos_axis_color, 1)
                        for info2 in infos2:
                            if info1 is info2: continue
                            c2 = info2['centroid']
                            cv2.arrowedLine(mask_image, c1, c2, relpos_orig_color, relpos_thickness, tipLength=0.05)
                            dx = c2[0] - c1[0]
                            dy = c2[1] - c1[1]
                            dx_rel = dx * cos_neg_a1 - dy * sin_neg_a1
                            dy_rel = dx * sin_neg_a1 + dy * cos_neg_a1
                            end_x_img = c1[0] + int(dx_rel * cos_pos_a1 - dy_rel * sin_pos_a1)
                            end_y_img = c1[1] + int(dx_rel * sin_pos_a1 + dy_rel * cos_pos_a1)
                            cv2.arrowedLine(mask_image, c1, (end_x_img, end_y_img), relpos_rot_color,
                                            relpos_thickness + 1, tipLength=0.07)
                            relpos_text = f"({dx_rel:.1f},{dy_rel:.1f})"
                            (tw_rel, th_rel), _ = cv2.getTextSize(relpos_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                  relpos_font_scale, relpos_thickness)
                            text_x_rel = end_x_img + 5
                            text_y_rel = end_y_img - 5
                            cv2.rectangle(mask_image, (text_x_rel - 2, text_y_rel - th_rel - 2),
                                          (text_x_rel + tw_rel + 2, text_y_rel + 2), (255, 255, 255), -1)
                            cv2.putText(mask_image, relpos_text, (text_x_rel, text_y_rel), cv2.FONT_HERSHEY_SIMPLEX,
                                        relpos_font_scale, relpos_text_color, relpos_thickness, cv2.LINE_AA)
                            drawn_relpos_count += 1
                except Exception as relpos_draw_err:
                    print(f"[Warning] Error drawing relative position for '{pair_key}': {relpos_draw_err}")
            print(f"[Debug] Drawn {drawn_relpos_count} relative position visualizations (filtered by input parameter).")
        else:
            print("[Info] Drawing relative positions disabled.")

        # --- Draw PWB Measurement Annotation (NEW) ---
        if pwb_measurement_info and pwb_target_draw_info:
            print("[Info] Drawing PWB measurement annotation...")
            try:
                pixel_dist = pwb_measurement_info.get("distance_pixels")
                mm_dist = pwb_measurement_info.get("distance_mm")
                target_centroid_x = pwb_target_draw_info["centroid_x"]
                target_bbox_y2 = pwb_target_draw_info["bbox_y2"]
                # target_bbox_x1 = pwb_target_draw_info["bbox_x1"] # Not needed for current drawing

                if pixel_dist is not None and mm_dist is not None:
                    arrow_start_y = target_bbox_y2
                    arrow_end_y = target_bbox_y2 - int(pixel_dist)
                    arrow_x = target_centroid_x
                    arrow_start_y = max(0, min(height - 1, arrow_start_y))
                    arrow_end_y = max(0, min(height - 1, arrow_end_y))
                    arrow_x = max(0, min(width - 1, arrow_x))
                    arrow_color = (0, 255, 255)  # Yellow
                    arrow_thickness = 2
                    cv2.arrowedLine(mask_image, (arrow_x, arrow_start_y), (arrow_x, arrow_end_y), arrow_color,
                                    arrow_thickness, tipLength=0.05)
                    cv2.arrowedLine(mask_image, (arrow_x, arrow_end_y), (arrow_x, arrow_start_y), arrow_color,
                                    arrow_thickness, tipLength=0.05)

                    # Draw text label
                    pwb_text = f"{mm_dist:.2f}mm"
                    pwb_text_color = arrow_color
                    pwb_text_bg_color = (0, 0, 0)  # Black background
                    pwb_font_scale = 0.5
                    pwb_thickness = 1
                    pwb_padding = 2

                    (tw_pwb, th_pwb), baseline_pwb = cv2.getTextSize(pwb_text, cv2.FONT_HERSHEY_SIMPLEX, pwb_font_scale,
                                                                     pwb_thickness)
                    # Position text next to the arrow (e.g., to the left)
                    text_x_pwb = max(pwb_padding, arrow_x - tw_pwb - 10 - pwb_padding)  # Position left of arrow
                    text_y_pwb = (arrow_start_y + arrow_end_y) // 2 + th_pwb // 2  # Vertically centered on arrow

                    # Draw background
                    cv2.rectangle(mask_image,
                                  (text_x_pwb - pwb_padding, text_y_pwb - th_pwb - pwb_padding),
                                  (text_x_pwb + tw_pwb + pwb_padding, text_y_pwb + baseline_pwb + pwb_padding),
                                  pwb_text_bg_color, -1)
                    # Draw text
                    cv2.putText(mask_image, pwb_text, (text_x_pwb, text_y_pwb), cv2.FONT_HERSHEY_SIMPLEX,
                                pwb_font_scale, pwb_text_color, pwb_thickness, cv2.LINE_AA)
                else:
                    print("[Warning] PWB measurement data missing (pixel_dist or mm_dist).")
            except Exception as pwb_draw_err:
                print(f"[Error] Failed to draw PWB measurement annotation: {pwb_draw_err}")
        # --- End PWB Measurement Annotation ---

        # --- Draw Final Status Text ---
        status_text = final_status
        status_color = (0, 255, 0) if final_status == "OK" else (0, 0, 255)  # GREEN for OK, RED for NG
        status_bg_color = (0, 0, 0)
        status_font_scale = 0.7
        status_thickness = 2
        status_padding = 5
        (tw_status, th_status), baseline_status = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX,
                                                                  status_font_scale, status_thickness)
        status_pos_x = status_padding
        status_pos_y = th_status + status_padding  # Position from top-left
        # Draw background rectangle
        cv2.rectangle(mask_image,
                      (status_pos_x - status_padding, status_pos_y - th_status - status_padding),
                      (status_pos_x + tw_status + status_padding, status_pos_y + baseline_status + status_padding),
                      status_bg_color, -1)
        # Draw text
        cv2.putText(mask_image, status_text, (status_pos_x, status_pos_y), cv2.FONT_HERSHEY_SIMPLEX, status_font_scale,
                    status_color, status_thickness, cv2.LINE_AA)
        # --- End Final Status Text ---

        # Finalize Table Image
        if draw_param_table and pillow_ready and table_image_pil is not None:
            try:
                table_image_np = cv2.cvtColor(np.array(table_image_pil), cv2.COLOR_RGB2BGR)
            except Exception as conv_e:
                print(f"[Error Table] PIL to OpenCV conversion failed: {conv_e}")

        return mask_image, table_image_np if draw_param_table else None

    # --- END draw_bounding_boxes METHOD ---

    # ... (draw_test_visualization method remains the same) ...

    # --- process_image METHOD (UPDATED SIGNATURE AND WORKFLOW) ---
    def process_image(self,
                      image_array: Optional[np.ndarray],
                      max_masks: Optional[int] = None,
                      test_mode: bool = False,
                      max_masks_to_show: int = 10,
                      edge_threshold: int = 5,
                      enable_mask_iou_filter: bool = True,
                      iou_threshold: float = 0.9,
                      enable_relative_position_check: bool = True,
                      relative_position_pairs_to_check: Optional[List[str]] = None,
                      enable_containment_check: bool = True,
                      containment_reference_type: str = "bezel",
                      containment_target_type: str = "stamped_mark",
                      enable_bbox_nms: bool = True,
                      bbox_nms_iou_threshold: float = 0.5,
                      bbox_nms_target_types: Optional[List[str]] = None,
                      sort_by_area: bool = True,
                      window_title: str = "Visualization",
                      is_image_shown: bool = False,
                      draw_param_table: bool = False,
                      draw_distance_lines: bool = False,
                      draw_relative_positions: bool = False,
                      draw_min_rect_bbox: bool = True,
                      # --- New Parameter ---
                      enable_pwb_check: bool = True  # Flag to enable the second stage check
                      ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, str, str]:
        """
        Processes image: inference, Bezel Check, optional PWB Check, visualize.
        Separates drawing computation time from display/wait time.
        """
        start_time = time.time()
        inf_time = 0.0
        eval_time_bezel = 0.0
        eval_time_pwb = 0.0
        draw_time_bezel = 0.0
        draw_time_final = 0.0
        # num_displayed_masks = 0 # Not used in eval mode
        mask_img, table_img = None, None
        final_status, final_reason = "NG", "Initialization Error"
        raw_masks = None
        masks = []
        blank_image = None
        bezel_classified_masks = {}  # Store results from Bezel check
        pwb_measurement_info = None  # Store results from PWB check

        # --- Input Validation ---
        if self.model is None:
            reason = "Model not initialized."
            print(f"[{self.__class__.__name__}] NG: {reason}")
            proc_time = time.time() - start_time
            blank_shape = (768, 1024, 3)
            blank_image = np.zeros(blank_shape, dtype=np.uint8)
            return blank_image.copy(), blank_image.copy(), proc_time, reason, reason
        if image_array is None or image_array.size == 0:
            reason = "Invalid input image."
            print(f"[{self.__class__.__name__}] NG: {reason}")
            proc_time = time.time() - start_time
            blank_shape = (768, 1024, 3)
            blank_image = np.zeros(blank_shape, dtype=np.uint8)
            return blank_image.copy(), blank_image.copy(), proc_time, reason, reason

        original_image_shape = image_array.shape[:2]
        try:
            blank_image = self.load_blank_image(original_image_shape)
        except AttributeError:
            blank_image = np.zeros((*original_image_shape, 3), dtype=np.uint8)
        mask_img, table_img = blank_image.copy(), blank_image.copy()

        try:
            # --- 1. Inference ---
            print(f"[{self.__class__.__name__}] Running inference...")
            inf_start_time = time.time()
            if not hasattr(self, 'run_inference'): raise AttributeError("'run_inference' method not found")
            raw_masks = self.run_inference(image_array)
            inf_time = time.time() - inf_start_time
            masks = [m for m in raw_masks if isinstance(m, np.ndarray) and m.ndim == 2] if raw_masks else []
            print(f"[{self.__class__.__name__}] Inference done ({len(masks)} valid masks): {inf_time:.3f}s.")

            filter_shape = None
            if masks:
                try:
                    filter_shape = masks[0].shape
                    print(f"[Info] Using mask shape for filtering: {filter_shape}")
                except IndexError:
                    print("[Warning] Masks list empty? Cannot get filter_shape.")
                except Exception as shape_err:
                    print(f"[Warning] Cannot get filter_shape: {shape_err}")
            else:
                print("[Warning] No valid masks from inference.")

            if self.rotation_invariant_checker is None:
                raise RuntimeError("RotationInvariantChecker failed to initialize. Cannot proceed.")

            # --- 2. Bezel Position Check (Stage 1) ---
            print(f"[{self.__class__.__name__}] Running Bezel Position Check (Stage 1)...")
            bezel_ok = False
            bezel_reason = "Bezel check not performed"
            if not masks or filter_shape is None:
                print("[Error] No valid masks or filter_shape for Bezel check.")
                bezel_reason = "No masks/filter_shape for Bezel check"
                bezel_classified_masks = {}
                eval_time_bezel = 0.0
            else:
                try:
                    t0_eval_bezel = time.time()
                    bezel_ok, bezel_reason, bezel_classified_masks = self.rotation_invariant_checker.evaluate(
                        masks=masks, filter_shape=filter_shape, edge_threshold=edge_threshold,
                        enable_mask_iou_filter=enable_mask_iou_filter,
                        iou_threshold=iou_threshold, enable_relative_position_check=enable_relative_position_check,
                        relative_position_pairs_to_check=relative_position_pairs_to_check,
                        enable_containment_check=enable_containment_check,
                        containment_reference_type=containment_reference_type,
                        containment_target_type=containment_target_type,
                        enable_bbox_nms=enable_bbox_nms, bbox_nms_iou_threshold=bbox_nms_iou_threshold,
                        bbox_nms_target_types=bbox_nms_target_types
                    )
                    eval_time_bezel = time.time() - t0_eval_bezel
                    print(
                        f"[{self.__class__.__name__}] Bezel Check Result: {'OK' if bezel_ok else 'NG'} ({bezel_reason})")
                    print(f"[TIME] Bezel Check Evaluate call: {eval_time_bezel:.4f}s")
                except Exception as eval_err:
                    print(f"[Error] Error during Bezel Check evaluation: {eval_err}\n{traceback.format_exc()}")
                    bezel_ok = False
                    bezel_reason = f"Bezel Check Error: {eval_err}"
                    bezel_classified_masks = {}
                    eval_time_bezel = 0.0

            # --- Intermediate Visualization (Optional) ---
            if is_image_shown:
                print(f"[{self.__class__.__name__}] Drawing intermediate Bezel Check visualization...")
                t0_draw_bezel = time.time()
                interim_status = "OK" if bezel_ok else "NG"
                try:
                    interim_mask_img, interim_table_img = self.draw_bounding_boxes(
                        image_array, bezel_classified_masks, final_status=interim_status, pwb_measurement_info=None,
                        relative_position_pairs_to_check=relative_position_pairs_to_check,
                        draw_param_table=draw_param_table,
                        draw_distance_lines=draw_distance_lines, draw_relative_positions=draw_relative_positions,
                        draw_min_rect_bbox=draw_min_rect_bbox
                    )
                    draw_time_bezel = time.time() - t0_draw_bezel
                    print(f"[TIME] Intermediate Bezel Draw call: {draw_time_bezel:.4f}s")
                    if interim_mask_img is not None:
                        display_width = 1280
                        display_height = 800
                        resized_interim_img = cv2.resize(interim_mask_img, (display_width, display_height),
                                                         interpolation=cv2.INTER_AREA)
                        cv2.imshow(f"{window_title} - Bezel Check Result ({interim_status})", resized_interim_img)
                        print(
                            ">>> Displaying Bezel Check Result. Press any key to continue to PWB Check (if applicable)...")
                        cv2.waitKey(0)
                        cv2.destroyWindow(f"{window_title} - Bezel Check Result ({interim_status})")
                    else:
                        print("[Warning] Intermediate mask image is None, cannot display.")
                except Exception as draw_err:
                    print(f"[Error] Failed to draw/display intermediate Bezel Check visualization: {draw_err}")
                    draw_time_bezel = time.time() - t0_draw_bezel

            # --- 3. PWB Position Check (Stage 2 - Conditional) ---
            final_status = "OK" if bezel_ok else "NG"
            final_reason = bezel_reason

            if bezel_ok and enable_pwb_check:
                print(f"[{self.__class__.__name__}] Running PWB Position Check (Stage 2)...")
                pwb_ok = False
                pwb_reason = "PWB check not performed"
                target_pwb_object = None
                try:
                    pwb_check_config = self.rotation_invariant_checker.config.get("internal_geometry_checks", {}).get(
                        "pwb_check")
                    if not pwb_check_config: raise ValueError(
                        "PWB check configuration ('internal_geometry_checks.pwb_check') not found in checker config.")
                    target_type = pwb_check_config.get("target_object_type")
                    if not target_type: raise ValueError("Missing 'target_object_type' in PWB check configuration.")

                    target_objects_list = bezel_classified_masks.get(target_type, [])
                    if len(target_objects_list) == 1:
                        target_pwb_object = target_objects_list[0]
                        print(f"[Info] Found target '{target_type}' for PWB check.")
                    elif len(target_objects_list) == 0:
                        pwb_reason = f"PWB_NG: Target object '{target_type}' not found after Bezel check."
                        print(f"[Warning] {pwb_reason}")
                        pwb_ok = False
                    else:
                        pwb_reason = f"PWB_NG: Expected 1 target '{target_type}', found {len(target_objects_list)} after Bezel check."
                        print(f"[Warning] {pwb_reason}")
                        pwb_ok = False

                    if target_pwb_object:
                        t0_eval_pwb = time.time()
                        pwb_ok, pwb_reason, measured_mm, measured_pixels = self.rotation_invariant_checker.check_internal_geometry(
                            target_object_info=target_pwb_object, original_image=image_array,
                            check_config=pwb_check_config, is_image_shown=is_image_shown
                        )
                        eval_time_pwb = time.time() - t0_eval_pwb
                        print(
                            f"[{self.__class__.__name__}] PWB Check Result: {'OK' if pwb_ok else 'NG'} ({pwb_reason})")
                        print(f"[TIME] PWB Check Evaluate call: {eval_time_pwb:.4f}s")
                        if pwb_ok or measured_pixels is not None:
                            pwb_measurement_info = {"target_object_type": target_type,
                                                    "distance_pixels": measured_pixels, "distance_mm": measured_mm,
                                                    "status": "OK" if pwb_ok else "NG"}

                except AttributeError as ae:
                    pwb_reason = f"PWB_NG: Checker missing 'check_internal_geometry' method? Error: {ae}"
                    print(f"[Error] {pwb_reason}")
                    pwb_ok = False
                    eval_time_pwb = 0.0
                except ValueError as ve:
                    pwb_reason = f"PWB_NG: Configuration error: {ve}"
                    print(f"[Error] {pwb_reason}")
                    pwb_ok = False
                    eval_time_pwb = 0.0
                except Exception as pwb_err:
                    print(f"[Error] Error during PWB Check: {pwb_err}\n{traceback.format_exc()}")
                    pwb_ok = False
                    pwb_reason = f"PWB Check Error: {pwb_err}"
                    eval_time_pwb = 0.0

                if not pwb_ok:
                    final_status = "NG"
                    final_reason = pwb_reason
                else:
                    final_status = "OK"
                    final_reason = "OK: Bezel and PWB checks passed."

            elif not bezel_ok:
                print(f"[{self.__class__.__name__}] Skipping PWB check because Bezel check failed.")
            elif not enable_pwb_check:
                print(f"[{self.__class__.__name__}] Skipping PWB check because it is disabled.")
            # --- End PWB Check ---

            # --- 4. Final Visualization ---
            print(f"[{self.__class__.__name__}] Drawing final visualization...")
            t0_draw_final = time.time()
            try:
                mask_img, table_img = self.draw_bounding_boxes(
                    image_array, bezel_classified_masks, final_status=final_status,
                    pwb_measurement_info=pwb_measurement_info,
                    relative_position_pairs_to_check=relative_position_pairs_to_check,
                    draw_param_table=draw_param_table,
                    draw_distance_lines=draw_distance_lines, draw_relative_positions=draw_relative_positions,
                    draw_min_rect_bbox=draw_min_rect_bbox
                )
                draw_time_final = time.time() - t0_draw_final
                print(f"[TIME] Final Draw call: {draw_time_final:.4f}s")
            except Exception as draw_err:
                print(f"[Error] Failed to draw final visualization: {draw_err}")
                draw_time_final = time.time() - t0_draw_final
            # --- End Final Visualization ---

        # --- Error Handling ---
        except Exception as e:
            proc_time = time.time() - start_time
            final_status = "NG"
            final_reason = f"Overall Processing Error: {e}"
            print(f"[{self.__class__.__name__}] NG: {final_reason}. Time: {proc_time:.3f}s")
            print(traceback.format_exc())
            if blank_image is None:
                blank_image = np.zeros((768, 1024, 3), dtype=np.uint8)
            mask_img, table_img = blank_image.copy(), blank_image.copy()
            if 'bezel_classified_masks' not in locals(): bezel_classified_masks = {}
            if 'masks' in locals() and masks is not None: del masks
            if 'raw_masks' in locals() and raw_masks is not None: del raw_masks
            if 'bezel_classified_masks' in locals(): del bezel_classified_masks
            gc.collect()
            return mask_img, table_img, proc_time, final_status, final_reason
        # --- End Error Handling ---

        # --- Final Summary and Display ---
        proc_time_before_display = time.time() - start_time
        print(f"[{self.__class__.__name__}] Final Result: {final_status}. Reason: {final_reason}.")
        print(
            f"[TIME SUMMARY (Computation)] Inference: {inf_time:.4f}s | Bezel Eval: {eval_time_bezel:.4f}s | PWB Eval: {eval_time_pwb:.4f}s | Bezel Draw: {draw_time_bezel:.4f}s | Final Draw: {draw_time_final:.4f}s | Total Comp: {proc_time_before_display:.4f}s")

        if is_image_shown:
            print(f"[{self.__class__.__name__}] Displaying final results (press any key)...")
            try:
                display_width = 1280
                display_height = 800
                if mask_img is not None:
                    resized_mask_img = cv2.resize(mask_img, (display_width, display_height),
                                                  interpolation=cv2.INTER_AREA)
                    cv2.imshow(f"{window_title} - Final Result ({final_status})", resized_mask_img)
                else:
                    print("[Warning] final mask_img is None, cannot display.")
                if table_img is not None and draw_param_table:
                    table_window_title = f"{window_title} - Parameters"
                    resized_table_img = cv2.resize(table_img, (display_width, display_height),
                                                   interpolation=cv2.INTER_AREA)
                    cv2.imshow(table_window_title, resized_table_img)
                else:
                    table_window_title = None
                cv2.waitKey(0)
                cv2.destroyWindow(f"{window_title} - Final Result ({final_status})")
                if table_window_title: cv2.destroyWindow(table_window_title)
            except Exception as display_err:
                print(f"[Error] Display failed during process_image: {display_err}")
                cv2.destroyAllWindows()
        # --- End Final Summary and Display ---

        # --- Final Cleanup and Return ---
        if 'masks' in locals(): del masks
        if 'raw_masks' in locals(): del raw_masks
        if 'bezel_classified_masks' in locals(): del bezel_classified_masks
        gc.collect()
        proc_time = time.time() - start_time
        final_mask_img = mask_img if isinstance(mask_img, np.ndarray) else blank_image.copy()
        final_table_img = table_img if isinstance(table_img, np.ndarray) else blank_image.copy()
        return final_mask_img, final_table_img, proc_time, final_status, final_reason
    # --- END process_image METHOD ---


# --- Function for Bezel/PWB Testing (Reverted to uppercase constants) ---
def bezel_pwb_classification_test_main():
    """Main function to run classification test, allows setting thresholds and checks."""
    print("--- Running BezelPWBPositionSegmenter Classification Test ---")
    # --- Project Path Setup ---
    try:
        PROJECT_ROOT_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if not os.path.basename(PROJECT_ROOT_ABS) == 'BoardDefectChecker':
            PROJECT_ROOT_ABS = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            if not os.path.basename(PROJECT_ROOT_ABS) == 'BoardDefectChecker':
                PROJECT_ROOT_ABS = r"D:\Working\BoardDefectChecker"  # Fallback
                print(f"[Warning] Could not reliably determine project root. Using fallback: {PROJECT_ROOT_ABS}")
    except NameError:
        PROJECT_ROOT_ABS = r"D:\Working\BoardDefectChecker"  # Fallback
        print(f"[Warning] __file__ not defined. Using fallback project root: {PROJECT_ROOT_ABS}")

    AI_MODEL_DIR = os.path.join(PROJECT_ROOT_ABS, "ai-models")
    ABSOLUTE_SAMPLE_IMAGE_DIR = r"C:\BoardDefectChecker\images\samples_for_learning"
    SAMPLE_IMAGE_DIR = ABSOLUTE_SAMPLE_IMAGE_DIR
    # --- End Project Path Setup ---

    # --- Configurable Thresholds & Settings (UPPERCASE for constants) ---
    TEST_EDGE_THRESHOLD = 50
    TEST_ENABLE_MASK_IOU_FILTER = False
    TEST_IOU_THRESHOLD = 0.9
    TEST_ENABLE_BBOX_NMS = True
    TEST_BBOX_NMS_IOU_THRESHOLD = 0.1
    TEST_BBOX_NMS_TARGET_TYPES = ["bezel", "copper_mark", "stamped_mark"]  # Corrected copper_mark
    TEST_ENABLE_RELPOS_CHECK = True
    TEST_RELPOS_PAIRS_TO_CHECK = ['bezel-stamped_mark']
    TEST_ENABLE_CONTAINMENT_CHECK = True
    TEST_CONTAINMENT_REFERENCE = "bezel"
    TEST_CONTAINMENT_TARGET = "stamped_mark"
    TEST_DRAW_PARAM_TABLE = False
    TEST_DRAW_DISTANCE_LINES = False
    TEST_DRAW_RELATIVE_POSITIONS = False
    TEST_DRAW_MIN_RECT_BBOX = True
    TEST_ENABLE_PWB_CHECK = True  # Enable PWB check for testing
    # --- End Thresholds & Settings ---

    # --- Initialization ---
    print(f"Project Root (determined): {PROJECT_ROOT_ABS}")
    print(f"AI Model Directory: {AI_MODEL_DIR}")
    print(f"Sample Image Directory: {SAMPLE_IMAGE_DIR}")
    try:
        if not os.path.isdir(AI_MODEL_DIR):
            print(f"[FATAL ERROR] AI Model directory not found: {AI_MODEL_DIR}")
            sys.exit(1)
        segmenter = BezelPWBPositionSegmenter(model_type="x", model_path=AI_MODEL_DIR)
        if segmenter.model is None:
            print("[FATAL ERROR] Segmenter model failed to load. Exiting.")
            sys.exit(1)
        if segmenter.rotation_invariant_checker is None:
            print("[FATAL ERROR] RotationInvariantChecker not initialized. Exiting.")
            sys.exit(1)
    except NameError:
        print("[FATAL ERROR] BezelPWBPositionSegmenter class not found.")
        sys.exit(1)
    except Exception as init_err:
        print(f"[FATAL ERROR] Failed to initialize Segmenter: {init_err}")
        traceback.print_exc()
        sys.exit(1)

    if not os.path.isdir(SAMPLE_IMAGE_DIR):
        print(f"[Error] Sample directory not found: {SAMPLE_IMAGE_DIR}")
        return
    valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")
    try:
        sample_files = [f for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(valid_extensions)]
        sample_image_paths = [os.path.join(SAMPLE_IMAGE_DIR, f) for f in sample_files]
    except Exception as e:
        print(f"[Error] Failed list images in {SAMPLE_IMAGE_DIR}: {e}")
        return
    if not sample_image_paths:
        print(f"[Warning] No valid images found in {SAMPLE_IMAGE_DIR}.")
        return
    # --- End Initialization ---

    # --- Log Settings ---
    print(f"\nFound {len(sample_image_paths)} images to process.")
    print(f"--- Using Edge Threshold: {TEST_EDGE_THRESHOLD} ---")
    print(f"--- Mask IoU Filter Enabled: {TEST_ENABLE_MASK_IOU_FILTER} (IoU Threshold: {TEST_IOU_THRESHOLD}) ---")
    print(
        f"--- BBox NMS Enabled: {TEST_ENABLE_BBOX_NMS} (IoU: {TEST_BBOX_NMS_IOU_THRESHOLD}, Targets: {'Default' if TEST_BBOX_NMS_TARGET_TYPES is None else TEST_BBOX_NMS_TARGET_TYPES}) ---")
    print(
        f"--- Containment Check Enabled: {TEST_ENABLE_CONTAINMENT_CHECK} ('{TEST_CONTAINMENT_TARGET}' in '{TEST_CONTAINMENT_REFERENCE}') ---")
    print(f"--- Relative Position Check Enabled: {TEST_ENABLE_RELPOS_CHECK} ---")
    print(
        f"--- Relative Position Pairs to Check: {'All Defined' if TEST_RELPOS_PAIRS_TO_CHECK is None else TEST_RELPOS_PAIRS_TO_CHECK} ---")
    print(f"--- Draw Parameter Table: {TEST_DRAW_PARAM_TABLE} ---")
    print(f"--- Draw Distance Lines: {TEST_DRAW_DISTANCE_LINES} ---")
    print(f"--- Draw Relative Positions: {TEST_DRAW_RELATIVE_POSITIONS} ---")
    print(f"--- Draw Min Rotated BBox: {TEST_DRAW_MIN_RECT_BBOX} ---")
    print(f"--- PWB Check Enabled: {TEST_ENABLE_PWB_CHECK} ---")  # Log new flag
    # --- End Log Settings ---

    # --- Processing Loop ---
    for img_path, img_filename in zip(sample_image_paths, sample_files):
        print(f"\n--- Processing image for evaluation: {img_filename} ---")
        image_array = cv2.imread(img_path)
        if image_array is None:
            print(f"[Error] Failed load image: {img_path}")
            continue
        try:
            # *** Pass all flags (UPPERCASE) to process_image ***
            mask_img, table_img, proc_time, status, reason = segmenter.process_image(
                image_array=image_array,
                test_mode=False,
                is_image_shown=True,  # Keep display ON for testing
                edge_threshold=TEST_EDGE_THRESHOLD,
                enable_mask_iou_filter=TEST_ENABLE_MASK_IOU_FILTER,
                iou_threshold=TEST_IOU_THRESHOLD,
                enable_relative_position_check=TEST_ENABLE_RELPOS_CHECK,
                relative_position_pairs_to_check=TEST_RELPOS_PAIRS_TO_CHECK,
                enable_containment_check=TEST_ENABLE_CONTAINMENT_CHECK,
                containment_reference_type=TEST_CONTAINMENT_REFERENCE,
                containment_target_type=TEST_CONTAINMENT_TARGET,
                enable_bbox_nms=TEST_ENABLE_BBOX_NMS,
                bbox_nms_iou_threshold=TEST_BBOX_NMS_IOU_THRESHOLD,
                bbox_nms_target_types=TEST_BBOX_NMS_TARGET_TYPES,
                draw_param_table=TEST_DRAW_PARAM_TABLE,
                draw_distance_lines=TEST_DRAW_DISTANCE_LINES,
                draw_relative_positions=TEST_DRAW_RELATIVE_POSITIONS,
                draw_min_rect_bbox=TEST_DRAW_MIN_RECT_BBOX,
                enable_pwb_check=TEST_ENABLE_PWB_CHECK  # Pass the new flag
            )
            # Result logging is handled within process_image now
        except Exception as eval_err:
            print(f"[FATAL ERROR] Error during process_image for {img_filename}: {eval_err}")
            traceback.print_exc()

    print("\n--- BezelPWBPositionSegmenter Evaluation Loop Finished ---")
    cv2.destroyAllWindows()


# --- End Function ---


class ModelManager:
    """Class to manage model files and configurations"""

    def __init__(self, models_dir="../ai-models"):
        self.models_dir = models_dir
        self.model_files = []
        self.model_list_file = '../model_list.txt'

        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)

    def scan_model_files(self):
        """
        Scan models directory and write list to model_list.txt
        Returns:
            list: Available model files
        """
        print("Scanning for model files...")
        self.model_files = []

        if os.path.exists(self.models_dir):
            for file in os.listdir(self.models_dir):
                if file.endswith(('.pt', '.pth')):
                    self.model_files.append(file)

        # Write to model_list.txt
        self._write_model_list()

        print(f"Found {len(self.model_files)} model files. List written to {self.model_list_file}")
        return self.model_files

    def _write_model_list(self):
        """Write available models to file"""
        with open(self.model_list_file, 'w') as f:
            f.write("Available model files:\n")
            for model in self.model_files:
                f.write(f"- {model}\n")

    def get_model_path(self, model_name):
        """Get full path for a model file"""
        return os.path.join(self.models_dir, model_name)

    def check_model_exists(self, model_name):
        """Check if a model file exists"""
        return os.path.exists(self.get_model_path(model_name))

    def get_available_models(self):
        """Get list of available models"""
        return self.model_files if self.model_files else self.scan_model_files()


def token_fpc_main():
    """Standalone function to process Token FPC images with defect checking and debug prints."""
    input_dir = r"C:\BoardDefectChecker\images\raw-images\token-fpc-17-03-2025"
    segmenter = TokenFPCFastImageSegmenter(
        model_type="x",
        model_path="../ai-models/",
        output_directory=r"C:\BoardDefectChecker\ai-outputs",
        is_image_shown=False
    )
    valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")

    print(f"[Debug] Starting token FPC main process for folder: {input_dir}")

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(valid_extensions):
            input_path = os.path.join(input_dir, filename)
            print(f"[Debug] Processing: {input_path}")

            # Process image with current return values
            visualized_image, bb_path, processing_time, detection_result, defect_reason = segmenter.process_image(
                input_path=input_path,
                visualize_all_masks=False,  # Optimized as requested
                label_color=(255, 0, 0)  # Red for bounding boxes
            )

            # Perform defect checks explicitly for debug output
            all_masks = [mask.numpy() for mask in
                         segmenter.model(input_path, device="cpu", retina_masks=True, imgsz=896, conf=0.4, iou=0.8)[
                             0].masks.data]
            filtered_masks = segmenter._filter_masks_in_areas(all_masks)
            connector_ok, connector_reason = segmenter.check_connector_lock(filtered_masks)
            fpc_ok, fpc_reason = segmenter.check_token_fpc_balance(filtered_masks)
            overall_ok = connector_ok and fpc_ok
            overall_reason = "All OK" if overall_ok else (connector_reason if not connector_ok else fpc_reason)

            # Debug prints
            print(f"[Debug] Connector check result: {connector_ok}, Reason: {connector_reason}")
            print(f"[Debug] FPC balance check result: {fpc_ok}, Reason: {fpc_reason}")
            print(f"[Debug] Overall result: {'OK' if overall_ok else 'NG'}, Reason: {overall_reason}")

            # Translate defect reason to Vietnamese
            if overall_ok:
                translated_reason = "Không có lỗi thao tác gắn TOKEN FPC"
            elif overall_reason:
                if ("Connector mask not found" in overall_reason or
                        "Connector mask not found for distance check" in overall_reason or
                        "Connector height" in overall_reason):
                    translated_reason = "Nắp khóa chưa đóng"
                elif ("Left or right FPC mask not found" in overall_reason or
                      "Unbalanced FPC" in overall_reason or
                      "Distance to connector exceeds" in overall_reason):
                    translated_reason = "Cáp FPC bị lệch hoặc chưa được gắn chặt"
                else:
                    translated_reason = overall_reason  # Fallback
            else:
                translated_reason = "Unknown defect"
            print(f"[Debug] Translated reason (VI): {translated_reason}")

            if visualized_image is not None:
                print(f"[Debug] Saved bounding box image to: {bb_path}")
                # Ensure processing_time is a float; strip units if present
                try:
                    processing_time_float = float(processing_time)
                except (ValueError, TypeError):
                    # If it's a string with units (e.g., "4.85 seconds"), extract the number
                    processing_time_float = float(str(processing_time).split()[0])
                print(f"[Debug] Processing time: {processing_time_float:.2f} seconds")
                print(f"[Debug] Image {filename} processed successfully\n")
            else:
                print(f"[Debug] Processing failed or no masks detected for {filename}\n")

            if segmenter.is_image_shown:
                cv2.destroyAllWindows()


def small_fpc_main():
    # Initialize model manager
    model_manager = ModelManager()
    model_manager.scan_model_files()

    # Configuration
    MODEL_TYPE = 'fast_x'  # 'fast_s', 'fast_x', 'sam_b', or 'sam_h'
    PROCESS_ALL_IMAGES = True

    # Create output directory
    os.makedirs("../ai-outputs", exist_ok=True)

    # Initialize appropriate segmenter based on model type
    if MODEL_TYPE.startswith('fast'):
        model_size = MODEL_TYPE.split('_')[1]  # get 'x' or 's'
        segmenter = SmallFPCFastImageSegmenter(model_type=model_size, angle_difference_threshold=0.7)
        prefix = "fast"
    elif MODEL_TYPE.startswith('sam'):
        model_size = MODEL_TYPE.split('_')[1]  # get 'b' or 'h'
        vit_type = f"vit_{model_size}"
        segmenter = SAMImageSegmenter(model_type=vit_type)
        prefix = f"sam_{model_size}"
    else:
        raise ValueError("Invalid MODEL_TYPE. Must be 'fast_s', 'fast_x', 'sam_b', or 'sam_h'")

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get list of images to process
    if PROCESS_ALL_IMAGES:
        # Get all image files from images directory
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        input_images = [
            os.path.join("../ai-test-images/dataset-7", f) for f in os.listdir("../ai-test-images/dataset-7")
            if f.lower().endswith(valid_extensions)
        ]
    else:
        # Use specific images
        input_images = [
            "../ai-test-images/Sample-7-04.png",
            "../ai-test-images/Sample-7-05.png"
        ]

    print(f"Found {len(input_images)} images to process")

    for image_path in input_images:
        # Create output path with timestamp
        original_image = cv2.imread(image_path)
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)

        segmenter.is_image_shown = False

        pin_count = 10

        # Process image and get outputs
        connector_output_path = os.path.join("../ai-outputs",
                                             f"{prefix}_connector_segmented_{base_name}_{timestamp}{ext}")
        # connector_result_image, connector_result_mask, connector_output_path, connector_process_time = segmenter.process_image(
        #     image_path,
        #     connector_output_path,
        #     pin_count,
        #     visualize_all_masks=False,
        #     input_image_type='connector')
        # segmenter.extract_connector_masked_image(original_image, result_mask[0], display_masked_image=True)

        # Check fpc size and balance
        fpc_output_path = os.path.join("../ai-outputs", f"{prefix}_fpc_segmented_{base_name}_{timestamp}{ext}")
        fpc_result_image, fpc_output_path, fpc_angle_difference, fpc_skeleton_image, fpc_skeleton_output_path = segmenter.process_image(
            image_path, fpc_output_path, pin_count, visualize_all_masks=True, input_image_type='fpc')
        # fpc_result_image, fpc_output_path, fpc_angle_difference, fpc_skeleton_image, fpc_skeleton_output_path, fpc_process_time = segmenter.process_image(
        #     image_path, fpc_output_path, pin_count, visualize_all_masks=True, input_image_type='fpc')

        # For 12 pins model
        if pin_count == 12:
            lower_width_threshold = 610
            higher_width_threshold = 860
            lower_height_threshold = 245
            higher_height_threshold = 290
        else:
            # For 10 pins model
            lower_width_threshold = 600
            higher_width_threshold = 700
            lower_height_threshold = 245
            higher_height_threshold = 270

        if fpc_angle_difference is not None:
            is_balanced, balance_output_path = segmenter.check_fpc_lead_balance(fpc_result_image, fpc_output_path,
                                                                                fpc_angle_difference,
                                                                                lower_width_threshold,
                                                                                higher_width_threshold,
                                                                                lower_height_threshold,
                                                                                higher_height_threshold,
                                                                                segmenter.current_result_mask)

    # Call the plotting method to visualize box size statistics
    # segmenter.plot_box_size_statistics()


# --- END REVERTED Test Function ---


# --- Main Execution Block ---
if __name__ == "__main__":
    # Choose which test function to run by uncommenting the desired line:

    bezel_pwb_classification_test_main()

    # print("\n--- Running Token FPC Main ---")
    # token_fpc_main() # Assumes token_fpc_main() function is defined elsewhere in the full file

    # print("\n--- Running Small FPC Main ---")
    # small_fpc_main() # Assumes small_fpc_main() function is defined elsewhere in the full file
