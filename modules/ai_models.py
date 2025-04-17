import os
import numpy as np
import matplotlib.pyplot as plt
import json
import math
import gc

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from fastsam import FastSAM
import cv2
from datetime import datetime
import time
from abc import ABC, abstractmethod
from modules.image_processing import ImageProcessor
from modules.token_fpc_image_processing import TokenFPCImageProcessor
# from modules.token_fpc_image_processing import TokenFPCImageProcessor # Not directly needed here
from modules.rotation_invariant_checking import RotationInvariantAOIChecker
from PySide6.QtCore import QThread
from PIL import Image, ImageDraw, ImageFont  # <--- Add this import


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


# --- Base Class for Bezel/PWB FastSAM Segmenters ---
class BaseFastSamSegmenter:
    def __init__(self, model_type="x", model_path="ai-models/", imgsz=1024, conf=0.2, iou=0.9):
        self.device = "cpu"
        model_file = f"FastSAM-{model_type}.pt"
        self.model_full_path = os.path.join(model_path, model_file)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.is_image_shown = False
        print(f"[{self.__class__.__name__}] Initializing...")
        start_time = time.time()
        self.model = self._load_model()
        load_time = time.time() - start_time
        if self.model:
            print(f"[{self.__class__.__name__}] Loaded {self.model_full_path} in {load_time:.2f}s")
        else:
            print(f"[Error][{self.__class__.__name__}] FAILED load: {self.model_full_path}")

    def _load_model(self):
        try:
            if os.path.exists(self.model_full_path):
                return FastSAM(self.model_full_path)
            else:
                print(f"[Error] Model not found: {self.model_full_path}")
                return None
        except Exception as e:
            print(f"[Error] Load model {self.model_full_path}: {e}")
            return None

    def run_inference(self, image_path_or_array):
        if self.model is None:
            return None
        try:
            results = self.model(image_path_or_array, device=self.device, retina_masks=True, imgsz=self.imgsz,
                                 conf=self.conf, iou=self.iou)
            if results and hasattr(results[0], 'masks') and results[0].masks is not None and hasattr(results[0].masks,
                                                                                                     'data'):
                return results[0].masks.data.cpu().numpy()
            return None
        except Exception as e:
            print(f"[Error] Inference: {e}")
            return None


# --- Combined Bezel/PWB Segmenter ---
class BezelPWBPositionSegmenter(BaseFastSamSegmenter):
    def __init__(self, model_type="x", model_path="ai-models/", output_dir=r"C:\BoardDefectChecker\ai-outputs"):
        super().__init__(model_type=model_type, model_path=model_path, imgsz=1024, conf=0.3, iou=0.3)
        self.output_dir = output_dir
        self.rotation_invariant_checking_config = '''
        {
            "target_objects": {
                "copper_mark": { "feature_ranges": { "area": [100, 200], "aspect_ratio": [0.8, 1.2], "larger_dim": [10, 20], "smaller_dim": [10, 20], "perimeter": [36, 44] }, "count": 2 },
                "bezel": { "feature_ranges": { "area": [2000, 2400], "aspect_ratio": [0.3, 0.7], "larger_dim": [60, 80], "smaller_dim": [30, 40], "perimeter": [200, 215] }, "count": 1 }
            },
            "spatial_relationships": [ {"type": "distance", "objects": ["bezel", "copper_mark"], "min_dist": 50, "max_dist": 100} ],
            "overlap_rules": [ {"objects": ["bezel", "copper_mark"], "threshold": 0.1} ]
        }
        '''
        config_dict = {}
        try:
            config_dict = json.loads(self.rotation_invariant_checking_config)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse rotation_invariant_checking_config JSON: {e}")

        if "target_objects" in config_dict:
            for obj_spec in config_dict["target_objects"].values():
                if "feature_ranges" in obj_spec:
                    for range_type, range_values in obj_spec["feature_ranges"].items():
                        if isinstance(range_values, list): obj_spec["feature_ranges"][range_type] = tuple(range_values)
        if "spatial_relationships" in config_dict:
            for rule in config_dict["spatial_relationships"]:
                if "objects" in rule and isinstance(rule["objects"], list): rule["objects"] = tuple(rule["objects"])
        if "overlap_rules" in config_dict:
            for rule in config_dict["overlap_rules"]:
                if "objects" in rule and isinstance(rule["objects"], list): rule["objects"] = tuple(rule["objects"])

        self.rotation_invariant_checker = RotationInvariantAOIChecker(config_dict)

        # --- Font Configuration ---
        self.font_path = "C:/Windows/Fonts/segoeui.ttf"  # !! Adjust !!
        self.font_size_large = 18
        self.font_size_small = 14
        self.font_large = None
        self.font_small = None
        try:
            # Assumes ImageFont is imported as: from PIL import ImageFont
            self.font_large = ImageFont.truetype(self.font_path, self.font_size_large)
            self.font_small = ImageFont.truetype(self.font_path, self.font_size_small)
            print(f"[{self.__class__.__name__}] Successfully loaded font: {self.font_path}")
        except IOError:
            print(f"[{self.__class__.__name__}] ********* FONT WARNING *********")
            print(f"[{self.__class__.__name__}] Font not found at '{self.font_path}'.")
            print(f"[{self.__class__.__name__}] Text rendering WILL likely FAIL or use a basic default font.")
            print(f"[{self.__class__.__name__}] Ensure the font path is correct for your operating system.")
            print(f"[{self.__class__.__name__}] ********************************")

        print(
            f"[{self.__class__.__name__}] Initialized with conf={self.conf}, iou={self.iou}, output_dir={self.output_dir}")

    # Helper to draw text using Pillow (simplified, no outline)
    def _draw_text_pillow(self, image_bgr, text, position, font, color_rgb):
        """Draws text on a BGR NumPy array using Pillow."""
        if font is None:
            return image_bgr
        try:
            # Assumes Image, ImageDraw, cv2, np are imported
            image_rgb = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(image_rgb)
            # Draw main text
            draw.text(position, text, font=font, fill=color_rgb)
            image_bgr_out = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
            return image_bgr_out
        except Exception as e:
            print(f"[Error] Failed during Pillow text drawing: {e}")
            return image_bgr

    def load_blank_image(self, target_shape):
        # Assumes cv2, np are imported
        blank_image_path = r"C:\BoardDefectChecker\resources\blank.png"
        blank_image = cv2.imread(blank_image_path)
        if blank_image is None:
            print(
                f"[{self.__class__.__name__}] Error: Failed to load blank image from {blank_image_path}. Returning black image.")
            if len(target_shape) == 2: target_shape = (target_shape[0], target_shape[1], 3)
            return np.zeros(target_shape, dtype=np.uint8)
        blank_image = cv2.resize(blank_image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_AREA)
        return blank_image

    def visualize_masks(self, image, masks, visualize_all_masks=False, label_color=(255, 0, 0),
                        window_title="Visualized Masks", is_image_shown=False):
        # Assumes cv2, np are imported
        result_image = image.copy()
        processed_masks = []
        for mask in masks:
            if hasattr(mask, 'cpu'): mask = mask.cpu()
            if hasattr(mask, 'numpy'): mask = mask.numpy()
            processed_masks.append(mask.astype(bool))

        if visualize_all_masks:
            for mask_area in processed_masks:
                color = np.random.rand(3) * 255
                result_image[mask_area] = result_image[mask_area] * 0.5 + color * 0.5

        if is_image_shown:
            print(f"[Debug] Showing {window_title}")
            resized_img = cv2.resize(result_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_title, resized_img)
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            print(f"[Debug] Closed {window_title}")
        return result_image

    def draw_bounding_boxes(self, image, classified_masks, label_color=(255, 0, 0), window_title="Bounding Boxes",
                            is_image_shown=False):
        # Assumes cv2, np, math are imported
        height, width = image.shape[:2]
        mask_image = image.copy()

        # --- Parameters ---
        block_width = 200
        block_height = 150
        line_spacing_pixels = 22
        margin = 20
        grey_color = (150, 150, 150)
        black_color_rgb = (0, 0, 0)  # Use black for parameter text
        cell_margin = 5
        # --- End Parameters ---

        max_display_masks = 30
        displayed_blocks = 0
        all_mask_info_with_type = []
        for obj_type, masks_list in classified_masks.items():
            for mask_info in masks_list:
                if displayed_blocks < max_display_masks:
                    all_mask_info_with_type.append({"type": obj_type, **mask_info})
                    displayed_blocks += 1
                else:
                    break
            if displayed_blocks >= max_display_masks: break
        total_blocks_to_display = len(all_mask_info_with_type)

        table_image = np.full((height, width, 3), grey_color, dtype=np.uint8)
        blocks_per_column = max(1, (height - 2 * margin) // block_height)
        num_columns = max(1, math.ceil(total_blocks_to_display / blocks_per_column))

        for idx, mask_data in enumerate(all_mask_info_with_type, 1):
            mask = mask_data["mask"]
            obj_type = mask_data["type"]
            features = mask_data.get("features", self.rotation_invariant_checker.extract_features(mask))
            labels = [f"Area: {features.get('area', 0):.0f}", f"Perimeter: {features.get('perimeter', 0):.0f}",
                      f"Larger Dim: {features.get('larger_dim', 0):.0f}",
                      f"Smaller Dim: {features.get('smaller_dim', 0):.0f}",
                      f"Aspect Ratio: {features.get('aspect_ratio', 0):.2f}"]

            mask_color_bgr = np.random.rand(3) * 255
            mask_color_rgb = (int(mask_color_bgr[2]), int(mask_color_bgr[1]), int(mask_color_bgr[0]))
            mask_area = mask.astype(bool)
            mask_image[mask_area] = mask_image[mask_area] * 0.5 + mask_color_bgr * 0.5

            y_indices, x_indices = np.where(mask)
            if len(y_indices) == 0: continue
            x1, y1 = np.min(x_indices), np.min(y_indices)
            x2, y2 = np.max(x_indices), np.max(y_indices)
            cv2.rectangle(mask_image, (max(0, x1), max(0, y1)), (min(width - 1, x2), min(height - 1, y2)), label_color,
                          2)

            centroid_x = int(features.get("centroid_x", x1 + (x2 - x1) // 2))
            centroid_y = int(features.get("centroid_y", y1 + (y2 - y1) // 2))
            centroid_x = max(0, min(width - 1, centroid_x))
            centroid_y = max(0, min(height - 1, centroid_y))

            circle_radius = 15
            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, (255, 255, 255), -1)
            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, (0, 0, 0), 1)
            number_text = str(idx)
            if self.font_small:
                text_width = self.font_small.getlength(number_text)
                text_height = self.font_size_small
                text_x = max(0, centroid_x - int(text_width / 2))
                text_y = max(0, centroid_y - int(text_height / 1.5))
                mask_image = self._draw_text_pillow(mask_image, number_text, (text_x, text_y), self.font_small,
                                                    mask_color_rgb)

            column = (idx - 1) // blocks_per_column
            row = (idx - 1) % blocks_per_column
            block_base_x = margin + column * block_width
            block_base_y = margin + row * block_height

            cell_tl_x = block_base_x - cell_margin
            cell_tl_y = block_base_y - cell_margin
            cell_br_x = block_base_x + block_width - cell_margin
            cell_br_y = block_base_y + block_height - cell_margin

            h_table, w_table = table_image.shape[:2]
            if 0 <= cell_tl_x < w_table and 0 <= cell_tl_y < h_table and \
                    cell_tl_x < cell_br_x < w_table and cell_tl_y < cell_br_y < h_table:

                cell_top_left_int = (int(cell_tl_x), int(cell_tl_y))
                cell_bottom_right_int = (int(cell_br_x), int(cell_br_y))
                cv2.rectangle(table_image, cell_top_left_int, cell_bottom_right_int, (0, 0, 0), 1)

                if self.font_large:
                    text_x = block_base_x + cell_margin
                    text_y = block_base_y + cell_margin
                    # Draw title text (Mask number/type) in its color
                    table_image = self._draw_text_pillow(table_image, f"Mask {idx} ({obj_type})", (text_x, text_y),
                                                         self.font_large, mask_color_rgb)
                    text_y += line_spacing_pixels

                    param_indent_pixels = 15
                    for label in labels:
                        if text_y + line_spacing_pixels < cell_br_y:
                            # Draw parameter text in BLACK
                            table_image = self._draw_text_pillow(table_image, label,
                                                                 (text_x + param_indent_pixels, text_y),
                                                                 self.font_large, black_color_rgb)
                            text_y += line_spacing_pixels
                        else:
                            break
            else:
                print(
                    f"[Warning] Skipping drawing cell/text for BBox Mask {idx} ({obj_type}) due to out-of-bounds coordinates/layout.")

        if is_image_shown:
            print(f"[Debug] Showing {window_title} (Mask Image)")
            resized_mask_img = cv2.resize(mask_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_title, resized_mask_img)
            table_window_title = f"{window_title} - Parameters"
            resized_table_img = cv2.resize(table_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow(table_window_title, resized_table_img)
            print(f"[Debug] Displaying images. Press any key in a window to close both.")
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            cv2.destroyWindow(table_window_title)
            print(f"[Debug] Closed {window_title} (Mask Image)")
            print(f"[Debug] Closed {table_window_title} (Table Image)")
        return mask_image, table_image

    def draw_test_visualization(self, image, test_results, label_color=(255, 0, 0), window_title="Test Visualization",
                                is_image_shown=False):
        # Assumes cv2, np, math are imported
        height, width = image.shape[:2]
        mask_image = image.copy()

        # --- Parameters ---
        block_width = 200
        block_height = 150
        line_spacing_pixels = 22
        margin = 20
        grey_color = (150, 150, 150)
        black_color_rgb = (0, 0, 0)  # Use black for parameter text
        cell_margin = 5
        # --- End Parameters ---

        total_blocks = len(test_results)
        table_image = np.full((height, width, 3), grey_color, dtype=np.uint8)
        blocks_per_column = max(1, (height - 2 * margin) // block_height)
        num_columns = max(1, math.ceil(total_blocks / blocks_per_column))

        for idx, result in enumerate(test_results, 1):
            mask = result["mask"]
            min_rect_vertices = result["min_rect_vertices"]
            features = result["features"]
            labels = result["labels"]

            mask_color_bgr = np.random.rand(3) * 255
            mask_color_rgb = (int(mask_color_bgr[2]), int(mask_color_bgr[1]), int(mask_color_bgr[0]))
            mask_area = mask.astype(bool)
            mask_image[mask_area] = mask_image[mask_area] * 0.5 + mask_color_bgr * 0.5

            points = np.array(min_rect_vertices, dtype=np.int32)
            cv2.polylines(mask_image, [points], isClosed=True, color=label_color, thickness=2)

            centroid_x = int(features.get("centroid_x", 0))
            centroid_y = int(features.get("centroid_y", 0))
            centroid_x = max(0, min(width - 1, centroid_x))
            centroid_y = max(0, min(height - 1, centroid_y))

            circle_radius = 15
            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, (255, 255, 255), -1)
            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, (0, 0, 0), 1)
            number_text = str(idx)
            if self.font_small:
                text_width = self.font_small.getlength(number_text)
                text_height = self.font_size_small
                text_x = max(0, centroid_x - int(text_width / 2))
                text_y = max(0, centroid_y - int(text_height / 1.5))
                mask_image = self._draw_text_pillow(mask_image, number_text, (text_x, text_y), self.font_small,
                                                    mask_color_rgb)

            column = (idx - 1) // blocks_per_column
            row = (idx - 1) % blocks_per_column
            block_base_x = margin + column * block_width
            block_base_y = margin + row * block_height

            cell_tl_x = block_base_x - cell_margin
            cell_tl_y = block_base_y - cell_margin
            cell_br_x = block_base_x + block_width - cell_margin
            cell_br_y = block_base_y + block_height - cell_margin

            h_table, w_table = table_image.shape[:2]
            if 0 <= cell_tl_x < w_table and 0 <= cell_tl_y < h_table and \
                    cell_tl_x < cell_br_x < w_table and cell_tl_y < cell_br_y < h_table:

                cell_top_left_int = (int(cell_tl_x), int(cell_tl_y))
                cell_bottom_right_int = (int(cell_br_x), int(cell_br_y))
                cv2.rectangle(table_image, cell_top_left_int, cell_bottom_right_int, (0, 0, 0), 1)

                if self.font_large:
                    text_x = block_base_x + cell_margin
                    text_y = block_base_y + cell_margin
                    # Draw title text (Mask number) in its color
                    table_image = self._draw_text_pillow(table_image, f"Mask {idx}", (text_x, text_y), self.font_large,
                                                         mask_color_rgb)
                    text_y += line_spacing_pixels

                    param_indent_pixels = 15
                    for label in labels:
                        if text_y + line_spacing_pixels < cell_br_y:
                            # Draw parameter text in BLACK
                            table_image = self._draw_text_pillow(table_image, label,
                                                                 (text_x + param_indent_pixels, text_y),
                                                                 self.font_large, black_color_rgb)
                            text_y += line_spacing_pixels
                        else:
                            break
            else:
                print(
                    f"[Warning] Skipping drawing cell/text for Test Mask {idx} due to out-of-bounds coordinates/layout.")

        if is_image_shown:
            print(f"[Debug] Showing {window_title} (Mask Image)")
            resized_mask_img = cv2.resize(mask_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow(window_title, resized_mask_img)
            table_window_title = f"{window_title} - Parameters"
            resized_table_img = cv2.resize(table_image, (1280, 800), interpolation=cv2.INTER_AREA)
            cv2.imshow(table_window_title, resized_table_img)
            print(f"[Debug] Displaying images. Press any key in a window to close both.")
            cv2.waitKey(0)
            cv2.destroyWindow(window_title)
            cv2.destroyWindow(table_window_title)
            print(f"[Debug] Closed {window_title} (Mask Image)")
            print(f"[Debug] Closed {table_window_title} (Table Image)")
        return mask_image, table_image

    def process_image(self, image_array, max_masks=None, label_color=(255, 0, 0), test_mode=False,
                      max_masks_to_show=30, edge_threshold=5, sort_by_area=True, window_title="Visualization",
                      is_image_shown=False):
        # (process_image method remains unchanged from the previous version)
        # Assumes time, gc, traceback are imported
        blank_shape = (768, 1024, 3)
        if image_array is not None and image_array.size > 0:
            blank_shape = image_array.shape if len(image_array.shape) == 3 else (
            image_array.shape[0], image_array.shape[1], 3)
        blank_image = self.load_blank_image(blank_shape)

        if self.model is None:
            print(f"[{self.__class__.__name__}] Result: NG. Reason: Model not initialized.")
            return blank_image, blank_image, 0.0, "NG", "Model N/A"
        if image_array is None or image_array.size == 0:
            print(f"[{self.__class__.__name__}] Result: NG. Reason: Invalid input image (None or empty).")
            return blank_image, blank_image, 0.0, "NG", "Invalid input image"

        start_time = time.time()
        print(f"[{self.__class__.__name__}] Running inference...")
        try:
            masks = self.run_inference(image_array)
        except Exception as e:
            inf_time = time.time() - start_time;
            print(f"[{self.__class__.__name__}] Result: NG. Reason: Inference Error ({e}). Time: {inf_time:.3f}s")
            if is_image_shown: cv2.destroyAllWindows(); return blank_image, blank_image, inf_time, "NG", f"Inference Error: {e}"
        inf_time = time.time() - start_time

        if masks is None or len(masks) == 0:
            reason = "No Masks (Infer)" if masks is None else "Zero Masks (Infer)"
            proc_time = time.time() - start_time;
            print(f"[{self.__class__.__name__}] Result: NG. Reason: {reason}. Time: {proc_time:.3f}s")
            if is_image_shown:
                try:
                    resized_blank_img = cv2.resize(blank_image, (1280, 800), interpolation=cv2.INTER_AREA)
                    cv2.imshow(window_title, resized_blank_img);
                    table_window_title = f"{window_title} - Parameters";
                    cv2.imshow(table_window_title, resized_blank_img)
                    print(f"[Debug] Displaying blank images. Press any key in a window to close both.");
                    cv2.waitKey(0);
                    cv2.destroyWindow(window_title);
                    cv2.destroyWindow(table_window_title)
                    print(f"[Debug] Closed {window_title} (Blank Mask Image)");
                    print(f"[Debug] Closed {table_window_title} (Blank Table Image)")
                except Exception as display_err:
                    print(f"[Debug] Error displaying blank images: {display_err}"); cv2.destroyAllWindows()
            return blank_image, blank_image, proc_time, "NG", reason

        if max_masks is not None and len(masks) > max_masks:
            print(f"[{self.__class__.__name__}] Limiting masks from {len(masks)} to {max_masks} before processing.");
            masks = masks[:max_masks]

        image_shape = image_array.shape[:2]
        mask_img, table_img = blank_image, blank_image
        status, reason = None, None

        try:
            if test_mode:
                test_results = self.rotation_invariant_checker.test_masks(masks, image_shape=image_shape,
                                                                          max_masks_to_show=max_masks_to_show,
                                                                          edge_threshold=edge_threshold,
                                                                          sort_by_area=sort_by_area)
                mask_img, table_img = self.draw_test_visualization(image_array, test_results, label_color=label_color,
                                                                   window_title=window_title,
                                                                   is_image_shown=is_image_shown)
                proc_time = time.time() - start_time;
                print(
                    f"[{self.__class__.__name__}] Test Mode: Displayed {len(test_results)} masks. Time: {proc_time:.3f}s (Inf: {inf_time:.3f}s)")
                status, reason = None, None
            else:
                classified_masks = self.rotation_invariant_checker.classify_masks(masks)
                mask_img, table_img = self.draw_bounding_boxes(image_array, classified_masks, label_color=label_color,
                                                               window_title=window_title, is_image_shown=is_image_shown)
                ok, spatial_reason = self.rotation_invariant_checker.verify_spatial_relationships(classified_masks)
                overlap_ok, overlap_reason = True, ""
                if ok: overlap_ok, overlap_reason = self.rotation_invariant_checker.check_overlaps(classified_masks)
                final_ok = ok and overlap_ok;
                status = "OK" if final_ok else "NG";
                reason = "All checks passed" if final_ok else (spatial_reason if not ok else overlap_reason)
                proc_time = time.time() - start_time;
                print(
                    f"[{self.__class__.__name__}] Result: {status}. Reason: {reason}. Time: {proc_time:.3f}s (Inf: {inf_time:.3f}s)")
        except Exception as e:
            proc_time = time.time() - start_time;
            status = "NG";
            reason = f"Processing Error: {e}"
            print(f"[{self.__class__.__name__}] Result: NG. Reason: {reason}. Time: {proc_time:.3f}s")
            import traceback;
            print(traceback.format_exc())
            if is_image_shown: cv2.destroyAllWindows(); mask_img, table_img = blank_image, blank_image
        finally:
            # Clean up large variables
            del masks  # Delete masks list now its use is finished in both branches
            if 'test_results' in locals(): del test_results
            if 'classified_masks' in locals(): del classified_masks
            gc.collect()
            # Ensure windows are closed if is_image_shown was False but windows were opened internally
            # or if an error occurred after opening windows
            if not is_image_shown:
                cv2.destroyAllWindows()  # Close any stray windows

        return mask_img, table_img, proc_time, status, reason


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


if __name__ == "__main__":
    token_fpc_main()

# if __name__ == "__main__":
# small_fpc_main()
