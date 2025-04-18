# modules/feature_learners/learning_visualizer.py

import numpy as np
import cv2
from typing import List, Dict, Any
import math

# Attempt to import Pillow and load fonts globally within this module
try:
    from PIL import Image, ImageDraw, ImageFont
    _pillow_available = True
    _font_large_display = None
    _font_small_display = None
    try:
        # --- ADJUST FONT PATH AS NEEDED ---
        _font_path_display = "C:/Windows/Fonts/segoeui.ttf"
        _font_size_large_display = 18
        _font_size_small_display = 14
        _font_large_display = ImageFont.truetype(_font_path_display, _font_size_large_display)
        _font_small_display = ImageFont.truetype(_font_path_display, _font_size_small_display)
        print(f"[Visualizer] Loaded font: {_font_path_display}")
    except IOError:
        print(f"[Visualizer][Warning] Display font not found at {_font_path_display}. Using basic text.")
    except Exception as font_e:
         print(f"[Visualizer][Warning] Error loading font: {font_e}")

except ImportError:
    _pillow_available = False
    print("[Visualizer][Warning] Pillow library not found. Text display will be basic.")
    # Define dummy classes/functions if Pillow is missing to avoid NameErrors later
    class ImageFont:
        @staticmethod
        def truetype(font, size): return None
    class Image: pass
    class ImageDraw: pass
    _font_large_display = None
    _font_small_display = None
    _font_size_small_display = 14 # Still needed for fallback calculation


def display_for_learning(image: np.ndarray, test_results: List[Dict[str, Any]]):
    """
    Displays the mask image and parameter table for interactive learning.
    Numbers on masks are drawn inside a grey circle with black text.

    Args:
        image: The original image (NumPy array).
        test_results: A list of dictionaries from RotationInvariantAOIChecker.test_masks.
                      Each dict should contain 'mask', 'features', 'min_rect_vertices', 'labels'.
    """
    if not test_results:
        print("[Visualizer][Info] No test results to display for learning.")
        return

    height, width = image.shape[:2]
    mask_image = image.copy() # Image for drawing overlays

    # --- Drawing Parameters ---
    block_width = 200
    block_height = 150
    line_spacing_pixels = 22
    margin = 20
    grey_color_bgr = (150, 150, 150) # Grey color for circle background (BGR for OpenCV)
    black_color_rgb = (0, 0, 0)      # Black color for number text (RGB for Pillow)
    black_color_bgr = (0, 0, 0)      # Black color for number text (BGR for OpenCV fallback)
    cell_margin = 5
    circle_radius = 15
    # --- End Drawing Parameters ---

    # --- Table Image Setup ---
    # Create a grey background image for the parameter table
    table_image_np = np.full((height, width, 3), grey_color_bgr, dtype=np.uint8)
    table_image_pil = None
    table_draw = None
    # Setup Pillow drawing context if available
    if _pillow_available and _font_large_display: # Check font too
        try:
            table_image_pil = Image.fromarray(cv2.cvtColor(table_image_np, cv2.COLOR_BGR2RGB))
            table_draw = ImageDraw.Draw(table_image_pil)
        except Exception as pil_e:
            print(f"[Visualizer][Warning] Pillow setup failed for table drawing: {pil_e}")
            table_image_pil = None; table_draw = None # Fallback

    # Calculate table layout
    blocks_per_column = max(1, (height - 2 * margin) // block_height)
    # num_columns = max(1, math.ceil(len(test_results) / blocks_per_column)) # Not needed

    # --- Draw elements for each mask result ---
    for idx, result in enumerate(test_results, 1):
        mask = result.get("mask")
        min_rect_vertices = result.get("min_rect_vertices", [])
        features = result.get("features")
        labels = result.get("labels", []) # Use pre-formatted labels

        # Skip if essential data is missing
        if mask is None or features is None:
            print(f"[Visualizer][Warning] Skipping mask {idx} due to missing mask or features.")
            continue

        # --- Draw on Mask Image ---
        mask_color_bgr = np.random.rand(3) * 255 # Random color for overlay
        mask_color_rgb = (int(mask_color_bgr[2]), int(mask_color_bgr[1]), int(mask_color_bgr[0]))

        if isinstance(mask, np.ndarray):
            # Draw mask overlay
            mask_area = mask.astype(bool)
            try:
                mask_image[mask_area] = mask_image[mask_area] * 0.5 + mask_color_bgr * 0.5
            except IndexError:
                print(f"[Visualizer][Warning] Skipping mask overlay for index {idx} due to shape mismatch.")
                continue

            # Draw minimum area rectangle (rotated)
            points = np.array(min_rect_vertices, dtype=np.int32)
            if len(points) == 4:
                 cv2.polylines(mask_image, [points], isClosed=True, color=(255,0,0), thickness=2) # Red polyline

            # Draw numbered circle at centroid
            centroid_x = int(features.get("centroid_x", 0))
            centroid_y = int(features.get("centroid_y", 0))
            # Clamp centroid to be within image bounds for drawing
            centroid_x = max(circle_radius, min(width - 1 - circle_radius, centroid_x))
            centroid_y = max(circle_radius, min(height - 1 - circle_radius, centroid_y))

            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, grey_color_bgr, -1) # Grey fill
            cv2.circle(mask_image, (centroid_x, centroid_y), circle_radius, black_color_bgr, 1) # Black border

            # Draw number text (Pillow preferred)
            number_text = str(idx)
            if _pillow_available and _font_small_display:
                 try:
                     img_pil_temp = Image.fromarray(cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB))
                     draw_temp = ImageDraw.Draw(img_pil_temp)
                     try: text_width = _font_small_display.getlength(number_text)
                     except AttributeError: text_width, _ = draw_temp.textsize(number_text, font=_font_small_display)
                     text_height = _font_size_small_display # Use loaded font size
                     # Center text within circle
                     text_x_num = max(0, centroid_x - int(text_width / 2))
                     text_y_num = max(0, centroid_y - int(text_height / 1.5)) # Adjust vertical centering
                     draw_temp.text((text_x_num, text_y_num), number_text, font=_font_small_display, fill=black_color_rgb)
                     mask_image = cv2.cvtColor(np.array(img_pil_temp), cv2.COLOR_RGB2BGR) # Convert back
                 except Exception as e_draw:
                     print(f"[Visualizer][Error] Pillow text drawing failed for mask {idx}: {e_draw}")
                     # Fallback to OpenCV text if Pillow fails
                     cv2.putText(mask_image, number_text, (centroid_x - 7, centroid_y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color_bgr, 1)
            else:
                # Fallback to OpenCV text if Pillow not available or font failed
                cv2.putText(mask_image, number_text, (centroid_x - 7, centroid_y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, black_color_bgr, 1)
        # --- End Draw on Mask Image ---

        # --- Draw Parameter Table Cell ---
        column = (idx - 1) // blocks_per_column
        row = (idx - 1) % blocks_per_column
        block_base_x = margin + column * block_width
        block_base_y = margin + row * block_height
        cell_tl_x = block_base_x - cell_margin; cell_tl_y = block_base_y - cell_margin
        cell_br_x = block_base_x + block_width - cell_margin; cell_br_y = block_base_y + block_height - cell_margin
        h_table, w_table = height, width # Table image dimensions

        # Check if cell coordinates are valid before drawing
        if 0 <= cell_tl_x < w_table and 0 <= cell_tl_y < h_table and \
           cell_tl_x < cell_br_x < w_table and cell_tl_y < cell_br_y < h_table:

            # Use Pillow drawing context if available
            if _pillow_available and table_draw and _font_large_display:
                table_draw.rectangle([(cell_tl_x, cell_tl_y), (cell_br_x, cell_br_y)], outline=black_color_rgb, width=1)
                text_x = block_base_x + cell_margin; text_y = block_base_y + cell_margin
                table_draw.text((text_x, text_y), f"Mask {idx}", font=_font_large_display, fill=mask_color_rgb) # Title
                text_y += line_spacing_pixels
                param_indent_pixels = 15
                for label in labels: # Draw pre-formatted labels
                    if text_y + line_spacing_pixels < cell_br_y:
                        table_draw.text((text_x + param_indent_pixels, text_y), label, font=_font_large_display, fill=black_color_rgb) # Params
                        text_y += line_spacing_pixels
                    else: break # Stop if no more space
            else: # Fallback basic OpenCV drawing for table
                cv2.rectangle(table_image_np, (cell_tl_x, cell_tl_y), (cell_br_x, cell_br_y), black_color_bgr, 1)
                text_x = block_base_x + cell_margin; text_y = block_base_y + cell_margin + 15
                cv2.putText(table_image_np, f"Mask {idx}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mask_color_bgr, 1)
                text_y += 20
                for label in labels:
                     if text_y + 20 < cell_br_y:
                         cv2.putText(table_image_np, label, (text_x + 15, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, black_color_bgr, 1)
                         text_y += 18
                     else: break
        else:
            print(f"[Visualizer][Warning] Skipping drawing cell/text for Mask {idx} due to invalid coordinates.")
        # --- End Draw Parameter Table Cell ---

    # --- Finalize Table Image and Display ---
    # Convert final PIL table image back to NumPy BGR format if needed
    if _pillow_available and table_image_pil:
        try:
            table_image_np = cv2.cvtColor(np.array(table_image_pil), cv2.COLOR_RGB2BGR)
        except Exception as convert_e:
             print(f"[Visualizer][Error] Failed converting PIL table image to OpenCV format: {convert_e}")
             # Keep the np.full grey image as fallback

    # Define window names
    win_mask = "Mask Image (for Learning)"
    win_table = "Parameter Table (for Learning)"
    try:
        # Resize for consistent display before showing
        display_width = 960
        display_height = 720 # Example fixed size
        resized_mask = cv2.resize(mask_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
        resized_table = cv2.resize(table_image_np, (display_width, display_height), interpolation=cv2.INTER_AREA)

        cv2.imshow(win_mask, resized_mask)
        cv2.imshow(win_table, resized_table)

        print("\n--- Waiting for Input ---")
        print("Windows showing Mask Image and Parameter Table.")
        print("Review the numbered masks and corresponding parameters.")
        print("Press any key in one of the OpenCV windows to continue to console labeling...")
        cv2.waitKey(0) # Wait indefinitely for user review
    except Exception as e:
        print(f"[Visualizer][Error] Failed to display images for learning: {e}")
    finally:
        # Ensure windows are closed even if errors occur
        cv2.destroyWindow(win_mask)
        cv2.destroyWindow(win_table)
        print("Display windows closed.")
    # --- End Finalize Table Image and Display ---

