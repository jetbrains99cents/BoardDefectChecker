# modules/feature_learners/learning_visualizer.py

import numpy as np
import cv2
# Added Set to imports
from typing import List, Dict, Any, Tuple, Optional, Set
import math

# Attempt to import Pillow and load fonts globally within this module
try:
    from PIL import Image, ImageDraw, ImageFont
    _pillow_available = True
    _font_large_display = None; _font_small_display = None
    try:
        _font_path_display = "C:/Windows/Fonts/segoeui.ttf" # Adjust path if needed
        _font_size_large_display = 18; _font_size_small_display = 14
        _font_large_display = ImageFont.truetype(_font_path_display, _font_size_large_display)
        _font_small_display = ImageFont.truetype(_font_path_display, _font_size_small_display)
        # print(f"[Visualizer] Loaded font: {_font_path_display}")
    except Exception as font_e:
         # print(f"[Visualizer][Warning] Error loading font: {font_e}")
         _font_large_display = None; _font_small_display = None
except ImportError:
    _pillow_available = False
    # print("[Visualizer][Warning] Pillow library not found.")
    # Define dummy classes/functions if Pillow is missing
    class ImageFont:
        @staticmethod
        def truetype(font, size): return None
    class Image: pass
    class ImageDraw: pass
    _font_large_display = None; _font_small_display = None
    _font_size_small_display = 14


# --- Helper Function for Collision Avoidance (Module Level - Unchanged) ---
def _find_non_overlapping_position(
                                   center_x: int, center_y: int, radius: int,
                                   placed_circles: List[Tuple[int, int, int]],
                                   max_attempts: int = 8, step_scale: float = 1.5) -> Tuple[int, int]:
    """Tries to find a non-overlapping position for a circle."""
    # (Implementation Unchanged)
    current_pos = (center_x, center_y); min_dist_sq = (2 * radius) ** 2
    is_overlapping = any( (current_pos[0] - px)**2 + (current_pos[1] - py)**2 < (radius + pr)**2 for px, py, pr in placed_circles )
    if not is_overlapping: return current_pos
    step = int(radius * step_scale)
    offsets = [(0, -step), (step, 0), (0, step), (-step, 0), (step, -step), (step, step), (-step, step), (-step, -step)]
    for attempt in range(max_attempts):
        dx, dy = offsets[attempt % len(offsets)]; next_x = center_x + dx; next_y = center_y + dy
        is_overlapping = any( (next_x - px)**2 + (next_y - py)**2 < (radius + pr)**2 for px, py, pr in placed_circles )
        if not is_overlapping: return (next_x, next_y)
    return (center_x, center_y)
# --- END HELPER FUNCTION ---


# --- UPDATED Interactive Selection Function ---
def select_masks_interactive(image: np.ndarray,
                             test_results: List[Dict[str, Any]],
                             object_types_to_learn: List[str]
                            ) -> Tuple[Dict[str, List[int]], bool]:
    """
    Allows interactive selection of masks using mouse clicks in an OpenCV window.
    Includes debugging steps for grey screen issue.
    """
    if image is None or not test_results:
        print("[Visualizer][Error] Invalid image or test_results for interactive selection.")
        return {}, False

    print("[Debug Viz] Entering select_masks_interactive...") # DEBUG
    original_height, original_width = image.shape[:2]
    base_image = image.copy()
    if base_image is None or base_image.size == 0:
        print("[Debug Viz][Error] base_image is invalid after copy!")
        return {}, False
    print(f"[Debug Viz] base_image shape: {base_image.shape}, dtype: {base_image.dtype}") # DEBUG

    # --- DEBUG: Show base image immediately ---
    # try:
    #     cv2.imshow("DEBUG Base Image", cv2.resize(base_image, (640, 480)))
    #     cv2.waitKey(0)
    #     cv2.destroyWindow("DEBUG Base Image")
    # except Exception as e:
    #     print(f"[Debug Viz] Error showing base image: {e}")
    # --- END DEBUG ---


    display_image = base_image.copy() # Image shown in the window, updated dynamically

    # --- Fixed Display Size ---
    display_width = 1280
    display_height = 800

    # --- Calculate Scaling Factors ---
    scale_x = display_width / original_width if original_width > 0 else 1
    scale_y = display_height / original_height if original_height > 0 else 1

    # --- State Variables ---
    all_selections: Dict[str, List[int]] = {obj_type: [] for obj_type in object_types_to_learn}
    current_type_index = 0
    current_type_selections: List[int] = []
    selected_mask_indices_overall: Set[int] = set()
    stop_requested = False
    redraw_needed = True

    # --- Drawing Parameters ---
    circle_radius = 15
    font_face = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.6; text_thickness = 1
    type_colors = { "bezel": (0, 255, 0), "copper_mark": (255, 0, 0), "stamped_mark": (0, 0, 255) }
    default_type_color = (200, 200, 200); highlight_color = (0, 255, 255)
    text_color = (255, 255, 255); text_bg_color = (0, 0, 0)

    # --- Pre-calculate initial circle positions & Assign distinct colors ---
    initial_circle_positions = []
    placed_circles_initial: List[Tuple[int, int, int]] = []
    mask_draw_styles = {}
    num_masks = len(test_results)
    hsv_colors = [(int(i * 180 / num_masks), 255, 200) for i in range(num_masks)] if num_masks > 0 else []
    bgr_colors = [cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0] for hsv in hsv_colors]
    print(f"[Debug Viz] Pre-calculating {num_masks} circle positions...") # DEBUG
    for idx, result in enumerate(test_results):
        features = result.get("features"); mask_idx_1_based = idx + 1
        distinct_color = tuple(map(int, bgr_colors[idx])) if idx < len(bgr_colors) else (255, 255, 255)
        mask_draw_styles[mask_idx_1_based] = {'color': distinct_color}
        if features:
            cx = int(features.get("centroid_x", 0)); cy = int(features.get("centroid_y", 0))
            final_cx, final_cy = _find_non_overlapping_position(cx, cy, circle_radius, placed_circles_initial)
            placed_circles_initial.append((final_cx, final_cy, circle_radius))
            initial_circle_positions.append((final_cx, final_cy))
        else: initial_circle_positions.append(None)
    print(f"[Debug Viz] Finished pre-calculating positions.") # DEBUG

    # --- Mouse Callback Function ---
    def on_mouse_event(event, x, y, flags, userdata):
        nonlocal current_type_selections, selected_mask_indices_overall, redraw_needed
        if event == cv2.EVENT_LBUTTONDOWN:
            if scale_x == 0 or scale_y == 0: return
            original_x = int(x / scale_x); original_y = int(y / scale_y)
            # print(f"Click: Win({x},{y}) -> Orig({original_x},{original_y})") # Debug
            clicked_mask_idx = -1
            for i in range(len(test_results) - 1, -1, -1):
                result = test_results[i]; vertices = result.get("min_rect_vertices")
                if vertices and len(vertices) == 4:
                    try:
                        contour = np.array(vertices, dtype=np.int32)
                        if cv2.pointPolygonTest(contour, (float(original_x), float(original_y)), False) >= 0:
                            clicked_mask_idx = i + 1; break
                    except Exception as poly_err: print(f"[Warning] Error during pointPolygonTest: {poly_err}"); continue
            if clicked_mask_idx != -1:
                if clicked_mask_idx in selected_mask_indices_overall: print(f"Mask #{clicked_mask_idx} already assigned."); return
                if clicked_mask_idx in current_type_selections: current_type_selections.remove(clicked_mask_idx); print(f"Deselected Mask #{clicked_mask_idx}")
                else: current_type_selections.append(clicked_mask_idx); print(f"Selected Mask #{clicked_mask_idx}")
                redraw_needed = True
    # --- End Mouse Callback ---

    # --- Main Interaction Loop ---
    window_name = "Interactive Mask Selection (1280x800)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, display_width, display_height)
    cv2.setMouseCallback(window_name, on_mouse_event)
    print("[Debug Viz] Starting main interaction loop...") # DEBUG

    while current_type_index < len(object_types_to_learn):
        current_obj_type = object_types_to_learn[current_type_index]
        # print(f"[Debug Viz] Loop Start: redraw_needed={redraw_needed}, current_type='{current_obj_type}'") # DEBUG

        if redraw_needed:
            print("[Debug Viz] Redrawing image...") # DEBUG
            display_image = base_image.copy() # Reset to base image
            if display_image is None or display_image.size == 0:
                 print("[Debug Viz][Error] display_image became invalid before drawing loop!")
                 break # Exit loop if image is bad

            # Draw rectangles and circles
            num_drawn = 0 # DEBUG
            for idx, result in enumerate(test_results):
                mask_idx = idx + 1
                pos_info = initial_circle_positions[idx]
                min_rect_vertices = result.get("min_rect_vertices", [])
                style = mask_draw_styles.get(mask_idx, {'color': (255, 255, 255)})
                if pos_info is None: continue # Skip if no position calculated
                final_cx, final_cy = pos_info
                is_selected_this_type = mask_idx in current_type_selections
                is_selected_other_type = mask_idx in selected_mask_indices_overall and not is_selected_this_type
                line_thickness = 2; draw_color = style['color']
                if is_selected_this_type: draw_color = highlight_color; line_thickness = 3
                elif is_selected_other_type: pass
                points = np.array(min_rect_vertices, dtype=np.int32)
                if len(points) == 4: cv2.polylines(display_image, [points], isClosed=True, color=draw_color, thickness=line_thickness)
                circle_border_color = highlight_color if is_selected_this_type else (0,0,0)
                circle_border_thickness = 2 if is_selected_this_type else 1
                cv2.circle(display_image, (final_cx, final_cy), circle_radius, (150, 150, 150), -1)
                cv2.circle(display_image, (final_cx, final_cy), circle_radius, circle_border_color, circle_border_thickness)
                num_text = str(mask_idx); (tw, th), _ = cv2.getTextSize(num_text, font_face, 0.5, 1)
                cv2.putText(display_image, num_text, (final_cx - tw // 2, final_cy + th // 2), font_face, 0.5, (0,0,0), 1, cv2.LINE_AA)
                num_drawn+=1 # DEBUG
            print(f"[Debug Viz] Drew {num_drawn} overlays.") # DEBUG
            redraw_needed = False

        # Display Instructions
        instruction_image = display_image.copy()
        if instruction_image is None or instruction_image.size == 0:
            print("[Debug Viz][Error] instruction_image became invalid before text drawing!")
            break # Exit loop if image is bad

        line1 = f"Labeling: '{current_obj_type}' ({len(current_type_selections)} selected)"
        line2 = "L-Click Rect: Select | Space/Enter: Next Type | G/S: Generate | Esc: Cancel"
        (lw1, lh1), bl1 = cv2.getTextSize(line1, font_face, font_scale, text_thickness)
        (lw2, lh2), bl2 = cv2.getTextSize(line2, font_face, font_scale, text_thickness)
        cv2.rectangle(instruction_image, (5, 5), (10 + max(lw1, lw2), 15 + lh1 + lh2 + 5), text_bg_color, -1)
        cv2.putText(instruction_image, line1, (10, 10 + lh1), font_face, font_scale, text_color, text_thickness, cv2.LINE_AA)
        cv2.putText(instruction_image, line2, (10, 15 + lh1 + lh2), font_face, font_scale, text_color, text_thickness, cv2.LINE_AA)

        # Resize the final image with instructions just before showing
        try:
            display_sized_image = cv2.resize(instruction_image, (display_width, display_height), interpolation=cv2.INTER_AREA)
            if display_sized_image is None or display_sized_image.size == 0:
                 raise ValueError("Resized image is invalid")
        except Exception as resize_err:
            print(f"[Debug Viz][Error] Could not resize image for display: {resize_err}")
            display_sized_image = instruction_image # Show original size on error

        # --- DEBUG: Check image before showing ---
        if display_sized_image is None or display_sized_image.size == 0:
             print("[Debug Viz][Error] display_sized_image is invalid before imshow!")
             break
        # print(f"[Debug Viz] Showing image sized {display_sized_image.shape}")
        # --- END DEBUG ---

        cv2.imshow(window_name, display_sized_image)
        key = cv2.waitKey(50) # Process events, wait 50ms

        # Handle Key Presses
        if key == 27: # ESC
            print("Operation cancelled by user."); all_selections = {}; stop_requested = True; break
        elif key in [ord(' '), 13]: # Space or Enter
            print(f"Confirmed {len(current_type_selections)} selections for '{current_obj_type}'.")
            all_selections[current_obj_type] = list(current_type_selections)
            selected_mask_indices_overall.update(current_type_selections)
            current_type_index += 1; current_type_selections = []; redraw_needed = True
            if current_type_index < len(object_types_to_learn): print(f"\nNow labeling: '{object_types_to_learn[current_type_index]}'")
        elif key in [ord('g'), ord('s')]: # Generate/Stop
            print("Generate/Stop requested by user.")
            all_selections[current_obj_type] = list(current_type_selections)
            selected_mask_indices_overall.update(current_type_selections)
            stop_requested = True; break

    # --- Cleanup ---
    try:
        cv2.destroyWindow(window_name)
    except Exception as e_destroy:
        print(f"[Debug Viz] Error destroying window: {e_destroy}") # Log error but continue
    print("\n--- Finished Interactive Selection ---")
    return all_selections, stop_requested
# --- END Interactive Selection Function ---


# --- display_for_learning (Legacy wrapper - unchanged) ---
def display_for_learning(image: np.ndarray, test_results: List[Dict[str, Any]]):
    """Legacy wrapper. No longer displays interactively."""
    # print("[Visualizer] display_for_learning called (legacy - non-interactive).")
    pass # Does nothing in the new workflow
