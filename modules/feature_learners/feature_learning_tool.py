# modules/feature_learners/feature_learning_tool.py

import os
import sys
import cv2
import numpy as np
# Ensure Set is imported if used elsewhere, though not directly needed in this file's logic now
from typing import List, Dict, Any, Optional, Set
import json
import traceback
import math
import itertools

# --- Imports from this package ---
try:
    from .base import BaseFeatureLearner
    from .statistical import StatisticalFeatureLearner
    # Import the interactive function from the visualizer
    from .learning_visualizer import select_masks_interactive
except ImportError as e_imp:
    print(f"[FATAL ERROR] Could not import base/statistical/visualizer: {e_imp}")
    print("Ensure base.py, statistical.py, and learning_visualizer.py are in the same directory.")
    sys.exit(1)

# --- Imports from other project modules ---
try:
    # Adjust path if necessary based on where you run from
    from modules.rotation_invariant_checking import RotationInvariantAOIChecker
    # Use the specific segmenter needed for this learning task
    from modules.ai_models import BezelPWBPositionSegmenter # Assuming this is the correct one
    SegmenterClass = BezelPWBPositionSegmenter
except ImportError as e:
     print(f"[FATAL ERROR] Failed import Checker/Segmenter: {e}")
     print("Ensure modules/rotation_invariant_checking.py and modules/ai_models.py exist relative to the execution path.")
     sys.exit(1)


# --- UPDATED interactive_learning_session ---
def interactive_learning_session(image_path: str,
                                 segmenter: SegmenterClass,
                                 checker: RotationInvariantAOIChecker,
                                 learner: BaseFeatureLearner,
                                 object_types_to_learn: List[str],
                                 max_masks_infer: int = 100,
                                 max_masks_display: int = 100) -> bool: # Return True if user wants to stop early
    """
    Runs inference, feature extraction, and then launches the interactive
    mask selection window. Adds selected samples and calculates distances.
    Returns True if the user requested to stop/generate early in the interactive window.
    """
    print(f"\n--- Starting Interactive Session for: {os.path.basename(image_path)} ---")
    stop_requested = False

    # --- Load Image ---
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return stop_requested # Cannot proceed without image
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Failed read image {image_path}")
        return stop_requested # Cannot proceed without image

    # --- Run Inference ---
    print("Running inference...")
    masks = []
    try:
        # Assuming run_inference returns a list of numpy arrays or None
        raw_masks = segmenter.run_inference(image)
        # Filter for valid numpy arrays
        if raw_masks is not None and isinstance(raw_masks, (list, np.ndarray)):
             masks = [m for m in raw_masks if isinstance(m, np.ndarray) and m.ndim == 2]
        print(f"Inference complete, got {len(masks)} valid masks.")
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        traceback.print_exc() # Show more details on inference error
        # Decide whether to stop or continue without masks
        return stop_requested # Stop if inference fails

    if not masks:
        print("[Warning] No valid masks found from inference. Skipping labeling for this image.")
        return stop_requested # Continue to next image

    # Optional: Truncate masks if too many
    if max_masks_infer is not None and len(masks) > max_masks_infer:
        print(f"[Info] Truncating masks from {len(masks)} to {max_masks_infer}")
        masks = masks[:max_masks_infer]

    # --- Extract Features ---
    print("Extracting features for detected masks...")
    test_results = []
    try:
        # Get features for all valid masks to allow selection of any
        test_results = checker.test_masks(masks, image.shape[:2], max_masks_to_show=len(masks))
    except Exception as e:
        print(f"[Error] Failed feature extraction (test_masks): {e}")
        return stop_requested # Stop if feature extraction fails

    if not test_results:
        print("[Warning] No valid features extracted from masks. Skipping labeling for this image.")
        return stop_requested # Continue to next image

    # Store results in learner for potential distance calculations later
    learner.set_last_test_results(test_results)

    # --- Launch Interactive Selection ---
    print("Launching interactive selection window...")
    # This function now handles the display and user interaction (clicks, key presses)
    # It returns the selections made and whether the user requested an early stop
    session_labels, stop_requested = select_masks_interactive(
        image=image,
        test_results=test_results,
        object_types_to_learn=object_types_to_learn
    )
    # --- End Interactive Selection ---

    # --- Process Selections ---
    if stop_requested:
         print("User requested early stop during selection.")
         # Add any samples selected before stopping, even if incomplete
         for obj_type, mask_numbers in session_labels.items():
              added_count = 0
              for mn in mask_numbers:
                   # Use learner's add_sample which checks validity
                   if learner.add_sample(obj_type, mn):
                        added_count += 1
              if added_count > 0:
                   print(f"Added {added_count} pre-stop sample(s) for '{obj_type}'.")
         # Return True to signal the main loop to stop
         return stop_requested

    # If not stopped early, add all selected samples
    print("\nAdding selected samples to learner...")
    for obj_type, mask_numbers in session_labels.items():
        added_count = 0
        for mn in mask_numbers:
            # add_sample handles checking index and adding features
            if learner.add_sample(obj_type, mn):
                added_count += 1
        if added_count > 0:
            print(f"Added {added_count} sample(s) for '{obj_type}'.")
    # --- End Add Samples ---


    # --- Calculate Distances (using session_labels from interactive selection) ---
    print("Calculating distances between labeled pairs...")
    distance_added_count = 0
    labeled_types = [t for t, nums in session_labels.items() if nums] # Types labeled in this session
    if len(labeled_types) >= 2:
        # Iterate through combinations of different object types labeled in this session
        for type1, type2 in itertools.combinations(labeled_types, 2):
            masks1 = session_labels.get(type1, [])
            masks2 = session_labels.get(type2, [])
            # Should not be empty based on labeled_types logic, but check anyway
            if not masks1 or not masks2: continue

            # Calculate distance for every pair across the two types
            for mask_num1 in masks1:
                for mask_num2 in masks2:
                    # Validate indices against the stored test_results in the learner
                    if not (0 <= mask_num1 - 1 < len(learner.last_test_results) and \
                            0 <= mask_num2 - 1 < len(learner.last_test_results)):
                         print(f"[Warn] Invalid mask index for distance calc: {mask_num1} or {mask_num2}")
                         continue
                    try:
                        # Retrieve features from the stored results
                        features1 = learner.last_test_results[mask_num1 - 1].get("features")
                        features2 = learner.last_test_results[mask_num2 - 1].get("features")
                        if features1 and features2:
                            c1x = features1.get("centroid_x")
                            c1y = features1.get("centroid_y")
                            c2x = features2.get("centroid_x")
                            c2y = features2.get("centroid_y")
                            # Ensure centroids are valid numbers
                            if None not in [c1x, c1y, c2x, c2y]:
                                distance = math.sqrt((c2x - c1x)**2 + (c2y - c1y)**2)
                                # Create consistent pair key (alphabetical order)
                                pair_key = "-".join(sorted((type1, type2)))
                                learner.add_distance_sample(pair_key, distance)
                                distance_added_count += 1
                            else:
                                print(f"[Warn] Missing centroid data for distance calc between Mask #{mask_num1} & #{mask_num2}")
                        else:
                             print(f"[Warn] Missing feature data for distance calc between Mask #{mask_num1} & #{mask_num2}")
                    except IndexError:
                         # This shouldn't happen if index validation above is correct, but good to have
                         print(f"[Error] Index out of bounds accessing test results for distance: {mask_num1-1} or {mask_num2-1}")
                    except Exception as dist_e:
                        print(f"[Error] Failed to calculate distance between Mask #{mask_num1} & #{mask_num2}: {dist_e}")

    if distance_added_count > 0:
        print(f"Added {distance_added_count} distance sample(s) for this image.")
    else:
        print("No valid pairs labeled on this image for distance calculation.")
    # --- End Distance Calculation ---

    print("--- Finished Interactive Session for this Image ---")
    # Return False if completed normally without early stop request
    return stop_requested
# --- END UPDATED interactive_learning_session ---


# --- run_learning_process (Main workflow logic) ---
def run_learning_process(session_name: Optional[str] = None):
    """
    Encapsulates the main logic for running the feature learning tool.
    Uses interactive mask selection and allows stopping early.
    """
    print("Starting Feature Learning Tool...")
    # --- Configuration ---
    # Adjust these paths and parameters as needed for your environment
    PROJECT_ROOT_ABS = r"D:\Working\BoardDefectChecker" # Example path
    SAMPLE_IMAGE_DIR = r"C:\BoardDefectChecker\images\samples_for_learning" # Example path
    AI_MODEL_DIR = os.path.join(PROJECT_ROOT_ABS, "ai-models")
    CONFIG_OUTPUT_DIR = r"C:\BoardDefectChecker\ai-training-data"
    AI_MODEL_TYPE = "x" # Or "s", depending on your FastSAM model
    OBJECT_TYPES_TO_LEARN = ["bezel", "copper_mark", "stamped_mark"] # Add all types you want to learn
    MIN_TOTAL_SAMPLES_TO_GENERATE = 5 # Minimum labeled masks across all types to generate config
    # Tolerances for calculating feature ranges (%)
    TOLERANCES_BY_TYPE = {
        "bezel": 10.0,
        "copper_mark": 5.0,
        "stamped_mark": 8.0 # Example tolerance for the new type
    }
    DEFAULT_TOLERANCE = 8.0 # Fallback tolerance
    DISTANCE_TOLERANCE_PERCENT = 15.0 # Tolerance for distance ranges
    # --- End Configuration ---

    # --- Get Learning Session Name ---
    if session_name: session_name = session_name.strip()
    if not session_name:
        while True:
            session_name = input("Enter name for this learning session (e.g., BezelPWB_Config_V2): ").strip()
            if session_name: break
            else: print("Session name cannot be empty.")
    print(f"--- Learning Session Name: {session_name} ---")

    # --- Get Desired Evaluation Counts ---
    evaluation_counts: Dict[str, int] = {}
    print("\nEnter the exact number of objects expected during EVALUATION:")
    for obj_type in OBJECT_TYPES_TO_LEARN:
        while True:
            try:
                count_input = input(f"  - Expected count for '{obj_type}' during evaluation: ")
                count = int(count_input)
                if count >= 0:
                    evaluation_counts[obj_type] = count
                    break
                else:
                    print("  Error: Count must be zero or positive.")
            except ValueError:
                print("  Error: Invalid input. Please enter a whole number.")
    print(f"Evaluation counts set: {evaluation_counts}")
    # --- End Get Evaluation Counts ---

    # --- Ask for Overlap Rule Preference ---
    chosen_overlap_mode = "absolute"; chosen_overlap_threshold = 0.05
    while True:
        mode_input = input("Choose overlap rule mode (0=Absolute [No Overlap], 1=Ratio [Use Threshold]): [Default: 0] ").strip()
        if mode_input == '0' or mode_input == '':
            chosen_overlap_mode = "absolute"; print(" -> Using ABSOLUTE overlap mode."); break
        elif mode_input == '1':
            chosen_overlap_mode = "ratio"
            while True:
                 threshold_input = input(f" -> Enter max IoU threshold (0.0-1.0) [Default: {chosen_overlap_threshold:.2f}]: ").strip()
                 if not threshold_input:
                      print(f"    -> Using default threshold: {chosen_overlap_threshold:.2f}"); break
                 try:
                      threshold = float(threshold_input)
                      if 0.0 <= threshold <= 1.0:
                           chosen_overlap_threshold = threshold; print(f"    -> Using threshold: {chosen_overlap_threshold:.3f}"); break
                      else: print("Error: Threshold must be between 0.0 and 1.0.")
                 except ValueError: print("Error: Invalid number for threshold.")
            break
        else: print("Invalid input. Please enter 0 or 1.")
    # --- End Overlap Rule Preference ---

    # --- Initialization ---
    # Checker is only used for feature extraction here, config doesn't matter much
    dummy_config = {"target_objects": {}}
    try: checker = RotationInvariantAOIChecker(config=dummy_config)
    except Exception as e: print(f"[FATAL ERROR] Failed init checker: {e}"); sys.exit(1)

    # Initialize Segmenter
    segmenter = None
    print(f"Attempting load segmenter model from: {AI_MODEL_DIR}")
    try:
        segmenter = SegmenterClass(model_type=AI_MODEL_TYPE, model_path=AI_MODEL_DIR)
        if getattr(segmenter, 'model', None) is None:
            raise RuntimeError("Segmenter model attribute is None after initialization.")
        print("Segmenter loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Failed init segmenter: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Initialize Learner
    learner = StatisticalFeatureLearner()
    # --------------------

    # --- Find Sample Images ---
    if not os.path.isdir(SAMPLE_IMAGE_DIR):
        print(f"\n[ERROR] Sample image directory not found: {SAMPLE_IMAGE_DIR}")
        sample_image_paths = []
    else:
         valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")
         try:
             sample_image_paths = [ os.path.join(SAMPLE_IMAGE_DIR, f) for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(valid_extensions) ]
             print(f"\nFound {len(sample_image_paths)} images in {SAMPLE_IMAGE_DIR}.")
         except Exception as e:
             print(f"[Error] Failed list images: {e}")
             sample_image_paths = []

    # --- Interactive Learning Loop (with early stop check) ---
    if not sample_image_paths:
        print("No sample images found. Cannot proceed with learning.")
    else:
         print("Starting interactive learning loop...")
         for img_path in sample_image_paths:
             # Call the session function which now handles interaction
             stop_early = interactive_learning_session(
                 image_path=img_path,
                 segmenter=segmenter,
                 checker=checker,
                 learner=learner,
                 object_types_to_learn=OBJECT_TYPES_TO_LEARN
             )
             # Check if the user requested to stop early
             if stop_early:
                 print("Exiting learning loop early as requested.")
                 break # Exit the image loop
             print("-" * 50) # Separator between images

         # --- Generate Config (runs after loop finishes OR breaks) ---
         print("\n--- Learning Loop Finished ---")
         print(learner.get_summary()) # Show summary of collected samples
         total_samples = learner.get_total_samples() # Total labeled feature samples
         print(f"Total feature samples labeled across all images: {total_samples}")

         # Check minimum samples BEFORE generating
         if total_samples >= MIN_TOTAL_SAMPLES_TO_GENERATE:
             print(f"\nReached minimum sample count ({total_samples}/{MIN_TOTAL_SAMPLES_TO_GENERATE}). Generating config JSON...")
             # Generate the config using collected data and user settings
             suggested_json = learner.generate_suggested_config(
                 tolerances_by_type=TOLERANCES_BY_TYPE,
                 distance_tolerance_percent=DISTANCE_TOLERANCE_PERCENT,
                 overlap_mode=chosen_overlap_mode,
                 overlap_threshold=chosen_overlap_threshold,
                 default_tolerance_percent=DEFAULT_TOLERANCE,
                 evaluation_counts=evaluation_counts # Pass the desired eval counts
             )

             # --- Save JSON to File ---
             if suggested_json:
                 print("\n------------------- Generated Configuration JSON -------------------")
                 print(suggested_json) # Print JSON to console
                 print("--------------------------------------------------------------------")
                 print("\nPaste the JSON content above into the checker's config string or save to a file.")
                 output_path = ""
                 try:
                     os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
                     # Create a meaningful filename
                     learner_name = learner.__class__.__name__.replace("FeatureLearner","") # e.g., Statistical
                     filename = f"{session_name}_{learner_name}_Config.json"
                     output_path = os.path.join(CONFIG_OUTPUT_DIR, filename)
                     # Write the JSON to the file
                     with open(output_path, 'w', encoding='utf-8') as f:
                         f.write(suggested_json)
                     print(f"\n[SUCCESS] Configuration saved to:\n{output_path}")
                 except Exception as save_err:
                     print(f"\n[ERROR] Failed save file to {output_path}: {save_err}")
                     traceback.print_exc()
             else:
                 print("\n[ERROR] Failed to generate suggested configuration JSON.")
         else:
             # Inform user if not enough samples were collected
             print(f"\nDid not reach minimum sample count ({total_samples}/{MIN_TOTAL_SAMPLES_TO_GENERATE}). No configuration generated.")

    print("\nFeature Learning Tool Finished.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    try:
        run_learning_process()
    except Exception as main_err:
         # Catch any unexpected errors during the main process
         print(f"\n[FATAL ERROR] An error occurred during the learning process: {main_err}")
         traceback.print_exc()
         # Optional: exit with error code
         # sys.exit(1)

