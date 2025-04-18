# modules/feature_learners/feature_learning_tool.py

import os
import sys
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
import json
import traceback
import math
import itertools

# --- Imports from this package ---
try:
    from .base import BaseFeatureLearner
    # Assuming statistical.py exists in the same directory now
    from .statistical import StatisticalFeatureLearner
    from .learning_visualizer import display_for_learning
except ImportError as e_imp:
    print(f"[FATAL ERROR] Could not import base/statistical/visualizer: {e_imp}")
    print("Ensure base.py, statistical.py, and learning_visualizer.py are in the same directory.")
    sys.exit(1)

# --- Imports from other project modules ---
try:
    # Adjust path if necessary based on where you run from
    from modules.rotation_invariant_checking import RotationInvariantAOIChecker
    from modules.ai_models import BezelPWBPositionSegmenter
    SegmenterClass = BezelPWBPositionSegmenter
except ImportError as e:
     print(f"[FATAL ERROR] Failed import Checker/Segmenter: {e}")
     print("Ensure modules/rotation_invariant_checking.py and modules/ai_models.py exist relative to the execution path.")
     sys.exit(1)


def interactive_learning_session(image_path: str,
                                 segmenter: SegmenterClass,
                                 checker: RotationInvariantAOIChecker,
                                 learner: BaseFeatureLearner,
                                 object_types_to_learn: List[str],
                                 max_masks_infer: int = 100,
                                 max_masks_display: int = 100):
    """
    Runs inference, displays results, prompts user for interactive labeling,
    and calculates/stores distances between labeled pairs for a single image.
    (Function content remains unchanged)
    """
    print(f"\n--- Starting Interactive Session for: {os.path.basename(image_path)} ---")
    if not os.path.exists(image_path): print(f"[Error] Image not found: {image_path}"); return
    print("Loading image..."); image = cv2.imread(image_path)
    if image is None: print(f"[Error] Failed read image {image_path}"); return
    print("Running inference..."); masks = []
    try:
        # Assuming run_inference returns a list of numpy arrays or None
        raw_masks = segmenter.run_inference(image)
        if raw_masks is not None and isinstance(raw_masks, (list, np.ndarray)):
             masks = [m for m in raw_masks if isinstance(m, np.ndarray)]
        print(f"Inference complete, got {len(masks)} masks.")
    except Exception as e: print(f"[Error] Inference failed: {e}")
    if not masks: print("[Warning] No valid masks from inference."); return
    if max_masks_infer is not None and len(masks) > max_masks_infer:
        print(f"[Info] Truncating masks from {len(masks)} to {max_masks_infer}")
        masks = masks[:max_masks_infer]
    print("Extracting features..."); test_results = []
    try:
        # Assuming test_masks returns a list of dicts or []
        test_results = checker.test_masks(masks, image.shape[:2], max_masks_to_show=max_masks_display)
    except Exception as e: print(f"[Error] Failed test_masks: {e}"); return
    if not test_results: print("[Warning] No valid test results generated."); return
    learner.set_last_test_results(test_results)
    display_for_learning(image, test_results) # Show visuals before console
    print("\n--- Start Labeling (Console Input) ---")
    max_mask_num = len(test_results); session_labels: Dict[str, List[int]] = {t: [] for t in object_types_to_learn}
    for obj_type in object_types_to_learn:
        while True:
            try:
                prompt = (f"Enter Mask #(s) for '{obj_type}' (1-{max_mask_num}, space-separated), or ENTER to skip: ")
                user_input = input(prompt).strip()
                if not user_input: print(f"Skipping '{obj_type}'."); break
                mask_numbers_str = user_input.split(); mask_numbers = []; valid_input = True
                for num_str in mask_numbers_str:
                    mask_num = int(num_str)
                    if not (1 <= mask_num <= max_mask_num): print(f"Error: Mask #{mask_num} out of range (1-{max_mask_num})."); valid_input = False; break
                    mask_numbers.append(mask_num)
                if not valid_input: continue
                added_count = 0
                for mn in mask_numbers:
                    if learner.add_sample(obj_type, mn): session_labels[obj_type].append(mn); added_count += 1
                print(f"Added {added_count} sample(s) for '{obj_type}'."); break
            except ValueError: print("Invalid input. Please enter space-separated numbers.")
            except Exception as e: print(f"Input error: {e}")
    print("Calculating distances between labeled pairs..."); distance_added_count = 0
    labeled_types = [t for t, nums in session_labels.items() if nums]
    if len(labeled_types) >= 2:
        for type1, type2 in itertools.combinations(labeled_types, 2):
            # Ensure we have features for the masks before calculating distance
            masks1 = session_labels.get(type1, [])
            masks2 = session_labels.get(type2, [])
            if not masks1 or not masks2: continue # Skip if one type wasn't labeled in this session

            for mask_num1 in masks1:
                for mask_num2 in masks2:
                    # Check indices are valid
                    if not (0 <= mask_num1 - 1 < len(learner.last_test_results) and \
                            0 <= mask_num2 - 1 < len(learner.last_test_results)):
                         print(f"[Warn] Invalid mask index for distance calc: {mask_num1} or {mask_num2}")
                         continue
                    try:
                        features1 = learner.last_test_results[mask_num1 - 1].get("features")
                        features2 = learner.last_test_results[mask_num2 - 1].get("features")
                        if features1 and features2:
                            c1x, c1y = features1.get("centroid_x"), features1.get("centroid_y")
                            c2x, c2y = features2.get("centroid_x"), features2.get("centroid_y")
                            if None not in [c1x, c1y, c2x, c2y]:
                                distance = math.sqrt((c2x - c1x)**2 + (c2y - c1y)**2)
                                pair_key = "-".join(sorted((type1, type2))) # Ensure consistent key order
                                learner.add_distance_sample(pair_key, distance); distance_added_count += 1
                            # else: print(f"[Warn] Missing centroid Mask #{mask_num1}/{mask_num2}") # Verbose
                        # else: print(f"[Warn] Missing features Mask #{mask_num1}/{mask_num2}") # Verbose
                    except IndexError:
                         print(f"[Error] Index out of bounds accessing test results for distance: {mask_num1-1} or {mask_num2-1}")
                    except Exception as dist_e: print(f"[Error] Calc distance fail Mask #{mask_num1}/{mask_num2}: {dist_e}")
    if distance_added_count > 0: print(f"Added {distance_added_count} distance sample(s) for this image.")
    else: print("No valid pairs labeled on this image for distance calculation.")
    print("--- Finished Interactive Session for this Image ---")


# --- UPDATED FUNCTION ---
def run_learning_process(session_name: Optional[str] = None):
    """
    Encapsulates the main logic for running the feature learning tool.
    Prompts for session name, desired evaluation counts, and overlap rule preference.
    Saves config including feature ranges, distance constraints, overlap rules,
    and the user-specified evaluation counts.
    """
    print("Starting Feature Learning Tool...")
    # --- Configuration ---
    PROJECT_ROOT_ABS = r"D:\Working\BoardDefectChecker" # Adjust if needed
    SAMPLE_IMAGE_DIR = r"C:\BoardDefectChecker\images\samples_for_learning" # Adjust if needed
    AI_MODEL_DIR = os.path.join(PROJECT_ROOT_ABS, "ai-models")
    CONFIG_OUTPUT_DIR = r"C:\BoardDefectChecker\ai-training-data"
    AI_MODEL_TYPE = "x"
    OBJECT_TYPES_TO_LEARN = ["bezel", "copper_mark"] # Types user will label
    MIN_TOTAL_SAMPLES_TO_GENERATE = 5 # Min *labeled samples* to generate config
    TOLERANCES_BY_TYPE = { "bezel": 10.0, "copper_mark": 5.0 } # % tolerance for feature ranges
    DEFAULT_TOLERANCE = 8.0 # % tolerance if type not in above dict
    DISTANCE_TOLERANCE_PERCENT = 15.0 # % tolerance for distance ranges
    # --- End Configuration ---

    # --- Get Learning Session Name ---
    if session_name: session_name = session_name.strip()
    if not session_name:
        while True:
            session_name = input("Enter name for this learning session (e.g., BezelPWB_Config_V1): ").strip()
            if session_name: break
            else: print("Session name cannot be empty.")
    print(f"--- Learning Session Name: {session_name} ---")

    # --- NEW: Get Desired Evaluation Counts ---
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
    chosen_overlap_mode = "absolute" # Default
    chosen_overlap_threshold = 0.05 # Default threshold if ratio is chosen
    while True:
        mode_input = input("Choose overlap rule mode (0 for Absolute [No Overlap], 1 for Ratio [Use Threshold]): [Default: 0] ").strip()
        if mode_input == '0' or mode_input == '':
            chosen_overlap_mode = "absolute"
            print(" -> Using ABSOLUTE overlap mode (no overlap allowed).")
            break
        elif mode_input == '1':
            chosen_overlap_mode = "ratio"
            while True:
                 threshold_input = input(f" -> Using RATIO overlap mode. Enter max allowed IoU threshold (0.0 to 1.0) [Default: {chosen_overlap_threshold:.2f}]: ").strip()
                 if not threshold_input:
                      print(f"    -> Using default threshold: {chosen_overlap_threshold:.2f}")
                      break # Use default threshold
                 try:
                      threshold = float(threshold_input)
                      if 0.0 <= threshold <= 1.0:
                           chosen_overlap_threshold = threshold
                           print(f"    -> Using threshold: {chosen_overlap_threshold:.3f}")
                           break # Valid threshold entered
                      else:
                           print("Error: Threshold must be between 0.0 and 1.0.")
                 except ValueError:
                      print("Error: Invalid number for threshold.")
            break # Exit outer loop once mode and threshold (if needed) are set
        else:
            print("Invalid input. Please enter 0 or 1.")
    # --- End Overlap Rule Preference ---

    # --- Initialization ---
    # Use a minimal config for the checker during learning (only feature extraction needed)
    dummy_config = {"target_objects": {}}
    try: checker = RotationInvariantAOIChecker(config=dummy_config)
    except Exception as e: print(f"[FATAL ERROR] Failed init checker: {e}"); sys.exit(1)
    segmenter = None
    print(f"Attempting load segmenter model from: {AI_MODEL_DIR}")
    try:
        segmenter = SegmenterClass(model_type=AI_MODEL_TYPE, model_path=AI_MODEL_DIR)
        if getattr(segmenter, 'model', None) is None: raise RuntimeError("Segmenter model failed load.")
        print("Segmenter loaded successfully.")
    except Exception as e: print(f"[FATAL ERROR] Failed init segmenter: {e}"); traceback.print_exc(); sys.exit(1)
    learner = StatisticalFeatureLearner() # Use the statistical learner
    # --------------------

    # --- Find Sample Images ---
    if not os.path.isdir(SAMPLE_IMAGE_DIR): print(f"\n[ERROR] Sample image directory not found: {SAMPLE_IMAGE_DIR}"); sample_image_paths = []
    else:
         valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")
         try: sample_image_paths = [ os.path.join(SAMPLE_IMAGE_DIR, f) for f in os.listdir(SAMPLE_IMAGE_DIR) if f.lower().endswith(valid_extensions) ]
         except Exception as e: print(f"[Error] Failed list images: {e}"); sample_image_paths = []

    # --- Interactive Learning Loop ---
    if not sample_image_paths: print(f"\nNo sample images found in {SAMPLE_IMAGE_DIR}. Cannot proceed.")
    else:
         print(f"\nFound {len(sample_image_paths)} images. Starting learning loop...")
         for img_path in sample_image_paths:
             interactive_learning_session(
                 image_path=img_path,
                 segmenter=segmenter,
                 checker=checker,
                 learner=learner,
                 object_types_to_learn=OBJECT_TYPES_TO_LEARN
            )
             print("-" * 50) # Separator between images

         # --- Generate Config after Labeling Loop ---
         print("\n--- Learning Loop Finished ---")
         print(learner.get_summary())
         total_samples = learner.get_total_samples() # Total labeled feature samples
         print(f"Total feature samples labeled: {total_samples}")

         if total_samples >= MIN_TOTAL_SAMPLES_TO_GENERATE:
             print(f"\nReached minimum sample count ({total_samples}/{MIN_TOTAL_SAMPLES_TO_GENERATE}). Generating config JSON...")
             # --- Pass user choices AND EVALUATION COUNTS to generator ---
             suggested_json = learner.generate_suggested_config(
                 tolerances_by_type=TOLERANCES_BY_TYPE,
                 distance_tolerance_percent=DISTANCE_TOLERANCE_PERCENT,
                 overlap_mode=chosen_overlap_mode,
                 overlap_threshold=chosen_overlap_threshold,
                 default_tolerance_percent=DEFAULT_TOLERANCE,
                 evaluation_counts=evaluation_counts # Pass the desired counts
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
                     learner_name = learner.__class__.__name__.replace("FeatureLearner","") # e.g., Statistical
                     filename = f"{session_name}_{learner_name}_Config.json" # Changed filename
                     output_path = os.path.join(CONFIG_OUTPUT_DIR, filename)
                     with open(output_path, 'w', encoding='utf-8') as f: f.write(suggested_json)
                     print(f"\n[SUCCESS] Configuration saved to:\n{output_path}")
                 except Exception as save_err: print(f"\n[ERROR] Failed save file to {output_path}: {save_err}"); traceback.print_exc()
             else: print("\n[ERROR] Failed to generate suggested configuration JSON.")
         else: print(f"\nDid not reach minimum sample count ({total_samples}/{MIN_TOTAL_SAMPLES_TO_GENERATE}). No configuration generated.")

    print("\nFeature Learning Tool Finished.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Add basic error handling for the main execution
    try:
        run_learning_process()
    except Exception as main_err:
         print(f"\n[FATAL ERROR] An error occurred during the learning process: {main_err}")
         traceback.print_exc()

