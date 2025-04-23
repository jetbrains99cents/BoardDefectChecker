# modules/feature_learners/feature_learning_tool.py
import os
import sys
import cv2
import numpy as np
# Removed unused: Any, Set, Tuple
from typing import List, Dict, Optional
# Removed unused: import json
import traceback
import math  # Needed for sin, cos, radians, atan2, sqrt
import itertools

# Imports from this package
try:
    from .base import BaseFeatureLearner
    from .statistical import StatisticalFeatureLearner
    from .learning_visualizer import select_masks_interactive
# Handling potential NameError if imports fail (addresses IDE warning)
except ImportError as e_imp:
    print(f"[FATAL ERROR] Could not import base/statistical/visualizer: {e_imp}")
    print("Ensure base.py, statistical.py, and learning_visualizer.py are in the same directory.")


    # Define dummy classes/functions if needed for script to load partially
    class BaseFeatureLearner:
        pass


    class StatisticalFeatureLearner(BaseFeatureLearner):
        pass


    def select_masks_interactive(*args, **kwargs):
        return {}, False


    # Exit if core components missing
    sys.exit(1)

# Imports from other project modules
try:
    from modules.rotation_invariant_checking import RotationInvariantAOIChecker
    from modules.ai_models import BezelPWBPositionSegmenter

    SegmenterClass = BezelPWBPositionSegmenter
# Handling potential NameError if imports fail
except ImportError as e:
    print(f"[FATAL ERROR] Failed import Checker/Segmenter: {e}")
    print(
        "Ensure modules/rotation_invariant_checking.py and modules/ai_models.py exist relative to the execution path.")


    # Define dummy classes if needed
    class RotationInvariantAOIChecker:
        pass


    class BezelPWBPositionSegmenter:
        pass


    SegmenterClass = BezelPWBPositionSegmenter  # Assign dummy
    # Exit if core components missing
    sys.exit(1)


# --- UPDATED interactive_learning_session (Syntax Fixes v2) ---
def interactive_learning_session(image_path: str,
                                 segmenter: SegmenterClass,
                                 checker: RotationInvariantAOIChecker,
                                 learner: BaseFeatureLearner,
                                 object_types_to_learn: List[str],
                                 max_masks_infer: int = 100,
                                 max_masks_display: int = 100) -> bool:
    """
    Runs inference, feature extraction, interactive selection, adds feature/distance samples,
    and calculates/adds relative position samples. Includes syntax fixes v2.
    Returns True if the user requested to stop early.
    """
    print(f"\n--- Starting Interactive Session for: {os.path.basename(image_path)} ---")
    stop_requested = False

    # --- Load Image ---
    if not os.path.exists(image_path):
        print(f"[Error] Image not found: {image_path}")
        return stop_requested
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[Error] Failed read image {image_path}")
        return stop_requested

    # --- Run Inference ---
    print("Running inference...")
    masks = []
    try:
        raw_masks = segmenter.run_inference(image)
        if raw_masks is not None and isinstance(raw_masks, (list, np.ndarray)):
            masks = [m for m in raw_masks if isinstance(m, np.ndarray) and m.ndim == 2]
        print(f"Inference complete, got {len(masks)} valid masks.")
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        traceback.print_exc()
        return stop_requested
    if not masks:
        print("[Warning] No valid masks found. Skipping labeling.")
        return stop_requested
    if max_masks_infer is not None and len(masks) > max_masks_infer:
        masks = masks[:max_masks_infer]

    # --- Extract Features (Includes angle) ---
    print("Extracting features (incl. angle) for detected masks...")
    test_results = []
    try:
        test_results = checker.test_masks(masks, image.shape[:2], max_masks_to_show=len(masks))
    except Exception as e:
        print(f"[Error] Failed feature extraction: {e}")
        return stop_requested
    if not test_results:
        print("[Warning] No valid features extracted. Skipping labeling.")
        return stop_requested
    learner.set_last_test_results(test_results)

    # --- Launch Interactive Selection ---
    print("Launching interactive selection window...")
    session_labels, stop_requested = select_masks_interactive(
        image=image, test_results=test_results, object_types_to_learn=object_types_to_learn)

    # --- Process Selections (Feature Samples) ---
    print("\nAdding selected feature samples to learner...")
    total_added_features = 0
    for obj_type, mask_numbers in session_labels.items():
        added_count = 0
        for mn in mask_numbers:
            if learner.add_sample(obj_type, mn):
                added_count += 1
        if added_count > 0:
            print(f"Added {added_count} feature sample(s) for '{obj_type}'.")
            total_added_features += added_count
    if total_added_features == 0 and not stop_requested:
        print("No valid feature samples selected in this session.")

    # --- Calculate Distances AND Relative Positions ---
    print("Calculating distances and relative positions between labeled pairs...")
    distance_added_count = 0
    relpos_added_count = 0
    labeled_types = [t for t, nums in session_labels.items() if nums]

    # --- Calculations for DIFFERENT types ---
    if len(labeled_types) >= 2:
        for type1, type2 in itertools.combinations(labeled_types, 2):
            masks1 = session_labels.get(type1, [])
            masks2 = session_labels.get(type2, [])
            if not masks1 or not masks2:
                continue
            pair_key = "-".join(sorted((type1, type2)))

            for mn1 in masks1:
                for mn2 in masks2:
                    idx1 = mn1 - 1
                    idx2 = mn2 - 1
                    if not (0 <= idx1 < len(learner.last_test_results) and 0 <= idx2 < len(learner.last_test_results)):
                        print(f"[Warn] Invalid mask index for calc: {mn1} or {mn2}")
                        continue

                    # *** Corrected try...except block structure ***
                    try:
                        features1 = learner.last_test_results[idx1].get("features")
                        features2 = learner.last_test_results[idx2].get("features")
                        if not features1 or not features2:
                            print(f"[Warn] Missing features for calc: {mn1} or {mn2}")
                            continue  # Skip this pair

                        c1x = features1.get("centroid_x")
                        c1y = features1.get("centroid_y")
                        c2x = features2.get("centroid_x")
                        c2y = features2.get("centroid_y")
                        angle1_deg = features1.get("angle")

                        # Calculate Distance
                        if None not in [c1x, c1y, c2x, c2y]:
                            distance = math.sqrt((c2x - c1x) ** 2 + (c2y - c1y) ** 2)
                            learner.add_distance_sample(pair_key, distance)
                            distance_added_count += 1

                        # Calculate Relative Position
                        if None not in [c1x, c1y, c2x, c2y, angle1_deg]:
                            dx = c2x - c1x
                            dy = c2y - c1y
                            angle1_rad = math.radians(angle1_deg)
                            cos_a = math.cos(-angle1_rad)
                            sin_a = math.sin(-angle1_rad)
                            dx_rel = dx * cos_a - dy * sin_a
                            dy_rel = dx * sin_a + dy * cos_a
                            if isinstance(learner, StatisticalFeatureLearner):
                                learner.add_relative_position_sample(pair_key, dx_rel, dy_rel)
                                relpos_added_count += 1

                    except Exception as calc_e:
                        # Correctly placed except block
                        print(f"[Error] Failed calc between Mask #{mn1} & #{mn2}: {calc_e}")
                    # End of try...except block

    # --- Calculations for SAME type ---
    for obj_type in labeled_types:
        mask_numbers = session_labels.get(obj_type, [])
        if len(mask_numbers) >= 2:
            pair_key = f"{obj_type}-{obj_type}"
            for i in range(len(mask_numbers)):
                for j in range(i + 1, len(mask_numbers)):
                    mn1 = mask_numbers[i]
                    mn2 = mask_numbers[j]
                    idx1 = mn1 - 1
                    idx2 = mn2 - 1
                    if not (0 <= idx1 < len(learner.last_test_results) and 0 <= idx2 < len(learner.last_test_results)):
                        continue

                    # *** Corrected try...except block structure ***
                    try:
                        features1 = learner.last_test_results[idx1].get("features")
                        features2 = learner.last_test_results[idx2].get("features")
                        if not features1 or not features2:
                            continue  # Skip if features missing

                        c1x = features1.get("centroid_x")
                        c1y = features1.get("centroid_y")
                        c2x = features2.get("centroid_x")
                        c2y = features2.get("centroid_y")
                        angle1_deg = features1.get("angle")

                        if None not in [c1x, c1y, c2x, c2y, angle1_deg]:
                            dx = c2x - c1x
                            dy = c2y - c1y
                            angle1_rad = math.radians(angle1_deg)
                            cos_a = math.cos(-angle1_rad)
                            sin_a = math.sin(-angle1_rad)
                            dx_rel = dx * cos_a - dy * sin_a
                            dy_rel = dx * sin_a + dy * cos_a
                            if isinstance(learner, StatisticalFeatureLearner):
                                learner.add_relative_position_sample(pair_key, dx_rel, dy_rel)
                                relpos_added_count += 1

                    except Exception as calc_e:
                        # Correctly placed except block
                        print(f"[Error] Failed same-type calc between Mask #{mn1} & #{mn2}: {calc_e}")
                    # End of try...except block

    if distance_added_count > 0:
        print(f"Added {distance_added_count} distance sample(s).")
    if relpos_added_count > 0:
        print(f"Added {relpos_added_count} relative position sample(s).")
    if distance_added_count == 0 and relpos_added_count == 0:
        print("No valid pairs labeled for distance/relpos calculation.")

    print("--- Finished Interactive Session for this Image ---")
    return stop_requested


# --- END UPDATED interactive_learning_session ---


# --- run_learning_process (Removed trailing semicolons) ---
def run_learning_process(session_name: Optional[str] = None):
    """Encapsulates the main logic for running the feature learning tool."""
    print("Starting Feature Learning Tool...")
    # --- Configuration ---
    PROJECT_ROOT_ABS = r"D:\Working\BoardDefectChecker"
    SAMPLE_IMAGE_DIR = r"C:\BoardDefectChecker\images\samples_for_learning"
    AI_MODEL_DIR = os.path.join(PROJECT_ROOT_ABS, "ai-models")
    CONFIG_OUTPUT_DIR = r"C:\BoardDefectChecker\ai-training-data"
    AI_MODEL_TYPE = "x"
    OBJECT_TYPES_TO_LEARN = ["bezel", "copper_mark", "stamped_mark"]
    MIN_TOTAL_SAMPLES_TO_GENERATE = 5
    TOLERANCES_BY_TYPE = {"bezel": 10.0, "copper_mark": 5.0, "stamped_mark": 8.0}
    DEFAULT_TOLERANCE = 8.0
    DISTANCE_TOLERANCE_PERCENT = 15.0
    RELPOS_TOLERANCE_STDEVS = 3.0
    # --- End Configuration ---

    # --- Get Session Name ---
    if session_name:
        session_name = session_name.strip()
    if not session_name:
        while True:
            session_name = input("Enter name for this learning session: ").strip()
            if session_name:
                break
            else:
                print("Session name cannot be empty.")
    print(f"--- Learning Session Name: {session_name} ---")

    # --- Get Desired Evaluation Counts ---
    evaluation_counts: Dict[str, int] = {}
    print("\nEnter the exact number of objects expected during EVALUATION:")
    for obj_type in OBJECT_TYPES_TO_LEARN:
        while True:
            try:
                count_str = input(f"  - Expected count for '{obj_type}': ")
                count = int(count_str)
                if count >= 0:
                    evaluation_counts[obj_type] = count
                    break
                else:
                    print("  Error: Count must be zero or positive.")
            except ValueError:
                print("  Error: Invalid input. Please enter a whole number.")
    print(f"Evaluation counts set: {evaluation_counts}")

    # --- Ask for Overlap Rule Preference ---
    chosen_overlap_mode = "absolute"
    chosen_overlap_threshold = 0.05
    while True:
        mode_input = input("Choose overlap rule mode (0=Absolute, 1=Ratio): [Default: 0] ").strip()
        if mode_input == '0' or mode_input == '':
            print(" -> Using ABSOLUTE overlap mode.")
            break
        elif mode_input == '1':
            chosen_overlap_mode = "ratio"
            while True:
                threshold_input = input(
                    f" -> Enter max IoU threshold (0.0-1.0) [Default: {chosen_overlap_threshold:.2f}]: ").strip()
                if not threshold_input:
                    print(f"    -> Using default threshold: {chosen_overlap_threshold:.2f}")
                    break
                try:
                    threshold = float(threshold_input)
                    if 0.0 <= threshold <= 1.0:
                        chosen_overlap_threshold = threshold
                        print(f"    -> Using threshold: {chosen_overlap_threshold:.3f}")
                        break
                    else:
                        print("Error: Threshold must be between 0.0 and 1.0.")
                except ValueError:
                    print("Error: Invalid number for threshold.")
            break
        else:
            print("Invalid input. Please enter 0 or 1.")

    # --- Initialization ---
    dummy_config = {"target_objects": {}}
    checker = None
    segmenter = None
    try:
        checker = RotationInvariantAOIChecker(config=dummy_config)
    except Exception as e:
        print(f"[FATAL ERROR] Failed init checker: {e}")
        sys.exit(1)
    print(f"Attempting load segmenter model from: {AI_MODEL_DIR}")
    try:
        segmenter = SegmenterClass(model_type=AI_MODEL_TYPE, model_path=AI_MODEL_DIR)
        if getattr(segmenter, 'model', None) is None:
            raise RuntimeError("Segmenter model attribute is None.")
        print("Segmenter loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Failed init segmenter: {e}")
        traceback.print_exc()
        sys.exit(1)
    learner = StatisticalFeatureLearner()

    # --- Find Sample Images ---
    if not os.path.isdir(SAMPLE_IMAGE_DIR):
        print(f"\n[ERROR] Sample dir not found: {SAMPLE_IMAGE_DIR}")
        sample_image_paths = []
    else:
        valid_extensions = (".bmp", ".png", ".jpg", ".jpeg")
        try:
            sample_image_paths = [os.path.join(SAMPLE_IMAGE_DIR, f) for f in os.listdir(SAMPLE_IMAGE_DIR) if
                                  f.lower().endswith(valid_extensions)]
            print(f"\nFound {len(sample_image_paths)} images.")
        except Exception as e:
            print(f"[Error] Failed list images: {e}")
            sample_image_paths = []

    # --- Interactive Learning Loop ---
    if not sample_image_paths:
        print("No sample images found.")
    else:
        print("Starting interactive learning loop...")
        for img_path in sample_image_paths:
            stop_early = interactive_learning_session(
                image_path=img_path, segmenter=segmenter, checker=checker, learner=learner,
                object_types_to_learn=OBJECT_TYPES_TO_LEARN)
            if stop_early:
                print("Exiting learning loop early as requested.")
                break
            print("-" * 50)

        # --- Generate Config ---
        print("\n--- Learning Loop Finished ---")
        print(learner.get_summary())
        total_samples = learner.get_total_samples()
        print(f"Total feature samples labeled: {total_samples}")
        if total_samples >= MIN_TOTAL_SAMPLES_TO_GENERATE:
            print(f"\nGenerating config JSON (RelPos Tolerance: {RELPOS_TOLERANCE_STDEVS} std dev)...")
            suggested_json = learner.generate_suggested_config(
                tolerances_by_type=TOLERANCES_BY_TYPE,
                distance_tolerance_percent=DISTANCE_TOLERANCE_PERCENT,
                relpos_tolerance_stdevs=RELPOS_TOLERANCE_STDEVS,
                overlap_mode=chosen_overlap_mode,
                overlap_threshold=chosen_overlap_threshold,
                default_tolerance_percent=DEFAULT_TOLERANCE,
                evaluation_counts=evaluation_counts)

            # --- Save JSON ---
            if suggested_json:
                print("\n------------------- Generated Configuration JSON -------------------")
                print(suggested_json)
                print("--------------------------------------------------------------------")
                output_path = ""
                try:
                    os.makedirs(CONFIG_OUTPUT_DIR, exist_ok=True)
                    learner_name = learner.__class__.__name__.replace("FeatureLearner", "")
                    filename = f"{session_name}_{learner_name}_Config.json"
                    output_path = os.path.join(CONFIG_OUTPUT_DIR, filename)
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(suggested_json)
                    print(f"\n[SUCCESS] Configuration saved to:\n{output_path}")
                except Exception as save_err:
                    print(f"\n[ERROR] Failed save file to {output_path}: {save_err}")
                    traceback.print_exc()
            else:
                print("\n[ERROR] Failed to generate suggested configuration JSON.")
        else:
            print(
                f"\nDid not reach minimum sample count ({total_samples}/{MIN_TOTAL_SAMPLES_TO_GENERATE}). No config generated.")

    print("\nFeature Learning Tool Finished.")


# --- Main Execution Guard ---
if __name__ == "__main__":
    try:
        run_learning_process()
    except Exception as main_err:
        print(f"\n[FATAL ERROR] An error occurred: {main_err}")
        traceback.print_exc()
