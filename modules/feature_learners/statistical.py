# modules/feature_learners/statistical.py

import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional, Set
import math
import traceback

# Import the base class from the same directory/package
try:
    from .base import BaseFeatureLearner
except ImportError:
    # Fallback for running directly or if structure differs
    print("[Warning][StatisticalLearner] Could not import BaseFeatureLearner using relative path. Trying direct import.")
    try:
        from base import BaseFeatureLearner
    except ImportError:
        raise ImportError("Could not find BaseFeatureLearner. Ensure base.py is accessible.")


class StatisticalFeatureLearner(BaseFeatureLearner):
    """
    Learns feature ranges and distance constraints based on statistical
    analysis (min, max, mean, stddev) of collected samples.
    Generates overlap rules based on observed pairs.
    Includes both total labeled samples and user-specified evaluation counts in the config.
    """

    def __init__(self):
        """Initializes the statistical learner."""
        super().__init__() # Call base class constructor
        print(f"[{self.__class__.__name__}] Initialized.")

    # --- UPDATED generate_suggested_config METHOD ---
    def generate_suggested_config(self,
                                  tolerances_by_type: Dict[str, float],
                                  distance_tolerance_percent: float,
                                  overlap_mode: str,
                                  overlap_threshold: float,
                                  default_tolerance_percent: float,
                                  evaluation_counts: Dict[str, int] # <-- User-specified counts for eval
                                 ) -> Optional[str]:
        """
        Generates a suggested configuration JSON string based on collected samples,
        tolerances, overlap preferences, and desired evaluation counts. Includes
        both total labeled samples and the evaluation count in the output.

        Args:
            tolerances_by_type: Dictionary mapping object type to tolerance percentage.
            distance_tolerance_percent: Tolerance percentage for distance ranges.
            overlap_mode: 'absolute' or 'ratio'.
            overlap_threshold: IoU threshold if overlap_mode is 'ratio'.
            default_tolerance_percent: Fallback tolerance percentage.
            evaluation_counts: Dictionary mapping object type to the desired count for evaluation.

        Returns:
            A JSON string representing the suggested configuration, or None on error.
        """
        print(f"[{self.__class__.__name__}] Generating configuration...")
        if not self.learned_features:
            print("[Error] Cannot generate config: No feature samples collected.")
            return None

        final_config = {"target_objects": {}, "distance_constraints": {}, "overlap_rules": []}

        # --- 1. Process Features ---
        print("Processing learned features...")
        for obj_type, samples in self.learned_features.items():
            if not samples: continue
            num_labeled_samples = len(samples) # Get the actual number of labeled samples
            print(f"  - Processing '{obj_type}' ({num_labeled_samples} samples)")
            stats = {} # To store min, max, mean, stddev for each feature
            feature_ranges = {} # To store calculated [min_bound, max_bound]

            for key in self.feature_keys:
                values = [s.get(key, 0.0) for s in samples if s.get(key) is not None] # Get valid values
                if not values:
                    print(f"    [Warning] No valid values found for feature '{key}' in '{obj_type}'. Skipping range calculation.")
                    stats[key] = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'stddev': 0.0, 'count': 0}
                    feature_ranges[key] = [0.0, 0.0] # Default range
                    continue

                # Calculate basic statistics
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                mean_val = float(np.mean(values))
                stddev_val = float(np.std(values))
                stats[key] = {'min': min_val, 'max': max_val, 'mean': mean_val, 'stddev': stddev_val, 'count': len(values)}

                # Calculate tolerance based on type or default
                tolerance_percent = tolerances_by_type.get(obj_type, default_tolerance_percent)
                tolerance_value = (tolerance_percent / 100.0) * max_val if max_val > 0 else 0.0 # Tolerance based on max value

                # Calculate bounds with tolerance
                min_bound = max(0.0, min_val - tolerance_value) # Ensure non-negative
                max_bound = max_val + tolerance_value

                # Special handling for aspect ratio (must be <= 1)
                if key == "aspect_ratio":
                    min_bound = max(0.0, min(1.0, min_bound))
                    max_bound = max(0.0, min(1.0, max_bound))

                feature_ranges[key] = [round(min_bound, 3), round(max_bound, 3)] # Store as list [min, max]

            # Get the user-specified evaluation count for this type
            eval_count = evaluation_counts.get(obj_type, 0) # Default to 0 if not specified by user
            print(f"    -> Using Evaluation Count for '{obj_type}': {eval_count}")
            print(f"    -> Total Samples Labeled for '{obj_type}': {num_labeled_samples}")

            # Add BOTH counts to the configuration under different keys
            final_config["target_objects"][obj_type] = {
                "expected_evaluation_count": eval_count, # Count for evaluation check
                "total_samples_labeled": num_labeled_samples, # Informational count
                "feature_ranges": feature_ranges
                # Optionally add statistics here if needed for review
                # "statistics": stats
            }
            print(f"    -> Feature ranges for '{obj_type}': {feature_ranges}")


        # --- 2. Process Distances (Unchanged) ---
        print("\nProcessing learned distances...")
        if not self.learned_distances:
             print("  - No distance samples collected.")
        else:
            for pair_key, distances in self.learned_distances.items():
                if not distances: continue
                print(f"  - Processing distance for pair '{pair_key}' ({len(distances)} samples)")
                min_dist = float(np.min(distances))
                max_dist = float(np.max(distances))
                mean_dist = float(np.mean(distances))
                stddev_dist = float(np.std(distances))

                # Calculate tolerance
                tolerance_value = (distance_tolerance_percent / 100.0) * max_dist if max_dist > 0 else 0.0

                # Calculate bounds
                min_bound = max(0.0, min_dist - tolerance_value)
                max_bound = max_dist + tolerance_value

                final_config["distance_constraints"][pair_key] = {
                    "range": [round(min_bound, 3), round(max_bound, 3)],
                    "mean": round(mean_dist, 3),
                    "stddev": round(stddev_dist, 3),
                    "count": len(distances)
                }
                print(f"    -> Distance range for '{pair_key}': {[round(min_bound, 3), round(max_bound, 3)]}")


        # --- 3. Generate Overlap Rules (Unchanged) ---
        print("\nGenerating overlap rules...")
        if not self.observed_pairs:
             print("  - No object pairs observed together. No overlap rules generated.")
        else:
            # Generate rules for all pairs observed together on the same image
            for pair_key in self.observed_pairs:
                 obj_types = pair_key.split('-')
                 if len(obj_types) == 2:
                     rule = {"objects": obj_types, "mode": overlap_mode}
                     if overlap_mode == "ratio":
                         rule["threshold"] = overlap_threshold
                     final_config["overlap_rules"].append(rule)
                     print(f"  - Added overlap rule for {obj_types} (Mode: {overlap_mode}{f', Threshold: {overlap_threshold}' if overlap_mode == 'ratio' else ''})")
            # Also add self-overlap rules for each type learned
            for obj_type in self.learned_features.keys():
                 rule = {"objects": [obj_type, obj_type], "mode": overlap_mode}
                 if overlap_mode == "ratio":
                      rule["threshold"] = overlap_threshold
                 final_config["overlap_rules"].append(rule)
                 print(f"  - Added self-overlap rule for ['{obj_type}', '{obj_type}'] (Mode: {overlap_mode}{f', Threshold: {overlap_threshold}' if overlap_mode == 'ratio' else ''})")


        # --- 4. Finalize and Format JSON ---
        try:
            json_output = json.dumps(final_config, indent=2) # Pretty print
            print(f"\n[{self.__class__.__name__}] Configuration generation successful.")
            return json_output
        except Exception as e:
            print(f"[Error][{self.__class__.__name__}] Failed to serialize configuration to JSON: {e}")
            traceback.print_exc()
            return None

