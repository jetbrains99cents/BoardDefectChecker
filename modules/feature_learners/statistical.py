# modules/feature_learners/statistical.py
import numpy as np
import json
from typing import List, Dict, Tuple, Any, Optional, Set # Keep Set if used by Base
import math
import traceback

try: from .base import BaseFeatureLearner
except ImportError:
    try: from base import BaseFeatureLearner
    except ImportError: raise ImportError("Could not find BaseFeatureLearner.")

class StatisticalFeatureLearner(BaseFeatureLearner):
    """
    Learns feature ranges, distance constraints, and relative position constraints
    based on statistical analysis of collected samples. Includes debug prints.
    """

    def __init__(self):
        """Initializes the statistical learner."""
        super().__init__()
        self.learned_relative_positions: Dict[str, List[Tuple[float, float]]] = {}
        print(f"[{self.__class__.__name__}] Initialized.")

    def add_relative_position_sample(self, pair_key: str, dx_relative: float, dy_relative: float):
        """Adds a relative position sample (dx', dy') for a given pair key."""
        # (Implementation unchanged from statistical_relative_pos)
        if not isinstance(dx_relative, (float, int)) or not isinstance(dy_relative, (float, int)):
             print(f"[Warning][RelPos] Invalid relative position value for '{pair_key}': ({dx_relative}, {dy_relative}). Skipping."); return
        type1, type2 = pair_key.split('-')
        sorted_pair_key = "-".join(sorted((type1, type2)))
        if sorted_pair_key not in self.learned_relative_positions:
            self.learned_relative_positions[sorted_pair_key] = []
        self.learned_relative_positions[sorted_pair_key].append((float(dx_relative), float(dy_relative)))

    # --- UPDATED generate_suggested_config METHOD (with More Debugging) ---
    def generate_suggested_config(self,
                                  tolerances_by_type: Dict[str, float],
                                  distance_tolerance_percent: float,
                                  relpos_tolerance_stdevs: float,
                                  overlap_mode: str,
                                  overlap_threshold: float,
                                  default_tolerance_percent: float,
                                  evaluation_counts: Dict[str, int]
                                 ) -> Optional[str]:
        """
        Generates a suggested configuration JSON string including feature ranges,
        distance constraints, relative position constraints, and overlap rules.
        Includes debugging for feature range calculation.
        """
        print(f"[{self.__class__.__name__}] Generating configuration...")
        if not self.learned_features:
            print("[Error] Cannot generate config: No feature samples collected.")
            return None

        final_config = {
            "target_objects": {}, "distance_constraints": {},
            "relative_position_constraints": {}, "overlap_rules": []
        }

        # --- 1. Process Features ---
        print("Processing learned features...")
        for obj_type, samples in self.learned_features.items():
            if not samples:
                print(f"  - No samples found for '{obj_type}'. Skipping.")
                continue

            num_labeled_samples = len(samples)
            print(f"  - Processing '{obj_type}' ({num_labeled_samples} samples)")
            if num_labeled_samples > 0:
                first_sample_dict = samples[0]
                print(f"    [DEBUG] First sample dict for '{obj_type}': {first_sample_dict}")
                print(f"    [DEBUG] Keys in first sample: {list(first_sample_dict.keys())}")

            stats = {}
            # Initialize feature_ranges dict HERE for the current obj_type
            feature_ranges = {}

            keys_to_process = self.feature_keys
            print(f"    [DEBUG] Expecting feature keys: {keys_to_process}")

            for key in keys_to_process:
                values = [s.get(key) for s in samples]
                valid_values = [v for v in values if v is not None]

                print(f"      [DEBUG] Valid (non-None) values count for '{key}': {len(valid_values)}")
                if len(valid_values) > 0:
                     print(f"      [DEBUG] First few valid values for '{key}': {valid_values[:5]}")

                if not valid_values:
                    print(f"    [Warning] No valid (non-None) values found for feature '{key}' in '{obj_type}'. Setting range to [0.0, 0.0].")
                    stats[key] = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'stddev': 0.0, 'count': 0}
                    feature_ranges[key] = [0.0, 0.0] # Assign default range for this key
                    continue

                try:
                    numeric_values = [float(v) for v in valid_values]
                except (ValueError, TypeError) as e:
                    print(f"    [Error] Could not convert values to float for feature '{key}' in '{obj_type}': {e}. Skipping range calculation.")
                    stats[key] = {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'stddev': 0.0, 'count': 0}
                    feature_ranges[key] = [0.0, 0.0] # Assign default range for this key
                    continue

                min_val = float(np.min(numeric_values)); max_val = float(np.max(numeric_values))
                mean_val = float(np.mean(numeric_values)); stddev_val = float(np.std(numeric_values))
                stats[key] = {'min': min_val, 'max': max_val, 'mean': mean_val, 'stddev': stddev_val, 'count': len(numeric_values)}

                tolerance_percent = tolerances_by_type.get(obj_type, default_tolerance_percent)
                tolerance_value = (tolerance_percent / 100.0) * max_val if max_val > 0 else 0.0
                print(f"      [DEBUG] Stats for '{key}': Min={min_val:.2f}, Max={max_val:.2f}, Mean={mean_val:.2f}, StdDev={stddev_val:.2f}, TolVal={tolerance_value:.2f}")

                min_bound = max(0.0, min_val - tolerance_value); max_bound = max_val + tolerance_value
                if key == "aspect_ratio":
                    min_bound = max(0.0, min(1.0, min_bound)); max_bound = max(0.0, min(1.0, max_bound))

                # Assign calculated range to the dict for the current key
                feature_ranges[key] = [round(min_bound, 3), round(max_bound, 3)]
                print(f"      [DEBUG] Calculated range for '{key}': {feature_ranges[key]}")

            # --- *** ADDED DEBUG PRINT *** ---
            # Print the complete feature_ranges dictionary for this object type
            # just before it's assigned to the final configuration.
            print(f"    [FINAL DEBUG] Feature ranges for '{obj_type}' before assignment: {feature_ranges}")
            # --- *** END ADDED DEBUG PRINT *** ---

            # Add to final config
            eval_count = evaluation_counts.get(obj_type, 0)
            final_config["target_objects"][obj_type] = {
                "expected_evaluation_count": eval_count,
                "total_samples_labeled": num_labeled_samples,
                "feature_ranges": feature_ranges # Assign the completed dict
            }
        # --- End Feature Processing Loop ---

        # --- 2. Process Distances (Unchanged) ---
        print("\nProcessing learned distances...")
        if not self.learned_distances: print("  - No distance samples collected.")
        else:
            for pair_key, distances in self.learned_distances.items():
                if not distances: continue
                print(f"  - Processing distance for pair '{pair_key}' ({len(distances)} samples)")
                min_dist, max_dist, mean_dist, stddev_dist = float(np.min(distances)), float(np.max(distances)), float(np.mean(distances)), float(np.std(distances))
                tolerance_value = (distance_tolerance_percent / 100.0) * max_dist if max_dist > 0 else 0.0
                min_bound = max(0.0, min_dist - tolerance_value); max_bound = max_dist + tolerance_value
                final_config["distance_constraints"][pair_key] = {
                    "range": [round(min_bound, 3), round(max_bound, 3)], "mean": round(mean_dist, 3),
                    "stddev": round(stddev_dist, 3), "count": len(distances) }

        # --- 3. Process Relative Positions (Unchanged) ---
        print("\nProcessing learned relative positions...")
        if not self.learned_relative_positions: print("  - No relative position samples collected.")
        else:
            for pair_key, positions in self.learned_relative_positions.items():
                if not positions: continue
                num_samples = len(positions); print(f"  - Processing relative position for pair '{pair_key}' ({num_samples} samples)")
                dx_values = [p[0] for p in positions]; dy_values = [p[1] for p in positions]
                mean_dx = float(np.mean(dx_values)); stddev_dx = float(np.std(dx_values))
                min_dx_bound = mean_dx - relpos_tolerance_stdevs * stddev_dx; max_dx_bound = mean_dx + relpos_tolerance_stdevs * stddev_dx
                mean_dy = float(np.mean(dy_values)); stddev_dy = float(np.std(dy_values))
                min_dy_bound = mean_dy - relpos_tolerance_stdevs * stddev_dy; max_dy_bound = mean_dy + relpos_tolerance_stdevs * stddev_dy
                final_config["relative_position_constraints"][pair_key] = {
                    "dx_range": [round(min_dx_bound, 3), round(max_dx_bound, 3)], "dy_range": [round(min_dy_bound, 3), round(max_dy_bound, 3)],
                    "mean_dx": round(mean_dx, 3), "stddev_dx": round(stddev_dx, 3), "mean_dy": round(mean_dy, 3), "stddev_dy": round(stddev_dy, 3),
                    "count": num_samples }

        # --- 4. Generate Overlap Rules (Unchanged) ---
        print("\nGenerating overlap rules...")
        if not self.observed_pairs: print("  - No object pairs observed together.")
        else:
            added_rules = set()
            for pair_key in self.observed_pairs:
                 obj_types = sorted(pair_key.split('-'))
                 rule_tuple = (tuple(obj_types), overlap_mode)
                 if len(obj_types) == 2 and rule_tuple not in added_rules:
                     rule = {"objects": obj_types, "mode": overlap_mode}
                     if overlap_mode == "ratio": rule["threshold"] = overlap_threshold
                     final_config["overlap_rules"].append(rule); added_rules.add(rule_tuple)
            for obj_type in self.learned_features.keys():
                 rule_tuple = (tuple(sorted([obj_type, obj_type])), overlap_mode)
                 if rule_tuple not in added_rules:
                     rule = {"objects": [obj_type, obj_type], "mode": overlap_mode}
                     if overlap_mode == "ratio": rule["threshold"] = overlap_threshold
                     final_config["overlap_rules"].append(rule); added_rules.add(rule_tuple)

        # --- 5. Finalize and Format JSON ---
        try:
            json_output = json.dumps(final_config, indent=2)
            print(f"\n[{self.__class__.__name__}] Configuration generation successful.")
            return json_output
        except Exception as e:
            print(f"[Error][{self.__class__.__name__}] Failed to serialize configuration to JSON: {e}")
            traceback.print_exc(); return None
    # --- END UPDATED generate_suggested_config METHOD ---

