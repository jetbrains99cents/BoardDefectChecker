import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import itertools
import math


class RotationInvariantAOIChecker:
    """
    Checks masks based on rotation-invariant features, distance constraints,
    and overlap rules defined in a configuration.
    Uses the 'expected_evaluation_count' specified in the configuration for evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RotationInvariantAOIChecker with configuration.
        Reads 'expected_evaluation_count' for evaluation checks.

        Args:
            config (Dict[str, Any]): Configuration dictionary specifying target objects
                                     (with 'expected_evaluation_count'),
                                     (optional) distance_constraints, and
                                     (optional) overlap_rules.
        """
        # Basic validation
        if not isinstance(config, dict):
             raise ValueError("Configuration must be a dictionary.")
        if "target_objects" not in config:
             print("[Warning][RotationInvariantAOIChecker] Config missing 'target_objects'.")
             config["target_objects"] = {}

        self.config = config
        self.target_objects = self.config.get("target_objects", {})
        self.distance_constraints = self.config.get("distance_constraints", {})
        self.overlap_rules = self.config.get("overlap_rules", [])

        # Pre-process config: Convert list ranges to tuples
        target_objects_dict = self.target_objects
        if target_objects_dict and isinstance(target_objects_dict, dict):
            for spec in target_objects_dict.values():
                if isinstance(spec, dict) and "feature_ranges" in spec and isinstance(spec["feature_ranges"], dict):
                    for key, value in spec["feature_ranges"].items():
                        if isinstance(value, list) and len(value) == 2:
                            try: spec["feature_ranges"][key] = tuple(map(float, value))
                            except (ValueError, TypeError): print(f"[Warning] Invalid feature range {key}: {value}.")
                elif not isinstance(spec, dict): print(f"[Warning] Invalid spec format: {spec}")

        if self.distance_constraints and isinstance(self.distance_constraints, dict):
            for constraint in self.distance_constraints.values():
                 if isinstance(constraint, dict) and "range" in constraint and isinstance(constraint["range"], list) and len(constraint["range"]) == 2:
                      try: constraint["range"] = tuple(map(float, constraint["range"]))
                      except (ValueError, TypeError): print(f"[Warning] Invalid distance range: {constraint['range']}.")

        # --- UPDATED: Precompute classification rules ---
        # Read 'expected_evaluation_count' instead of 'count'
        self.classification_rules = []
        if target_objects_dict and isinstance(target_objects_dict, dict):
            for obj_type, spec in target_objects_dict.items():
                 # Ensure spec is a dictionary and contains necessary keys
                 # *** Check for 'expected_evaluation_count' ***
                 if isinstance(spec, dict) and "feature_ranges" in spec and "expected_evaluation_count" in spec:
                      feature_ranges = spec.get("feature_ranges", {})
                      # *** Read the evaluation count ***
                      expected_eval_count = spec.get("expected_evaluation_count", 0)
                      # Store as (obj_type, ranges_dict, expected_eval_count)
                      self.classification_rules.append( (obj_type, feature_ranges, expected_eval_count) )
                      # Optionally log the informational count if present
                      # total_labeled = spec.get("total_samples_labeled", "N/A")
                      # print(f"  Rule for '{obj_type}': Eval Count={expected_eval_count}, Labeled={total_labeled}")
                 else:
                      print(f"[Warning] Skipping rule for '{obj_type}': Invalid spec format or missing keys ('feature_ranges', 'expected_evaluation_count'). Spec: {spec}")

            print(f"[RotationInvariantAOIChecker] Loaded {len(self.classification_rules)} classification rules (using 'expected_evaluation_count').")
        else:
            print("[RotationInvariantAOIChecker] Warning: 'target_objects' missing or invalid. No classification rules loaded.")
        # --- END UPDATED RULE LOADING ---

        if self.distance_constraints: print(f"[RotationInvariantAOIChecker] Loaded {len(self.distance_constraints)} distance constraint rule(s).")
        if self.overlap_rules: print(f"[RotationInvariantAOIChecker] Loaded {len(self.overlap_rules)} overlap rule(s).")
        print("[RotationInvariantAOIChecker] Initialized.")


    def extract_features(self, mask: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute rotation-invariant features from a mask."""
        # (Implementation assumed unchanged)
        if not isinstance(mask, np.ndarray) or mask.ndim != 2: return None
        if mask.dtype != np.uint8: mask_uint8 = mask.astype(np.uint8)
        else: mask_uint8 = mask
        area = float(np.sum(mask_uint8 > 0));
        if area == 0: return None
        y, x = np.where(mask_uint8 > 0)
        if len(x) == 0 or len(y) == 0: return None
        centroid_x, centroid_y = float(np.mean(x)), float(np.mean(y))
        points = np.column_stack((x, y)).astype(np.float32)
        try: rect = cv2.minAreaRect(points); raw_width, raw_height = rect[1]
        except cv2.error: raw_width, raw_height = 0.0, 0.0
        larger_dim = float(max(raw_width, raw_height)); smaller_dim = float(min(raw_width, raw_height))
        aspect_ratio = smaller_dim / larger_dim if larger_dim > 0 else 0.0
        perimeter = 0.0
        try:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours: largest_contour = max(contours, key=cv2.contourArea); perimeter = float(cv2.arcLength(largest_contour, closed=True))
        except cv2.error: pass
        return {"area": area, "aspect_ratio": aspect_ratio, "centroid_x": centroid_x, "centroid_y": centroid_y,
                "perimeter": perimeter, "larger_dim": larger_dim, "smaller_dim": smaller_dim}

    def _filter_edge_masks(self, masks: List[np.ndarray], image_shape: Tuple[int, int], edge_threshold: int) -> List[np.ndarray]:
        """ Filters masks too close to image edges. """
        # (Implementation assumed unchanged)
        image_height, image_width = image_shape; filtered_masks = []
        for mask in masks:
            if not isinstance(mask, np.ndarray): continue
            y_indices, x_indices = np.where(mask > 0);
            if len(y_indices) == 0: continue
            x1, x2 = np.min(x_indices), np.max(x_indices); y1, y2 = np.min(y_indices), np.max(y_indices)
            if not (x1 < edge_threshold or x2 > image_width - edge_threshold or y1 < edge_threshold or y2 > image_height - edge_threshold): filtered_masks.append(mask)
        return filtered_masks

    def test_masks(self, masks: List[np.ndarray], image_shape: Tuple[int, int], max_masks_to_show: int = None,
                   edge_threshold: int = 5, sort_by_area: bool = True) -> List[Dict[str, Any]]:
        """Tests masks by extracting features and preparing data for visualization."""
        # (Implementation assumed unchanged)
        valid_masks = self._filter_edge_masks(masks, image_shape, edge_threshold);
        if not valid_masks: return []
        mask_features_list = []
        for mask in valid_masks:
            features = self.extract_features(mask)
            if features: mask_features_list.append({"mask": mask, "features": features})
        if not mask_features_list: return []
        if sort_by_area: mask_features_list.sort(key=lambda x: x["features"]["area"], reverse=True)
        if max_masks_to_show is not None and len(mask_features_list) > max_masks_to_show: mask_features_list = mask_features_list[:max_masks_to_show]
        results = []
        for item in mask_features_list:
            mask = item["mask"]; features = item["features"]; min_rect_vertices = [[0, 0]] * 4
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                points = np.column_stack((x_indices, y_indices)).astype(np.float32)
                try: rect = cv2.minAreaRect(points); box = cv2.boxPoints(rect); min_rect_vertices = box.astype(np.int32).tolist()
                except cv2.error: pass
            labels = [f"Area: {features['area']:.0f}", f"Perimeter: {features['perimeter']:.0f}",
                      f"Larger Dim: {features['larger_dim']:.0f}", f"Smaller Dim: {features['smaller_dim']:.0f}",
                      f"Aspect Ratio: {features['aspect_ratio']:.2f}"]
            results.append({"mask": mask, "features": features, "min_rect_vertices": min_rect_vertices, "labels": labels})
        return results

    def classify_masks(self, masks: List[np.ndarray]) -> Dict[str, List[Dict]]:
        """ Classifies masks based on feature ranges defined in the config. """
        # (Implementation assumed unchanged)
        if not self.classification_rules:
             print("[Error] Cannot classify: No rules defined.");
             target_keys = list(self.target_objects.keys()) if self.target_objects else []
             return {t: [] for t in target_keys}
        classified = {obj_type: [] for obj_type, _, _ in self.classification_rules}
        unclassified_count = 0
        for mask in masks:
            features = self.extract_features(mask);
            if not features: continue
            classified_flag = False
            for obj_type, ranges, _ in self.classification_rules: # Ignore count from rule tuple here
                match = True
                if not isinstance(ranges, dict): match = False
                else:
                    for feature_key, range_tuple in ranges.items():
                        if not isinstance(range_tuple, (tuple, list)) or len(range_tuple) != 2: match = False; break
                        min_val, max_val = range_tuple
                        feature_value = features.get(feature_key)
                        if feature_value is None or not (min_val <= feature_value <= max_val): match = False; break
                if match: classified[obj_type].append({"features": features, "mask": mask}); classified_flag = True; break
            if not classified_flag: unclassified_count += 1
        # print(f"[Checker][classify_masks] Classified: { {k: len(v) for k, v in classified.items()} }, Unclassified: {unclassified_count}") # Verbose
        return classified

    def _check_distance_constraints(self, classified: Dict[str, List[Dict]]) -> Tuple[bool, str]:
        """Checks if at least one defined distance constraint is met."""
        # (Implementation assumed unchanged)
        if not self.distance_constraints: return True, "No distance constraints defined"
        # print("[Checker] Checking distance constraints...") # Verbose
        any_constraint_satisfied = False
        for pair_key, constraint_data in self.distance_constraints.items():
            try:
                dist_range = constraint_data.get("range"); min_dist, max_dist = dist_range
                obj_types = pair_key.split('-'); obj1_type, obj2_type = obj_types[0], obj_types[1]
                obj1_list = classified.get(obj1_type, []); obj2_list = classified.get(obj2_type, [])
                if not obj1_list or not obj2_list: continue
                this_constraint_pair_found = False
                for o1_info in obj1_list:
                    for o2_info in obj2_list:
                        f1 = o1_info.get("features"); f2 = o2_info.get("features");
                        if not f1 or not f2: continue
                        c1x, c1y = f1.get("centroid_x"), f1.get("centroid_y"); c2x, c2y = f2.get("centroid_x"), f2.get("centroid_y")
                        if None not in [c1x, c1y, c2x, c2y]:
                            dist = math.sqrt((c2x - c1x) ** 2 + (c2y - c1y) ** 2)
                            if min_dist <= dist <= max_dist: this_constraint_pair_found = True; break
                    if this_constraint_pair_found: break
                if this_constraint_pair_found: any_constraint_satisfied = True; break
            except Exception as e: print(f"[Error] Distance check failed for '{pair_key}': {e}"); continue
        if any_constraint_satisfied: return True, "At least one distance constraint satisfied"
        else:
            if not self.distance_constraints: return True, "No distance constraints defined"
            else: return False, "No defined distance constraints satisfied by any object pair"

    def check_overlaps(self, classified: Dict[str, List[Dict]]) -> Tuple[bool, str]:
        """Detects forbidden overlaps based on rules defined in the config."""
        # (Implementation assumed unchanged)
        if not self.overlap_rules: return True, "No overlap rules defined"
        # print("[Checker] Checking overlap rules...") # Verbose
        for rule in self.overlap_rules:
            obj_types = rule.get("objects"); mode = rule.get("mode", "absolute").lower(); threshold = rule.get("threshold", 0.0)
            if not obj_types or len(obj_types) != 2: print(f"[Warning] Invalid 'objects' list in overlap rule: {rule}. Skipping."); continue
            obj1_type, obj2_type = obj_types[0], obj_types[1]
            obj1_list = classified.get(obj1_type, []); obj2_list = classified.get(obj2_type, [])
            if not obj1_list or not obj2_list: continue
            for o1_idx, o1 in enumerate(obj1_list):
                start_idx = o1_idx + 1 if obj1_type == obj2_type else 0
                for o2_idx in range(start_idx, len(obj2_list)):
                    o2 = obj2_list[o2_idx]
                    mask1 = o1.get("mask"); mask2 = o2.get("mask")
                    if mask1 is None or mask2 is None: print(f"[Warning] Missing mask for overlap check between {obj1_type} and {obj2_type}."); continue
                    mask1_bool = mask1.astype(bool); mask2_bool = mask2.astype(bool); intersection = np.sum(mask1_bool & mask2_bool)
                    if mode == "absolute":
                        if intersection > 0: reason = f"Overlap_NG: Absolute overlap detected between {obj1_type} and {obj2_type} (Intersection: {intersection} pixels)"; return False, reason
                    elif mode == "ratio":
                        union = np.sum(mask1_bool | mask2_bool); overlap_ratio = intersection / union if union > 0 else 0.0
                        if overlap_ratio > threshold: reason = f"Overlap_NG: Ratio between {obj1_type} & {obj2_type} ({overlap_ratio:.3f}) > threshold {threshold:.3f}"; return False, reason
                    else: # Fallback
                        if intersection > 0: reason = f"Overlap_NG: Absolute overlap detected between {obj1_type} and {obj2_type} (Unknown mode)"; return False, reason
        return True, "No excessive overlaps detected"


    # --- evaluate METHOD (Uses count loaded from config during __init__) ---
    def evaluate(self,
                 masks: List[np.ndarray],
                 image_shape: Tuple[int, int],
                 edge_threshold: int = 5,
                 sort_by_area: bool = True) -> Tuple[bool, str]:
        """
        Evaluates masks based on classification counts (from config's
        'expected_evaluation_count'), distance, and overlap.

        Args:
            masks (List[np.ndarray]): List of masks detected in the image.
            image_shape (Tuple[int, int]): Shape of the original image (height, width).
            edge_threshold (int): Pixels to ignore near the border.
            sort_by_area (bool): Whether to sort masks by area.

        Returns:
            Tuple[bool, str]: (True if OK, False if NG), Reason string.
        """
        print("[Checker] Starting evaluation...")
        valid_masks = self._filter_edge_masks(masks, image_shape, edge_threshold);
        if not valid_masks:
            print("[Checker] Eval NG: No valid masks after edge filtering")
            return False, "No valid masks found after edge filtering"

        # Classification (based on feature ranges only)
        print("[Checker] Classifying masks...")
        try:
            classified = self.classify_masks(valid_masks)
        except Exception as e:
            print(f"[Error] Classification error during evaluation: {e}");
            return False, f"Unexpected classification error: {e}"

        # --- Count Check (Uses expected_eval_count loaded during __init__) ---
        print("[Checker] Performing count check using 'expected_evaluation_count' from config.")
        count_ok = True
        count_reason_parts = []
        if not self.classification_rules:
             print("[Warning] No classification rules loaded from config for count check.")
             count_ok = True # Assume OK if no rules defined? Or NG? Let's assume OK.
        else:
            # Iterate through rules: (obj_type, ranges, expected_eval_count)
            for obj_type, _, expected_eval_count in self.classification_rules:
                found_count = len(classified.get(obj_type, []))
                # Fail if the found count does not exactly match the expected evaluation count
                if found_count != expected_eval_count:
                    count_ok = False
                    count_reason_parts.append(f"Count_NG(Cfg): Expected {expected_eval_count} '{obj_type}', found {found_count}")

        if not count_ok:
            reason = "; ".join(count_reason_parts)
            print(f"[Checker] Eval NG ({reason})")
            return False, reason
        print("[Checker] Count check passed.")
        # --- End Count Check ---

        # Distance Constraint Check
        distance_ok, distance_reason = self._check_distance_constraints(classified);
        if not distance_ok:
            print(f"[Checker] Eval NG ({distance_reason})")
            return False, distance_reason
        print(f"[Checker] Distance check passed ({distance_reason}).")

        # Overlap Check
        overlap_ok, overlap_reason = self.check_overlaps(classified);
        if not overlap_ok:
            print(f"[Checker] Eval NG ({overlap_reason})")
            return False, overlap_reason
        print(f"[Checker] Overlap check passed ({overlap_reason}).")

        # If all checks passed
        final_reason = "All checks passed (Counts(Eval Cfg), Distance, Overlap)"
        print(f"[Checker] Evaluation Result: OK ({final_reason})")
        return True, final_reason
    # --- END evaluate METHOD ---

