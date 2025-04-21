# rotation_invariant_checking.py
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
import itertools
import math
import traceback # Added for potential error logging

# --- Helper Function for IoU Calculation (Optimized: Assumes boolean inputs) ---
def calculate_iou(mask1_bool: np.ndarray, mask2_bool: np.ndarray) -> float:
    """
    Calculates Intersection over Union (IoU) for two boolean masks.
    Assumes inputs are already boolean NumPy arrays.
    """
    # Inputs are assumed boolean, directly use bitwise operations
    intersection = np.sum(mask1_bool & mask2_bool)
    union = np.sum(mask1_bool | mask2_bool)

    if union == 0:
        return 0.0
    else:
        iou = intersection / union
        return float(iou)
# --- End Helper Function ---

# --- Helper Function for Bounding Box IoU ---
def calculate_bbox_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Calculates IoU for two bounding boxes (x1, y1, x2, y2)."""
    x1_i, y1_i, x2_i, y2_i = box1
    x1_j, y1_j, x2_j, y2_j = box2

    # Determine the coordinates of the intersection rectangle
    x_left = max(x1_i, x1_j)
    y_top = max(y1_i, y1_j)
    x_right = min(x2_i, x2_j)
    y_bottom = min(y2_i, y2_j)

    # If width or height of intersection is negative, no overlap
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    area1 = (x2_i - x1_i) * (y2_i - y1_i)
    area2 = (x2_j - x1_j) * (y2_j - y1_j)

    # Calculate union area
    union_area = float(area1 + area2 - intersection_area)

    if union_area == 0:
        return 0.0 # Should not happen if areas > 0 and intersection exists
    else:
        iou = intersection_area / union_area
        return float(iou)
# --- End Helper Function ---


class RotationInvariantAOIChecker:
    """
    Checks masks based on rotation-invariant features, distance constraints,
    and overlap rules defined in a configuration.
    Classification now integrates IoU filtering (optimized), feature checks,
    and strict distance filtering.
    Uses the 'expected_evaluation_count' specified in the configuration for evaluation.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RotationInvariantAOIChecker with configuration.
        Reads 'expected_evaluation_count' for evaluation checks.
        """
        # --- Initialization (unchanged) ---
        if not isinstance(config, dict): raise ValueError("Configuration must be a dictionary.")
        if "target_objects" not in config: print(
            "[Warning][RotationInvariantAOIChecker] Config missing 'target_objects'."); config["target_objects"] = {}
        self.config = config
        self.target_objects = self.config.get("target_objects", {})
        self.distance_constraints = self.config.get("distance_constraints", {})
        self.overlap_rules = self.config.get("overlap_rules", [])
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
        self.classification_rules = []
        if target_objects_dict and isinstance(target_objects_dict, dict):
            for obj_type, spec in target_objects_dict.items():
                if isinstance(spec, dict) and "feature_ranges" in spec and "expected_evaluation_count" in spec:
                    feature_ranges = spec.get("feature_ranges", {})
                    expected_eval_count = spec.get("expected_evaluation_count", 0)
                    self.classification_rules.append({'type': obj_type, 'ranges': feature_ranges, 'expected_count': expected_eval_count})
                else: print(f"[Warning] Skipping rule for '{obj_type}': Invalid spec or missing keys ('feature_ranges', 'expected_evaluation_count').")
            print(f"[RotationInvariantAOIChecker] Loaded {len(self.classification_rules)} classification rules.")
        else: print("[RotationInvariantAOIChecker] Warning: 'target_objects' missing or invalid.")
        if self.distance_constraints: print(f"[RotationInvariantAOIChecker] Loaded {len(self.distance_constraints)} distance constraint rule(s).")
        if self.overlap_rules: print(f"[RotationInvariantAOIChecker] Loaded {len(self.overlap_rules)} overlap rule(s).")
        print("[RotationInvariantAOIChecker] Initialized.")
        # --- End Initialization ---

    def extract_features(self, mask: np.ndarray) -> Optional[Dict[str, float]]:
        """Compute rotation-invariant features from a mask."""
        # (Implementation unchanged)
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
        aspect_ratio = smaller_dim / larger_dim if larger_dim > 0 else 0.0; perimeter = 0.0
        try:
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours: largest_contour = max(contours, key=cv2.contourArea); perimeter = float(cv2.arcLength(largest_contour, closed=True))
        except cv2.error: pass
        return {"area": area, "aspect_ratio": aspect_ratio, "centroid_x": centroid_x, "centroid_y": centroid_y,
                "perimeter": perimeter, "larger_dim": larger_dim, "smaller_dim": smaller_dim}

    def _filter_edge_masks(self, masks: List[np.ndarray], filter_shape: Tuple[int, int], edge_threshold: int) -> List[np.ndarray]:
        """Filters masks too close to image edges, using the provided filter_shape."""
        # (Implementation unchanged)
        filter_height, filter_width = filter_shape
        filtered_masks = []
        for i, mask in enumerate(masks):
            if not isinstance(mask, np.ndarray) or mask.ndim != 2: continue
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) == 0: continue
            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            is_near_edge = (x1 < edge_threshold or x2 >= filter_width - edge_threshold or
                            y1 < edge_threshold or y2 >= filter_height - edge_threshold)
            if not is_near_edge: filtered_masks.append(mask)
        return filtered_masks

    def test_masks(self, masks: List[np.ndarray], filter_shape: Tuple[int, int], max_masks_to_show: int = None,
                   edge_threshold: int = 5, sort_by_area: bool = True) -> List[Dict[str, Any]]:
        """
        Tests masks by extracting features and preparing data for visualization.
        Uses filter_shape for edge filtering. Does NOT perform classification or distance checks.
        """
        # (Implementation unchanged)
        valid_masks = self._filter_edge_masks(masks, filter_shape, edge_threshold);
        if not valid_masks: return []
        mask_features_list = []
        for mask in valid_masks:
            features = self.extract_features(mask)
            if features: mask_features_list.append({"mask": mask, "features": features})
        if not mask_features_list: return []
        if sort_by_area: mask_features_list.sort(key=lambda x: x["features"]["area"], reverse=True)
        if max_masks_to_show is not None and len(mask_features_list) > max_masks_to_show:
            mask_features_list = mask_features_list[:max_masks_to_show]
        results = []
        for item in mask_features_list:
            mask = item["mask"]; features = item["features"]; min_rect_vertices = [[0, 0]] * 4
            y_indices, x_indices = np.where(mask > 0)
            if len(y_indices) > 0:
                points = np.column_stack((x_indices, y_indices)).astype(np.float32)
                try: rect = cv2.minAreaRect(points); box = cv2.boxPoints(rect); min_rect_vertices = box.astype(np.int32).tolist()
                except cv2.error: pass
            labels = [f"Area: {features.get('area', 0):.0f}", f"Perimeter: {features.get('perimeter', 0):.0f}",
                      f"Larger Dim: {features.get('larger_dim', 0):.0f}", f"Smaller Dim: {features.get('smaller_dim', 0):.0f}",
                      f"Aspect Ratio: {features.get('aspect_ratio', 0):.2f}"]
            results.append({"mask": mask, "features": features, "min_rect_vertices": min_rect_vertices, "labels": labels})
        return results

    # --- *** UPDATED classify_masks METHOD (Optimized IoU Filtering) *** ---
    def classify_masks(self,
                       masks: List[np.ndarray],
                       filter_shape: Tuple[int, int],
                       edge_threshold: int,
                       iou_threshold: float = 0.9,
                       bbox_iou_skip_threshold: float = 0.0 # Threshold for bbox IoU to skip mask IoU (0.0 means skip only if no bbox overlap)
                       ) -> Dict[str, List[Dict]]:
        """
        Classifies masks based on feature ranges AND distance constraints,
        after filtering edge masks and highly overlapping duplicates (IoU).

        Workflow: Edge Filter -> IoU Filter (Optimized) -> Feature Classify -> Distance Filter
        """
        print("[Checker][classify_masks] Starting integrated classification...")

        # 1. Edge Filtering
        edge_filtered_masks = self._filter_edge_masks(masks, filter_shape, edge_threshold)
        if not edge_filtered_masks:
            print("[Checker][classify_masks] No valid masks after edge filtering.")
            return {rule['type']: [] for rule in self.classification_rules} if self.classification_rules else {}
        print(f"[Checker][classify_masks] Edge filtering complete ({len(edge_filtered_masks)} masks remain).")

        # 2. IoU Filtering (Optimized with BBox Pre-Check)
        print(f"[Checker][classify_masks] Applying IoU filter (Mask IoU Threshold: {iou_threshold:.2f}, BBox Skip Threshold: {bbox_iou_skip_threshold:.2f})...")
        num_initial_masks = len(edge_filtered_masks)
        if num_initial_masks <= 1:
             print("[Checker][classify_masks] Skipping IoU filter (0 or 1 mask).")
             iou_filtered_masks = edge_filtered_masks # No filtering needed
        else:
            # Pre-calculate bounding boxes, areas, and boolean masks
            mask_bboxes = []
            mask_areas = []
            boolean_masks = []
            valid_indices_for_iou = [] # Keep track of indices that yield valid bboxes/areas
            for idx, m in enumerate(edge_filtered_masks):
                y_indices, x_indices = np.where(m > 0)
                if len(y_indices) == 0: continue # Skip empty masks
                x1, y1 = np.min(x_indices), np.min(y_indices)
                x2, y2 = np.max(x_indices), np.max(y_indices)
                mask_bboxes.append((x1, y1, x2, y2))
                mask_areas.append(len(y_indices)) # Area is just count of non-zero pixels
                boolean_masks.append(m.astype(bool)) # Convert to boolean once
                valid_indices_for_iou.append(idx)

            if len(valid_indices_for_iou) <= 1:
                 print("[Checker][classify_masks] Skipping IoU filter (0 or 1 mask after validation).")
                 iou_filtered_masks = [edge_filtered_masks[i] for i in valid_indices_for_iou]
            else:
                # Map valid_indices back to 0-based index for lists above
                index_map = {original_idx: new_idx for new_idx, original_idx in enumerate(valid_indices_for_iou)}
                num_valid_masks = len(valid_indices_for_iou)
                to_remove_original_indices = set()

                for i_new in range(num_valid_masks):
                    original_idx_i = valid_indices_for_iou[i_new]
                    if original_idx_i in to_remove_original_indices: continue

                    for j_new in range(i_new + 1, num_valid_masks):
                        original_idx_j = valid_indices_for_iou[j_new]
                        if original_idx_j in to_remove_original_indices: continue

                        # BBox Pre-Check
                        bbox_i = mask_bboxes[i_new]
                        bbox_j = mask_bboxes[j_new]
                        bbox_iou = calculate_bbox_iou(bbox_i, bbox_j)

                        # Skip expensive mask IoU if bbox IoU is below threshold
                        if bbox_iou <= bbox_iou_skip_threshold:
                            continue

                        # Calculate mask IoU only if bboxes overlap sufficiently
                        mask_iou = calculate_iou(boolean_masks[i_new], boolean_masks[j_new])

                        if mask_iou > iou_threshold:
                            # High overlap: remove the mask with smaller area
                            if mask_areas[i_new] >= mask_areas[j_new]:
                                to_remove_original_indices.add(original_idx_j)
                            else:
                                to_remove_original_indices.add(original_idx_i)
                                break # Mask i is removed, stop comparing it

                # Create the final list after IoU filtering
                iou_filtered_masks = [edge_filtered_masks[i] for i in valid_indices_for_iou if i not in to_remove_original_indices]

        num_removed_iou = num_initial_masks - len(iou_filtered_masks)
        print(f"[Checker][classify_masks] IoU filtering complete ({num_removed_iou} masks removed, {len(iou_filtered_masks)} remain).")

        if not iou_filtered_masks:
            print("[Checker][classify_masks] No masks remain after IoU filtering.")
            return {rule['type']: [] for rule in self.classification_rules} if self.classification_rules else {}

        # 3. Initial Feature Classification (on IoU filtered masks)
        # ... (Rest of feature classification logic is unchanged) ...
        print("[Checker][classify_masks] Performing initial feature classification...")
        if not self.classification_rules: print("[Error][classify_masks] Cannot classify: No rules defined."); return {}
        classified_by_features: Dict[str, List[Dict]] = {rule['type']: [] for rule in self.classification_rules}
        feature_classified_list: List[Dict] = []
        unclassified_count = 0
        for mask in iou_filtered_masks:
            features = self.extract_features(mask)
            if not features: continue
            classified_flag = False
            for rule in self.classification_rules:
                obj_type = rule['type']; ranges = rule['ranges']; match = True
                if not isinstance(ranges, dict): match = False
                else:
                    for feature_key, range_tuple in ranges.items():
                        if not isinstance(range_tuple, (tuple, list)) or len(range_tuple) != 2: match = False; break
                        min_val, max_val = range_tuple; feature_value = features.get(feature_key)
                        if feature_value is None or not (min_val <= feature_value <= max_val): match = False; break
                if match:
                    mask_info = {"features": features, "mask": mask, "type": obj_type}
                    classified_by_features[obj_type].append(mask_info)
                    feature_classified_list.append(mask_info)
                    classified_flag = True; break
            if not classified_flag: unclassified_count += 1
        num_feature_classified = len(feature_classified_list)
        print(f"[Checker][classify_masks] Feature classification done. Classified: { {k: len(v) for k, v in classified_by_features.items()} }, Unclassified: {unclassified_count}")
        if num_feature_classified == 0: print("[Checker][classify_masks] No masks passed feature classification. Returning empty."); return classified_by_features

        # 4. Strict Distance Filtering (Applied to feature-classified masks)
        # ... (Distance filtering logic is unchanged) ...
        print("[Checker][classify_masks] Applying strict distance constraint filter...")
        if not self.distance_constraints: print("[Checker][classify_masks] No distance constraints defined. Skipping distance filter."); return classified_by_features
        final_classification: Dict[str, List[Dict]] = {rule['type']: [] for rule in self.classification_rules}
        objects_failed_distance_count = 0; first_distance_failure_reason = "All objects satisfy applicable distance constraints"
        for obj1_info in feature_classified_list:
            obj1_type = obj1_info['type']; obj1_features = obj1_info.get("features", {})
            obj1_passes_all_constraints = True; relevant_constraints = {}; constraint_check_possible = True
            for pair_key, constraint_data in self.distance_constraints.items():
                obj_types_in_pair = pair_key.split('-')
                if obj1_type in obj_types_in_pair:
                    dist_range = constraint_data.get("range")
                    if not isinstance(dist_range, (tuple, list)) or len(dist_range) != 2:
                        obj1_passes_all_constraints = False; failure_reason = f"Distance_NG: Invalid range format for '{pair_key}' affecting {obj1_type}"
                        if objects_failed_distance_count == 0: first_distance_failure_reason = failure_reason
                        constraint_check_possible = False; break
                    relevant_constraints[pair_key] = {'data': constraint_data, 'satisfied': False}
            if not constraint_check_possible: objects_failed_distance_count += 1; continue
            if not relevant_constraints: pass
            else:
                for pair_key, status_info in relevant_constraints.items():
                    if not obj1_passes_all_constraints: break
                    constraint_data = status_info['data']; min_dist, max_dist = constraint_data["range"]
                    obj_types_in_pair = pair_key.split('-'); obj2_type = next(t for t in obj_types_in_pair if t != obj1_type)
                    obj2_list = classified_by_features.get(obj2_type, [])
                    if not obj2_list:
                        obj1_passes_all_constraints = False; failure_reason = f"Distance_NG: {obj1_type} failed '{pair_key}' - no feature-classified '{obj2_type}' found."
                        if objects_failed_distance_count == 0: first_distance_failure_reason = failure_reason; break
                    found_satisfying_pair = False
                    for obj2_info in obj2_list:
                        if obj1_info is obj2_info: continue
                        f1 = obj1_info.get("features"); f2 = obj2_info.get("features")
                        if not f1 or not f2: continue
                        c1x, c1y = f1.get("centroid_x"), f1.get("centroid_y"); c2x, c2y = f2.get("centroid_x"), f2.get("centroid_y")
                        if None not in [c1x, c1y, c2x, c2y]:
                            try:
                                dist = math.sqrt((c2x - c1x)**2 + (c2y - c1y)**2)
                                if min_dist <= dist <= max_dist: status_info['satisfied'] = True; found_satisfying_pair = True; break
                            except Exception as calc_e:
                                 obj1_passes_all_constraints = False; failure_reason = f"Distance_NG: Error calculating distance for {obj1_type} constraint '{pair_key}'"
                                 if objects_failed_distance_count == 0: first_distance_failure_reason = failure_reason; break
                    if not obj1_passes_all_constraints: break
                    if not found_satisfying_pair:
                        obj1_passes_all_constraints = False; failure_reason = f"Distance_NG: {obj1_type} failed constraint '{pair_key}' - no partner in range [{min_dist:.1f}, {max_dist:.1f}]."
                        if objects_failed_distance_count == 0: first_distance_failure_reason = failure_reason; break
            if obj1_passes_all_constraints: final_classification[obj1_type].append(obj1_info)
            else: objects_failed_distance_count += 1
        num_passed_all = sum(len(v) for v in final_classification.values())
        print(f"[Checker][classify_masks] Distance filter applied. {objects_failed_distance_count} object(s) removed.")
        print(f"[Checker][classify_masks] Final classification counts: { {k: len(v) for k, v in final_classification.items()} }")
        return final_classification
    # --- *** END UPDATED classify_masks METHOD *** ---

    def check_overlaps(self, classified: Dict[str, List[Dict]]) -> Tuple[bool, str]:
        """Detects forbidden overlaps based on rules defined in the config."""
        # (Implementation unchanged)
        if not self.overlap_rules: return True, "No overlap rules defined"
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
                    mask1_bool = mask1.astype(bool) if mask1.dtype != bool else mask1
                    mask2_bool = mask2.astype(bool) if mask2.dtype != bool else mask2
                    intersection = np.sum(mask1_bool & mask2_bool)
                    if mode == "absolute":
                        if intersection > 0: reason = f"Overlap_NG: Absolute overlap detected between {obj1_type} and {obj2_type} (Intersection: {intersection} pixels)"; return False, reason
                    elif mode == "ratio":
                        union = np.sum(mask1_bool | mask2_bool); overlap_ratio = intersection / union if union > 0 else 0.0
                        if overlap_ratio > threshold: reason = f"Overlap_NG: Ratio between {obj1_type} & {obj2_type} ({overlap_ratio:.3f}) > threshold {threshold:.3f}"; return False, reason
                    else: # Default to absolute if mode unknown
                        if intersection > 0: reason = f"Overlap_NG: Absolute overlap detected between {obj1_type} and {obj2_type} (Unknown mode)"; return False, reason
        return True, "No excessive overlaps detected"

    # --- *** UPDATED evaluate METHOD (Passes IoU Threshold) *** ---
    def evaluate(self,
                 masks: List[np.ndarray],
                 filter_shape: Tuple[int, int],
                 edge_threshold: int = 5,
                 iou_threshold: float = 0.9 # <-- Pass this down
                 ) -> Tuple[bool, str, Dict[str, List[Dict]]]:
        """
        Evaluates masks: Calls integrated classification (Edge->IoU->Feature->Distance) -> Count Check -> Overlap Check.
        Returns the final status, reason, and the dictionary of masks that passed all checks.
        """
        print("[Checker] Starting evaluation...")
        final_reason = "Evaluation not completed"
        final_classified_masks: Dict[str, List[Dict]] = {rule['type']: [] for rule in self.classification_rules} if self.classification_rules else {}

        # 1. Integrated Classification (now includes IoU)
        try:
            # Pass iou_threshold to the updated classify_masks
            final_classified_masks = self.classify_masks(masks, filter_shape, edge_threshold, iou_threshold)
            num_final_classified = sum(len(v) for v in final_classified_masks.values())
            if num_final_classified == 0 and len(masks) > 0 :
                 final_reason = "Eval_NG: No masks passed the combined classification and filtering steps."
                 print(f"[Checker] {final_reason}")
                 return False, final_reason, final_classified_masks
            print(f"[Checker] Integrated classification/filtering complete ({num_final_classified} masks remain).")
        except Exception as e:
            final_reason = f"Eval_NG: Unexpected error during integrated classification: {e}"
            print(f"[Checker] {final_reason}\n{traceback.format_exc()}")
            return False, final_reason, final_classified_masks

        # 2. Count Check (unchanged)
        print("[Checker] Performing count check using 'expected_evaluation_count' from config...")
        count_ok = True; count_reason_parts = []
        if not self.classification_rules: print("[Warning] No classification rules loaded for count check.")
        else:
            for rule in self.classification_rules:
                obj_type = rule['type']; expected_eval_count = rule['expected_count']
                found_count = len(final_classified_masks.get(obj_type, []))
                if found_count != expected_eval_count:
                    count_ok = False; count_reason_parts.append(f"Count_NG(Cfg): Expected {expected_eval_count} '{obj_type}', found {found_count}")
        if not count_ok:
            final_reason = "Eval_NG: " + "; ".join(count_reason_parts)
            print(f"[Checker] {final_reason}")
            return False, final_reason, final_classified_masks
        print("[Checker] Count check passed.")

        # 3. Overlap Check (unchanged)
        print("[Checker] Performing overlap check...")
        overlap_ok, overlap_reason = self.check_overlaps(final_classified_masks)
        if not overlap_ok:
            final_reason = f"Eval_NG: {overlap_reason}"
            print(f"[Checker] {final_reason}")
            return False, final_reason, final_classified_masks
        print(f"[Checker] Overlap check passed ({overlap_reason}).")

        # If all checks passed
        final_reason = "OK: All checks passed (Integrated Classification/Filtering, Count, Overlap)"
        print(f"[Checker] Evaluation Result: {final_reason}")
        return True, final_reason, final_classified_masks
    # --- *** END UPDATED evaluate METHOD *** ---

