# modules/feature_learners/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional, Set # Added Set
import math

class BaseFeatureLearner(ABC):
    """
    Abstract base class for different feature learning approaches.
    Defines the common interface for collecting samples (features & distances)
    and generating configurations. Tracks observed pairs for overlap rule generation.
    """

    def __init__(self):
        """Initializes the learner's data structures."""
        self.learned_features: Dict[str, List[Dict[str, float]]] = {}
        self.learned_distances: Dict[str, List[float]] = {}
        # --- NEW: Track pairs seen together on the same image ---
        self.observed_pairs: Set[str] = set() # Stores keys like "type1-type2"
        # --- End New ---
        self.last_test_results: List[Dict[str, Any]] = []
        self.feature_keys = ["area", "aspect_ratio", "larger_dim", "smaller_dim", "perimeter"]
        # Initialization message moved to concrete classes

    def set_last_test_results(self, test_results: List[Dict[str, Any]]):
        """Stores the latest results from the checker's test_masks method."""
        # (Content Unchanged)
        if isinstance(test_results, list): self.last_test_results = test_results
        else: print(f"[Error][{self.__class__.__name__}] Invalid test_results format."); self.last_test_results = []

    def add_sample(self, object_type: str, mask_number: int) -> bool:
        """Adds the feature dictionary for a given object type and mask number."""
        # (Content Unchanged)
        if not self.last_test_results: print("[Error] Cannot add sample: No test results loaded."); return False
        index = mask_number - 1
        if 0 <= index < len(self.last_test_results):
            mask_data = self.last_test_results[index]; features = mask_data.get("features")
            if features and isinstance(features, dict):
                if object_type not in self.learned_features: self.learned_features[object_type] = []
                valid_feature_copy = {}
                for k, v in features.items():
                    if k in self.feature_keys:
                        try: valid_feature_copy[k] = float(v) if v is not None else 0.0
                        except (ValueError, TypeError): print(f"[Warning] Invalid value '{k}' for Mask #{mask_number} ({object_type}): {v}. Storing 0.0."); valid_feature_copy[k] = 0.0
                    else: valid_feature_copy[k] = v
                self.learned_features[object_type].append(valid_feature_copy)
                print(f"[{self.__class__.__name__}] Added sample Mask #{mask_number} as type '{object_type}'. Total samples for '{object_type}': {len(self.learned_features[object_type])}")
                return True
            else: print(f"[Error] Could not find valid features for Mask #{mask_number}.")
        else: print(f"[Error] Invalid Mask Number {mask_number}. Max is {len(self.last_test_results)}.");
        return False

    def add_distance_sample(self, pair_key: str, distance: float):
        """Adds a distance sample and tracks the observed pair."""
        if not isinstance(distance, (float, int)) or distance < 0:
             print(f"[Warning] Invalid distance value for '{pair_key}': {distance}. Skipping."); return

        if pair_key not in self.learned_distances: self.learned_distances[pair_key] = []
        self.learned_distances[pair_key].append(float(distance))
        # --- NEW: Track the pair ---
        self.observed_pairs.add(pair_key)
        # --- End New ---
        # print(f"[Debug][{self.__class__.__name__}] Added distance sample for '{pair_key}': {distance:.2f}. Total: {len(self.learned_distances[pair_key])}")

    @abstractmethod
    def generate_suggested_config(self, **kwargs) -> Optional[str]:
        """Abstract method for generating the suggested configuration JSON string."""
        pass

    def clear_samples(self):
        """Clears all collected samples."""
        self.learned_features = {}; self.learned_distances = {}; self.observed_pairs = set(); self.last_test_results = []
        print(f"[{self.__class__.__name__}] All learned samples cleared.")

    def get_summary(self) -> str:
        """Returns a string summarizing the number of samples collected."""
        # (Content Unchanged)
        if not self.learned_features and not self.learned_distances: return "No samples collected yet."
        summary = "Collected samples summary:\n"; total_feature_samples = 0
        if self.learned_features:
             summary += " Features:\n"
             for obj_type, samples in self.learned_features.items(): count = len(samples); summary += f"  - '{obj_type}': {count} samples\n"; total_feature_samples += count
             summary += f" Total feature samples: {total_feature_samples}\n"
        else: summary += " No feature samples.\n"
        total_distance_samples = 0
        if self.learned_distances:
            summary += " Distances:\n"
            for pair_key, distances in self.learned_distances.items(): count = len(distances); summary += f"  - Pair '{pair_key}': {count} samples\n"; total_distance_samples += count
            summary += f" Total distance samples: {total_distance_samples}\n"
        else: summary += " No distance samples.\n"
        return summary

    def get_total_samples(self) -> int:
        """Returns the total number of feature samples collected."""
        # (Content Unchanged)
        return sum(len(v) for v in self.learned_features.values())

    def get_total_distance_samples(self) -> int:
         """Returns the total number of distance samples collected."""
         # (Content Unchanged)
         return sum(len(v) for v in self.learned_distances.values())

