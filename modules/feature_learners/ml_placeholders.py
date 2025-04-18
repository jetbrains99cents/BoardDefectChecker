# modules/feature_learners/ml_placeholders.py

from typing import Optional

# Import the base class from the same directory/package
try:
    from .base import BaseFeatureLearner
except ImportError:
    # Fallback for running directly or if structure differs
    from base import BaseFeatureLearner

class BasicMachineLearningFeatureLearner(BaseFeatureLearner):
    """ Placeholder for learning using basic ML/DL models (e.g., SVM, simple NN). """
    def generate_suggested_config(self, **kwargs) -> Optional[str]:
        """Abstract method override - Not implemented."""
        print(f"[Warning][{self.__class__.__name__}] generate_suggested_config not implemented yet.")
        raise NotImplementedError("This is a placeholder class.")


class AdvancedAIModelFeatureLearner(BaseFeatureLearner):
    """ Placeholder for learning using advanced AI models (e.g., complex CNNs, Transformers). """
    def generate_suggested_config(self, **kwargs) -> Optional[str]:
        """Abstract method override - Not implemented."""
        print(f"[Warning][{self.__class__.__name__}] generate_suggested_config not implemented yet.")
        raise NotImplementedError("This is a placeholder class.")

