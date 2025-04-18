# modules/feature_learning.py
# This script now acts as a simple entry point to run the interactive learning tool.

import sys
import os
import traceback

print(f"Attempting to run feature learning tool...")
print(f"Current working directory: {os.getcwd()}")
print(f"Sys Path: {sys.path}")

try:
    # Try importing from the feature_learners sub-package/directory
    # This assumes feature_learning.py is in 'modules' and feature_learners is inside 'modules'
    from feature_learners.feature_learning_tool import run_learning_process
    print("Successfully imported run_learning_process from feature_learners.feature_learning_tool")

except ImportError as e1:
    print(f"[ImportError 1] Failed to import from feature_learners.feature_learning_tool: {e1}")
    try:
        # Fallback: Maybe feature_learners is directly importable (if modules is in PYTHONPATH)
        print("Attempting fallback import...")
        # This might require 'modules' or the project root to be in PYTHONPATH
        from modules.feature_learners.feature_learning_tool import run_learning_process
        print("Successfully imported run_learning_process via modules.feature_learners...")
    except ImportError as e2:
         print(f"[ImportError 2] Failed fallback import from modules.feature_learners.feature_learning_tool: {e2}")
         print("\n[FATAL ERROR] Could not locate the feature learning tool.")
         print("Please ensure the directory structure is correct:")
         print("  modules/")
         print("    ├── feature_learning.py (this file)")
         print("    └── feature_learners/")
         print("          ├── base.py")
         print("          ├── statistical.py")
         print("          ├── learning_visualizer.py")
         print("          └── feature_learning_tool.py")
         print("Also ensure you are running this script from a context where 'modules' is accessible.")
         sys.exit(1)
    except Exception as e_other:
         print(f"[FATAL ERROR] An unexpected error occurred during import: {e_other}")
         traceback.print_exc()
         sys.exit(1)


if __name__ == "__main__":
    try:
        run_learning_process()
    except Exception as main_err:
         print(f"\n[FATAL ERROR] An error occurred while running the learning process: {main_err}")
         traceback.print_exc()

