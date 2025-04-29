# launcher.py
import subprocess
import os
import sys
import platform # Import platform module

try:
    # Determine the base directory correctly whether running as script or bundled exe
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as a bundled executable (PyInstaller)
        # sys.executable is the path to the exe itself (e.g., D:\...\dist\YourApp.exe)
        # We assume .venv and main.py are in the PARENT directory of the folder containing the exe (e.g., D:\...)
        application_path = os.path.dirname(sys.executable)    # ...\dist
        project_dir = os.path.dirname(application_path)        # D:\Working\BoardDefectChecker
        print("Running from bundled EXE.")
    else:
        # Running as a standard .py script
        # __file__ is the path to launcher.py itself
        project_dir = os.path.dirname(os.path.abspath(__file__))
        print("Running as Python script.")

    # Define paths relative to the determined project_dir
    # Adjust venv path based on OS
    if platform.system() == "Windows":
        venv_python_relative = os.path.join(".venv", "Scripts", "python.exe")
    else: # Linux/macOS
        venv_python_relative = os.path.join(".venv", "bin", "python")

    venv_python_abs = os.path.join(project_dir, venv_python_relative)
    # Assuming main.py is directly inside project_dir. Adjust if it's in a subfolder e.g., "src"
    main_script_abs = os.path.join(project_dir, "main.py")

    # Print determined paths for debugging
    print(f"Project directory determined as: {project_dir}")
    print(f"Venv Python path determined as: {venv_python_abs}")
    print(f"Main script path determined as: {main_script_abs}")

    # Check if paths exist
    if not os.path.exists(venv_python_abs):
        print(f"ERROR: Virtual environment Python executable not found at '{venv_python_abs}'")
        if platform.system() == "Windows": os.system("pause") # Pause on error for compiled exe
        sys.exit(1)

    if not os.path.exists(main_script_abs):
        print(f"ERROR: main.py not found at '{main_script_abs}'")
        if platform.system() == "Windows": os.system("pause")
        sys.exit(1)

    # Run the main script using the virtual environment's Python
    # Set the working directory (cwd) to the project directory so main.py can use relative paths correctly
    print(f"\n--- Attempting to run main.py ---")
    result = subprocess.run([venv_python_abs, main_script_abs], cwd=project_dir, check=False) # Runs and waits

    print(f"--- main.py finished with exit code {result.returncode} ---")

except Exception as e:
    print(f"\n--- An unexpected error occurred in launcher.py ---")
    print(f"ERROR: {e}")
    if platform.system() == "Windows": os.system("pause") # Pause on exception
    sys.exit(1)

# Add the pause back to mimic original .bat behavior IF needed, only on Windows
# You might remove this if main.py handles its own closing or user interaction
if platform.system() == "Windows":
     print("\nScript execution finished.")
     os.system("pause") # Keeps window open until key press

sys.exit(result.returncode if 'result' in locals() else 1)