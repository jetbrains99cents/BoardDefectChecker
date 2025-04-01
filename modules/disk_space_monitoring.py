from PySide6.QtCore import QThread, Signal
import psutil
import time
import os
import shutil

class DiskSpaceMonitor(QThread):
    """Thread to monitor total free disk space across all drives, delete files if low, and emit updates."""
    disk_space_updated = Signal(str)  # Regular text update
    low_space_warning = Signal(bool)  # True if near threshold, False if not
    critical_space_stop = Signal(bool)  # True to stop software, False to resume

    def __init__(self, interval=5, language="en", low_space_threshold=5):
        super().__init__()
        self.interval = interval
        self._language = language  # Use private attribute for language
        self.low_space_threshold = low_space_threshold * (1024 ** 3)  # GB to bytes
        self.warning_threshold = (low_space_threshold + 10) * (1024 ** 3)  # 10 GB above
        self.running = True
        self.folders_to_clean = [
            r"C:\BoardDefectChecker\ai-outputs",
            r"C:\BoardDefectChecker\images\binary-images",
            r"C:\BoardDefectChecker\images\raw-images",
            r"C:\BoardDefectChecker\images\resized-images"
        ]

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value
        print(f"[DiskSpaceMonitor] Language updated to: {value}")

    def get_all_drives(self):
        """Get a list of all mounted disk partitions."""
        drives = []
        for partition in psutil.disk_partitions():
            if partition.fstype:
                drives.append(partition.mountpoint)
        return drives

    def clean_folders(self):
        """Delete all files and subfolders in specified folders."""
        for folder in self.folders_to_clean:
            if os.path.exists(folder):
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    try:
                        if os.path.isfile(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f"[DiskSpaceMonitor] Failed to delete {item_path}: {e}")

    def run(self):
        """Monitor disk space, clean if low, and emit status signals."""
        while self.running:
            try:
                drives = self.get_all_drives()
                if not drives:
                    text = "Dung lượng ổ đĩa: \nKhông phát hiện ổ đĩa" if self.language == "vi" else "Free disk size: \nNo drives detected"
                    self.disk_space_updated.emit(text)
                    self.low_space_warning.emit(False)
                    self.critical_space_stop.emit(False)
                    time.sleep(self.interval)
                    continue

                total_free = 0
                total_capacity = 0
                for drive in drives:
                    try:
                        usage = psutil.disk_usage(drive)
                        total_free += usage.free
                        total_capacity += usage.total
                    except Exception:
                        continue

                total_free_gb = total_free / (1024 ** 3)
                total_capacity_gb = total_capacity / (1024 ** 3)
                text = (f"Dung lượng ổ đĩa: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB" if self.language == "vi"
                        else f"Free disk size: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB")

                # Check space conditions
                if total_free < self.low_space_threshold:
                    print(f"[DiskSpaceMonitor] Free space {total_free_gb:.0f} GB below threshold {self.low_space_threshold / (1024 ** 3)} GB. Cleaning folders...")
                    self.clean_folders()
                    total_free = sum(psutil.disk_usage(drive).free for drive in drives if psutil.disk_usage(drive))
                    total_free_gb = total_free / (1024 ** 3)
                    text = (f"Dung lượng ổ đĩa: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB" if self.language == "vi"
                            else f"Free disk size: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB")
                    if total_free < self.low_space_threshold:
                        self.critical_space_stop.emit(True)
                        print(f"[DiskSpaceMonitor] Free space still low after cleaning: {total_free_gb:.0f} GB")
                    else:
                        self.critical_space_stop.emit(False)
                    self.low_space_warning.emit(True)
                elif total_free < self.warning_threshold:
                    self.low_space_warning.emit(True)
                    self.critical_space_stop.emit(False)
                else:
                    self.low_space_warning.emit(False)
                    self.critical_space_stop.emit(False)

                self.disk_space_updated.emit(text)
            except Exception as e:
                error_text = f"Lỗi: {str(e)}" if self.language == "vi" else f"Error: {str(e)}"
                self.disk_space_updated.emit(error_text)
                self.low_space_warning.emit(False)
                self.critical_space_stop.emit(False)
            time.sleep(self.interval)

    def stop(self):
        """Stop the monitoring thread."""
        self.running = False
        self.wait()