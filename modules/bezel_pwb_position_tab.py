import re
from datetime import datetime
import os
import cv2
import psutil
import numpy as np
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer, QThread, Slot
from PySide6.QtWidgets import QMessageBox

from modules.bezel_pwb_position_image_processing import BezelPWBPositionImageProcessor
from modules.ai_models import BezelPWBPositionSegmenter
from modules.language import SimpleLanguageManager
from modules.disk_space_monitoring import DiskSpaceMonitor
from modules.processing_working import BezelPWBProcessingWorker, ProcessingDialog

from typing import List, Dict, Tuple, Any, Optional


class BezelPWBTabController:
    def __init__(self, ui, camera_worker, camera_interaction, snap_worker_left, snap_worker_right,
                 detection_log_worker=None, parent=None):
        self.processing_worker = None
        self.processing_dialog = None
        self.ui = ui
        self.parent = parent
        self.camera_worker = camera_worker
        self.camera_interaction = camera_interaction
        self.snap_worker_left = snap_worker_left
        self.snap_worker_right = snap_worker_right
        self.detection_log_worker = detection_log_worker

        self.is_counter_turned_on = False
        self.is_processed_images_shown = False
        self.is_options_locked = False
        self.is_pwb_check_enabled = True
        self.current_connection_state = False
        self.last_selected_index_left = None
        self.last_selected_index_right = None
        # Add streaming state variables
        self.is_streaming_left = False
        self.is_streaming_right = False

        lang_path = os.path.join(os.path.dirname(__file__), "language.json")
        self.lang_manager = SimpleLanguageManager(lang_path)
        self.lang_manager.set_tab("bezel_pwb_position_tab")
        self.lang_manager.set_language("en")
        self.apply_translations()

        self.processor = BezelPWBPositionImageProcessor()
        self.bezel_pwb_ai_model = BezelPWBPositionSegmenter(model_type="x", model_path="ai-models/")

        self.disk_monitor = DiskSpaceMonitor(interval=5, language="en", low_space_threshold=5)
        self.disk_blink_timer = QTimer()
        self.is_blinking = False
        self.is_critical = False
        self.was_low_space = False
        self.was_critical = False
        self.blink_visible = True
        self.disk_monitor.disk_space_updated.connect(self.update_disk_space_label)
        self.disk_monitor.low_space_warning.connect(self.handle_low_space_warning)
        self.disk_monitor.critical_space_stop.connect(self.handle_critical_space)
        self.disk_blink_timer.timeout.connect(self.toggle_blink)
        self.disk_monitor.start()

        self.load_config()

        # Set the log directory for Bezel PWB Position
        self.update_log_directory()

        # By default, enable counter, disk space monitoring, and checking without JIG options
        self.ui.checkBox_26.setChecked(True)  # Enable counter
        self.ui.checkBox_12.setChecked(True)  # Enable disk space monitoring
        self.ui.checkBox_12.setEnabled(False)  # Disable checkBox_12 so users can't toggle it
        self.ui.checkBox_14.setChecked(True)  # Enable checking without JIG
        self.ui.checkBox_14.setEnabled(False)  # Disable checkBox_14 so users can't toggle it

        self.toggle_counter(2)  # Correctly enable the counter logic (2 = checked state)

        self.ui.pushButton_51.clicked.connect(self.open_camera_settings)
        self.ui.pushButton_54.clicked.connect(self.toggle_video_streaming_left)
        self.ui.pushButton_55.clicked.connect(self.toggle_video_streaming_right)
        self.ui.checkBox_26.stateChanged.connect(self.toggle_counter)
        self.ui.checkBox_27.stateChanged.connect(self.toggle_image_displayed)
        self.ui.checkBox_28.stateChanged.connect(self.toggle_options_lock)
        self.ui.radioButton_13.toggled.connect(self.on_english_selected)
        self.ui.radioButton_14.toggled.connect(self.on_vietnamese_selected)
        self.ui.lineEdit_30.returnPressed.connect(self.update_staff_id)
        self.ui.lineEdit_29.returnPressed.connect(self.handle_part_number_return_pressed)
        self.ui.checkBox_16.stateChanged.connect(self.toggle_pwb_check)

        self.ui.radioButton_13.setText("ENG")
        self.ui.radioButton_14.setText("VIE")

        if self.snap_worker_left:
            self.snap_worker_left.image_captured.connect(self.update_video_frame_left)
            self.snap_worker_left.devices_available.connect(self.populate_combo_box_left)
        if self.snap_worker_right:
            self.snap_worker_right.image_captured.connect(self.update_video_frame_right)
            self.snap_worker_right.devices_available.connect(self.populate_combo_box_right)

        if self.snap_worker_left:
            print("[BezelPWB] Forcing left device enumeration on startup...")
            self.snap_worker_left.enumerate_devices()
        if self.snap_worker_right:
            print("[BezelPWB] Forcing right device enumeration on startup...")
            self.snap_worker_right.enumerate_devices()
        QThread.msleep(500)

        self.ui.label_166.setText("--")
        self.ui.label_168.setText("--")
        self.ui.label_174.setText("--")
        self.ui.label_167.setText("--")
        self.ui.label_171.setText("--")
        self.ui.label_179.setText("--")  # Initialize label_179 to normal state
        self.ui.label_179.setStyleSheet("color: white;")  # Set to white in normal state
        self.ui.label_315.setText(self.lang_manager.get_text("label_315"))
        self.ui.label_322.setText(self.lang_manager.get_text("label_322"))
        self.update_detected_result_count()

        if self.camera_worker and hasattr(self.camera_worker, "connection_status_changed"):
            self.camera_worker.connection_status_changed.connect(self.update_connection_status)
            if not self.camera_worker.isRunning():
                self.camera_worker.start()

        # Ensure the log_result signal is connected
        if self.detection_log_worker:
            self.detection_log_worker.log_result_signal.connect(self.on_log_result)
            print("[BezelPWB] Connected log_result_signal to on_log_result slot")
        else:
            print("[BezelPWB] Warning: detection_log_worker is None, cannot connect log_result_signal")

    # -------------------- Log Directory Management --------------------
    def update_log_directory(self):
        """Set the log directory to bezel-pwb-position-dd-mm-yyyy format inside operation-logs."""
        if not self.detection_log_worker:
            print("[BezelPWB] Warning: detection_log_worker is None, cannot set log directory")
            return

        # Base directory updated to operation-logs
        base_dir = r"C:\BoardDefectChecker\operation-logs"
        current_date = datetime.now().strftime("%d-%m-%Y")
        log_dir = os.path.join(base_dir, f"bezel-pwb-position-{current_date}")

        print(f"[BezelPWB] Log directory set to: {log_dir}")

        # Use the setter method instead of direct attribute access
        self.detection_log_worker.set_log_directory(log_dir)

    # Translation Methods
    def apply_translations(self):
        """Update translatable UI elements for this tab."""
        translatable_widgets = {
            "label_307": self.ui.label_307, "label_306": self.ui.label_306,
            "checkBox_26": self.ui.checkBox_26, "checkBox_27": self.ui.checkBox_27,
            "pushButton_51": self.ui.pushButton_51, "label_313": self.ui.label_313,
            "label_316": self.ui.label_316, "checkBox_28": self.ui.checkBox_28,
            "label_177": self.ui.label_177, "label_182": self.ui.label_182,
            "checkBox_14": self.ui.checkBox_14, "checkBox_15": self.ui.checkBox_15,
            "label_320": self.ui.label_320, "label_183": self.ui.label_183,
            "label_184": self.ui.label_184, "label_314": self.ui.label_314,
            "label_318": self.ui.label_318, "label_185": self.ui.label_185,
            "label_186": self.ui.label_186,
            "label_104": self.ui.label_104,
            "label_158_connected": self.ui.label_158,  # Connected state
            "label_158_disconnected": self.ui.label_158,  # Disconnected state
            "label_164": self.ui.label_164, "label_133": self.ui.label_133,
            "label_157": self.ui.label_157, "label_163": self.ui.label_163,
            "label_162": self.ui.label_162, "label_159": self.ui.label_159,
            "label_102": self.ui.label_102, "label_103": self.ui.label_103,
            "label_169": self.ui.label_169, "label_172": self.ui.label_172,
            "label_173": self.ui.label_173, "label_166": self.ui.label_166,
            "label_165": self.ui.label_165, "label_168": self.ui.label_168,
            "label_174": self.ui.label_174, "label_167": self.ui.label_167,
            "label_170": self.ui.label_170, "label_171": self.ui.label_171,
            "checkBox_12": self.ui.checkBox_12,
            "label_315": self.ui.label_315, "label_322": self.ui.label_322,
            "label_311": self.ui.label_311, "label_319": self.ui.label_319,
            "label_310": self.ui.label_310, "label_317": self.ui.label_317,
            "label_321": self.ui.label_321, "checkBox_16": self.ui.checkBox_16
        }

        # Update all translatable widgets except pushButton_54 and pushButton_55
        for key, widget in translatable_widgets.items():
            if key not in ["pushButton_54", "pushButton_55", "label_158_connected", "label_158_disconnected"]:
                widget.setText(self.lang_manager.get_text(key))

        # Special handling for pushButton_54 and pushButton_55 to reflect streaming state
        if self.snap_worker_left and self.snap_worker_left.isRunning():
            self.ui.pushButton_54.setText(
                "Dừng truyền video" if self.lang_manager.current_language == "vi" else "Stop video streaming"
            )
        else:
            self.ui.pushButton_54.setText(self.lang_manager.get_text("pushButton_54"))

        if self.snap_worker_right and self.snap_worker_right.isRunning():
            self.ui.pushButton_55.setText(
                "Dừng truyền video" if self.lang_manager.current_language == "vi" else "Stop video streaming"
            )
        else:
            self.ui.pushButton_55.setText(self.lang_manager.get_text("pushButton_55"))

        self.update_connection_status(self.current_connection_state or False)
        print(
            f"[BezelPWB] Translations applied for tab: {self.lang_manager.current_tab}, language: {self.lang_manager.current_language}")

    def on_english_selected(self, checked):
        if checked:
            self.lang_manager.set_language("en")
            self.disk_monitor.language = "en"
            self.apply_translations()
            self.update_connection_status(self.current_connection_state or False)
            # Update tab names for English
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_operation_log), "Operation log")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_robotic_arm), "Robotic arm")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_aoi_machine), "AOI checking machine")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_tp_fpc),
                                         "TP FPC insertion checker")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_fpc_mounting_machine),
                                         "FPC mounting machine")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_bezel_pwb_position),
                                         "Bezel - PWB position checker")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_led_fpc),
                                         "LED FPC insertion checker")

    def on_vietnamese_selected(self, checked):
        if checked:
            self.lang_manager.set_language("vi")
            self.disk_monitor.language = "vi"
            self.apply_translations()
            self.update_connection_status(self.current_connection_state or False)
            # Update tab names for Vietnamese
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_operation_log), "Nhật ký hoạt động")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_robotic_arm), "Cánh tay robot")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_aoi_machine), "Máy kiểm AOI tự động")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_tp_fpc),
                                         "Kiểm TP FPC")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_fpc_mounting_machine),
                                         "Máy gắn FPC tự động")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_bezel_pwb_position),
                                         "Kiểm bẻ ngàm - dán PWB")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_led_fpc),
                                         "Kiểm LED FPC")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_optional_aoi),
                                         "Kiểm AOI tùy chọn")

    @Slot(str)
    def update_language(self, new_language):
        print(f"[BezelPWB] Language changed signal received: {new_language}")
        self.disk_monitor.language = new_language
        if new_language == "en":
            self.on_english_selected(True)
        else:
            self.on_vietnamese_selected(True)

    # Config Loading
    def load_config(self):
        """Load config and update label_157 and label_164."""
        try:
            config_data = self.processor.load_config()  # Expecting a dict with at least 'part_name'

            # Set default successful status even if config is empty
            self.ui.label_157.setText(self.lang_manager.get_text("label_157"))  # "Successful" or "Thành công"

            if config_data and isinstance(config_data, dict):
                # Update label_164 with "Part name:" followed by the actual part name
                part_name = config_data.get("part_name", "")
                self.ui.label_164.setText(f"Part name: {part_name}")
                print(f"[BezelPWB] Config loaded successfully: {config_data}")
            else:
                # Keep "Successful" but indicate empty config for part name
                self.ui.label_164.setText("Part name: ")
                print("[BezelPWB] Config loading: No data available")
        except Exception as e:
            self.ui.label_157.setText("Failed")
            self.ui.label_164.setText("Part name: Error")
            print(f"[BezelPWB] Config loading error: {e}")

    # Camera & Streaming Methods
    def populate_combo_box_left(self, device_info_list):
        self.ui.comboBox_4.clear()
        if not device_info_list:
            self.ui.comboBox_4.addItem("No devices available")
        else:
            for display_text, dev_info in device_info_list:
                self.ui.comboBox_4.addItem(display_text, dev_info)
            self.ui.comboBox_4.setCurrentIndex(0)
        print(
            f"[BezelPWB] Left combo box populated: {self.ui.comboBox_4.count()} items, Selected: {self.ui.comboBox_4.currentText()} (Index: {self.ui.comboBox_4.currentIndex()})")
        self.update_connection_status(len(device_info_list) > 0)

    def populate_combo_box_right(self, device_info_list):
        self.ui.comboBox_5.clear()
        if not device_info_list:
            self.ui.comboBox_5.addItem("No devices available")
        else:
            for display_text, dev_info in device_info_list:
                self.ui.comboBox_5.addItem(display_text, dev_info)
            self.ui.comboBox_5.setCurrentIndex(1 if len(device_info_list) > 1 else 0)
        print(
            f"[BezelPWB] Right combo box populated: {self.ui.comboBox_5.count()} items, Selected: {self.ui.comboBox_5.currentText()} (Index: {self.ui.comboBox_5.currentIndex()})")
        self.update_connection_status(len(device_info_list) > 0)

    def open_camera_settings(self):
        if self.camera_interaction:
            self.camera_interaction.select_device()

    def toggle_video_streaming_left(self):
        if self.snap_worker_left:
            current_index = self.ui.comboBox_4.currentIndex()
            if self.snap_worker_left.isRunning():
                self.stop_video_streaming_left()
                if current_index != self.last_selected_index_left:
                    self.start_video_streaming_left()
            else:
                self.start_video_streaming_left()
            self.last_selected_index_left = current_index

    def toggle_video_streaming_right(self):
        if self.snap_worker_right:
            current_index = self.ui.comboBox_5.currentIndex()
            if self.snap_worker_right.isRunning():
                self.stop_video_streaming_right()
                if current_index != self.last_selected_index_right:
                    self.start_video_streaming_right()
            else:
                self.start_video_streaming_right()
            self.last_selected_index_right = current_index

    def start_video_streaming_left(self):
        if self.snap_worker_left and not self.snap_worker_left.isRunning():
            selected_index = self.ui.comboBox_4.currentIndex()
            if selected_index == -1 or self.ui.comboBox_4.currentText() == "No devices available":
                title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
                message = ("Vui lòng chọn một thiết bị camera từ danh sách trước khi bắt đầu streaming.\n"
                           "Please select a camera device from the list before starting streaming.")
                QMessageBox.warning(None, title, message)
                return
            print(
                f"[BezelPWB] Starting left stream - Device index: {selected_index}, Combo text: {self.ui.comboBox_4.currentText()}")
            self.snap_worker_left.open_device(selected_index)
            self.snap_worker_left.pre_process()
            self.snap_worker_left.start()
            self.ui.pushButton_54.setText(
                "Dừng truyền video" if self.lang_manager.current_language == "vi" else "Stop video streaming"
            )
            self.ui.comboBox_4.setEnabled(False)
            # Update streaming state
            self.is_streaming_left = True
            print("[BezelPWB] Left camera streaming state: ON")

    def start_video_streaming_right(self):
        if self.snap_worker_right and not self.snap_worker_right.isRunning():
            selected_index = self.ui.comboBox_5.currentIndex()
            if selected_index == -1 or self.ui.comboBox_5.currentText() == "No devices available":
                title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
                message = ("Vui lòng chọn một thiết bị camera từ danh sách trước khi bắt đầu streaming.\n"
                           "Please select a camera device from the list before starting streaming.")
                QMessageBox.warning(None, title, message)
                return
            print(
                f"[BezelPWB] Starting right stream - Device index: {selected_index}, Combo text: {self.ui.comboBox_5.currentText()}")
            self.snap_worker_right.open_device(selected_index)
            self.snap_worker_right.pre_process()
            self.snap_worker_right.start()
            self.ui.pushButton_55.setText(
                "Dừng truyền video" if self.lang_manager.current_language == "vi" else "Stop video streaming"
            )
            self.ui.comboBox_5.setEnabled(False)
            # Update streaming state
            self.is_streaming_right = True
            print("[BezelPWB] Right camera streaming state: ON")

    def stop_video_streaming_left(self):
        if self.snap_worker_left and self.snap_worker_left.isRunning():
            self.snap_worker_left.stop()
            self.ui.pushButton_54.setText(
                "Bắt đầu truyền video" if self.lang_manager.current_language == "vi" else "Start video streaming"
            )
            self.ui.comboBox_4.setEnabled(True)
            self.ui.label_315.setPixmap(QPixmap())
            self.ui.label_315.setText(self.lang_manager.get_text("label_315"))
            # Update streaming state
            self.is_streaming_left = False
            print("[BezelPWB] Left camera streaming state: OFF")

    def stop_video_streaming_right(self):
        if self.snap_worker_right and self.snap_worker_right.isRunning():
            self.snap_worker_right.stop()
            self.ui.pushButton_55.setText(
                "Bắt đầu truyền video" if self.lang_manager.current_language == "vi" else "Start video streaming"
            )
            self.ui.comboBox_5.setEnabled(True)
            self.ui.label_322.setPixmap(QPixmap())
            self.ui.label_322.setText(self.lang_manager.get_text("label_322"))
            # Update streaming state
            self.is_streaming_right = False
            print("[BezelPWB] Right camera streaming state: OFF")

    def update_video_frame_left(self, frame):
        if frame is not None:
            h, w, c = frame.shape
            q_img = QImage(frame.data, w, h, c * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(250, 250, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.label_315.setGeometry(180, 190, 250, 250)
            self.ui.label_315.setPixmap(pixmap)
            self.ui.label_315.setScaledContents(True)
            self.ui.label_315.setText("")

    def update_video_frame_right(self, frame):
        if frame is not None:
            h, w, c = frame.shape
            q_img = QImage(frame.data, w, h, c * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(250, 250, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.label_322.setGeometry(450, 190, 250, 250)
            self.ui.label_322.setPixmap(pixmap)
            self.ui.label_322.setScaledContents(True)
            self.ui.label_322.setText("")

    # Disk Space Management
    def toggle_blink(self):
        self.blink_visible = not self.blink_visible
        # Blink both label_177 and label_179 when critical or low space
        if self.is_critical or self.is_blinking:
            color = "#be2b25"  # Red for critical or low space
            self.ui.label_177.setStyleSheet(f"color: {'transparent' if not self.blink_visible else color};")
            self.ui.label_179.setStyleSheet(f"color: {'transparent' if not self.blink_visible else color};")
        else:
            self.ui.label_177.setStyleSheet("color: #25be7c;")  # Green for normal state
            self.ui.label_179.setStyleSheet("color: white;")  # White for normal state

    @Slot(str)
    def update_disk_space_label(self, text):
        self.ui.label_177.setText(text)
        if not self.is_blinking and not self.is_critical:
            self.ui.label_177.setStyleSheet("color: #25be7c;")

    @Slot(bool)
    def handle_low_space_warning(self, is_low):
        if is_low and not self.is_blinking:
            self.is_blinking = True
            self.disk_blink_timer.start(500)
            if not self.was_low_space:
                print("[BezelPWB] Low disk space warning activated")
                self.was_low_space = True
        elif not is_low and not self.is_critical:
            self.is_blinking = False
            self.disk_blink_timer.stop()
            self.ui.label_177.setStyleSheet("color: #25be7c;")
            if self.was_low_space:
                print("[BezelPWB] Low disk space warning deactivated")
                self.was_low_space = False

    @Slot(bool)
    def handle_critical_space(self, is_critical):
        self.is_critical = is_critical
        if is_critical:
            self.is_blinking = True
            self.disk_blink_timer.start(500)
            self.ui.tabWidget.setEnabled(False)
            if not self.was_critical:
                print("[BezelPWB] Critical disk space: Software stopped")
                self.was_critical = True
            # Update label_179 with the critical message based on language
            if self.lang_manager.current_language == "vi":
                self.ui.label_179.setText("Xóa bớt dung lượng \nổ cứng")
            else:
                self.ui.label_179.setText("Delete some files on disk")

        elif not is_critical:
            self.is_blinking = False
            self.disk_blink_timer.stop()
            self.ui.tabWidget.setEnabled(True)
            if self.was_critical:
                print("[BezelPWB] Critical disk space resolved")
                self.was_critical = False
            # Reset label_179 to normal state
            self.ui.label_179.setText("--")
            self.ui.label_179.setStyleSheet("color: white;")

    # Toggle Methods
    def toggle_counter(self, state):
        self.is_counter_turned_on = (state == 2)
        print(f"[BezelPWB] Counter turned {'ON' if self.is_counter_turned_on else 'OFF'}")
        self.update_detected_result_count()

    def toggle_image_displayed(self, state):
        """
        Toggle display of processing steps and intermediate images based on checkbox state.
        When enabled, this shows images during processing (binary conversion, etc).

        Args:
            state: State of the checkbox (2 = checked, 0 = unchecked)
        """
        self.is_processed_images_shown = (state == 2)
        self.processor.is_images_shown = self.is_processed_images_shown
        print(f"[BezelPWB] Show processed images: {self.is_processed_images_shown}")
        if self.is_options_locked:
            QMessageBox.information(None, "Message", "Options are locked.")
            # Restore checkbox to its previous state if locked
            self.ui.checkBox_27.setChecked(not self.is_processed_images_shown)

    def toggle_options_lock(self, state):
        self.is_options_locked = (state == 2)
        if self.is_options_locked:
            print("[BezelPWB] Options lock turned on.")
            self.ui.checkBox_26.setEnabled(False)
            self.ui.checkBox_27.setEnabled(False)
            self.ui.pushButton_51.setEnabled(False)
            self.ui.checkBox_28.setEnabled(False)
            self.ui.checkBox_12.setEnabled(False)
            self.ui.checkBox_14.setEnabled(False)
            self.ui.checkBox_15.setEnabled(False)
            self.ui.checkBox_16.setEnabled(False)
            QMessageBox.information(None, "Message", "Options are locked.")
        else:
            print("[BezelPWB] Options lock turned off.")
            self.ui.checkBox_26.setEnabled(True)
            self.ui.checkBox_27.setEnabled(True)
            self.ui.pushButton_51.setEnabled(True)
            self.ui.checkBox_28.setEnabled(True)
            self.ui.checkBox_12.setEnabled(True)
            self.ui.checkBox_14.setEnabled(True)
            self.ui.checkBox_15.setEnabled(True)
            self.ui.checkBox_16.setEnabled(True)

    # --- Method to toggle PWB check state (MODIFIED) ---
    def toggle_pwb_check(self, state):
        """
        Toggles the PWB check based on the checkbox state.
        Checked (state=2) means PWB check is DISABLED (False).
        Unchecked (state=0) means PWB check is ENABLED (True).
        Updates the internal state and shows a translated message directly defined here.
        """
        # Reverse the logic: checked (state=2) means DISABLED (False)
        self.is_pwb_check_enabled = (state != 2) # True if unchecked, False if checked

        # --- MODIFICATION: Define translations directly ---
        title = "PWB Check Status" # Default English title
        status_message = ""
        lang = "en" # Default language

        # Try to get current language from manager if it exists
        if hasattr(self, 'lang_manager') and self.lang_manager is not None:
            lang = self.lang_manager.current_language

        # Set title and message based on language and state
        if lang == "vi":
            title = "Trạng thái kiểm PWB"
            if self.is_pwb_check_enabled:
                status_message = "Kiểm PWB được BẬT."
            else:
                status_message = "Kiểm PWB bị TẮT."
        else: # Default to English
            title = "PWB Check Status"
            if self.is_pwb_check_enabled:
                status_message = "PWB check is ENABLED."
            else:
                status_message = "PWB check is DISABLED."
        # --- END MODIFICATION ---

        print(f"[BezelPWB] PWB Check toggled. New state: {'ENABLED' if self.is_pwb_check_enabled else 'DISABLED'} (Lang: {lang})")
        QMessageBox.information(None, title, status_message) # Show the determined message

        # Optional: Check if options are locked and potentially revert UI state
        # if hasattr(self, 'is_options_locked') and self.is_options_locked:
        #     locked_message_default_en = "Options are locked."
        #     locked_message_default_vi = "Các tùy chọn đã bị khóa."
        #     locked_message = locked_message_default_vi if lang == "vi" else locked_message_default_en
        #     # Note: No fetching needed here either if embedding directly
        #     QMessageBox.information(None, "Info", locked_message)
        #     # Find the checkbox widget (replace checkBox_15 if needed)
        #     # pwb_checkbox = self.ui.checkBox_15
        #     # pwb_checkbox.blockSignals(True) # Prevent recursive signals
        #     # pwb_checkbox.setChecked(not self.is_pwb_check_enabled) # Revert UI to match actual state
        #     # pwb_checkbox.blockSignals(False)

    # Staff ID and Part Number
    def update_staff_id(self):
        staff_id = self.ui.lineEdit_30.text().strip()
        if not staff_id:
            title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Input Error"
            message = ("Mã nhân viên không được để trống.\n"
                       "Staff ID cannot be empty.")
            QMessageBox.warning(None, title, message)
            return
        if self.detection_log_worker:
            self.detection_log_worker.set_staff_id(staff_id)
        title = "Thông báo / Notification" if self.lang_manager.current_language == "vi" else "Notification"
        message = (f"Mã nhân viên đã được cập nhật: {staff_id}\n"
                   f"Staff ID updated: {staff_id}")
        QMessageBox.information(None, title, message)
        print(f"[BezelPWB] Staff ID updated to: {staff_id}")

    def handle_part_number_return_pressed(self):
        part_number = self.ui.lineEdit_29.text().strip()
        if part_number:
            print(f"[BezelPWB] Part number entered: {part_number}")
            # Reset labels to "--" when part number is entered
            self.ui.label_174.setText("--")  # Left result
            self.ui.label_167.setText("--")  # Right result
            self.ui.label_171.setText("--")  # Final result (NG or OK)
            self.ui.label_166.setText("--")  # Left time
            self.ui.label_168.setText("--")  # Right time
            self.ui.label_103.setText("--")  # Defect reason
            self.handle_capture_and_log()

    # Capture and Log
    def handle_capture_and_log(self):
        # Check if cameras are streaming
        if not self.is_streaming_left or not self.is_streaming_right:
            title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
            message = ""
            if not self.is_streaming_left and not self.is_streaming_right:
                message = ("Vui lòng bắt đầu truyền video trên cả camera trái và phải trước khi chụp ảnh.\n"
                           "Please start video streaming on both the left and right cameras before capturing images.")
            elif not self.is_streaming_left:
                message = ("Vui lòng bắt đầu truyền video trên camera trái trước khi chụp ảnh.\n"
                           "Please start video streaming on the left camera before capturing images.")
            elif not self.is_streaming_right:
                message = ("Vui lòng bắt đầu truyền video trên camera phải trước khi chụp ảnh.\n"
                           "Please start video streaming on the right camera before capturing images.")
            QMessageBox.warning(None, title, message)
            print("[BezelPWB] Cannot process: Camera streaming not active")
            return

        if self.is_critical:
            print("[BezelPWB] Cannot process: Insufficient disk space")
            return

        staff_id = self.ui.lineEdit_30.text().strip()
        part_number = self.ui.lineEdit_29.text().strip()
        if not staff_id or not part_number:
            title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
            message = ("Mã nhân viên và Mã sản phẩm không được để trống.\n"
                       "Staff ID and Part Number cannot be empty.")
            QMessageBox.warning(None, title, message)
            return

        # Verify detection_log_worker is not None
        if self.detection_log_worker is None:
            print("[BezelPWB] Error: detection_log_worker is None, cannot log results")
            title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
            message = ("Không thể ghi log kết quả: detection_log_worker không được khởi tạo.\n"
                       "Cannot log results: detection_log_worker is not initialized.")
            QMessageBox.warning(None, title, message)
            return

        left_raw_image_path = None
        if self.snap_worker_left:
            left_raw_image_path = self.snap_worker_left.capture_single_image(save_image=True)
            if left_raw_image_path:
                left_raw_img = cv2.imread(left_raw_image_path)
                if left_raw_img is not None:
                    saved_left_raw_path = self.processor.save_raw_image(
                        left_raw_img, "left", left_raw_image_path, show_image=False
                    )
                    print(f"[Debug] Left raw path returned: {saved_left_raw_path}")
                    print(f"[Debug] Is left path absolute? {os.path.isabs(saved_left_raw_path)}")
                    try:
                        os.remove(left_raw_image_path)
                        print(f"[BezelPWB] Original left raw image deleted: {left_raw_image_path}")
                        left_raw_image_path = saved_left_raw_path
                    except Exception as e:
                        print(f"[BezelPWB] Failed to delete left raw image: {e}")
                else:
                    print(f"[BezelPWB] Failed to load left raw image from: {left_raw_image_path}")
                    left_raw_image_path = None

        right_raw_image_path = None
        if self.snap_worker_right:
            right_raw_image_path = self.snap_worker_right.capture_single_image(save_image=True)
            if right_raw_image_path:
                right_raw_img = cv2.imread(right_raw_image_path)
                if right_raw_img is not None:
                    saved_right_raw_path = self.processor.save_raw_image(
                        right_raw_img, "right", right_raw_image_path, show_image=False
                    )
                    try:
                        os.remove(right_raw_image_path)
                        print(f"[BezelPWB] Original right raw image deleted: {right_raw_image_path}")
                        right_raw_image_path = saved_right_raw_path
                    except Exception as e:
                        print(f"[BezelPWB] Failed to delete right raw image: {e}")
                else:
                    print(f"[BezelPWB] Failed to load right raw image from: {right_raw_image_path}")
                    right_raw_image_path = None

        if not left_raw_image_path or not right_raw_image_path:
            title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Capture Error"
            message = ("Không thể chụp ảnh từ cả hai camera.\n"
                       "Failed to capture images from both cameras.")
            QMessageBox.warning(None, title, message)
            return

        self.processing_worker = BezelPWBProcessingWorker(
            bezel_pwb_ai_model=self.bezel_pwb_ai_model,
            unused_model=None,
            left_input_path=left_raw_image_path,
            right_input_path=right_raw_image_path,
            detection_log_worker=self.detection_log_worker,
            part_number=part_number,
            staff_id=staff_id,
            is_counter_turned_on=self.is_counter_turned_on,
            is_show_images=self.is_processed_images_shown,
            is_pwb_check_enabled=self.is_pwb_check_enabled
        )

        self.processing_dialog = ProcessingDialog(None)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        self.processing_worker.finished.connect(self.processing_dialog.accept)
        self.processing_worker.start()
        self.processing_dialog.exec()

    # Define the path to the blank image as a class attribute or constant
    BLANK_IMAGE_PATH = r"C:\BoardDefectChecker\resources\blank.png"

    # --- UPDATED HELPER METHOD ---
    def _parse_and_translate_reason(self, reason: Optional[str], side: str) -> str:
        """
        Parses the technical reason string and returns a user-friendly,
        translated message that ALWAYS includes the side ('left'/'right' or 'trái'/'phải').
        Handles None or empty string as non-error cases.
        """
        # Ensure lang_manager is initialized and accessible
        if not hasattr(self, 'lang_manager') or self.lang_manager is None:
            print("[Error][Reason Parse] Language manager not available.")
            lang = "en"  # Fallback to English
        else:
            lang = self.lang_manager.current_language

        side_vi = "trái" if side == "left" else "phải"
        side_en = side

        # --- Translations for object types ---
        object_translations_vi = {
            "copper_mark": "dấu đồng",  # Simplified
            "bezel": "ngàm",
            "stamped_mark": "dấu dập",
            "pwb": "mạch PWB",  # Simplified
            "part": ""  # Removed fallback word
        }
        object_translations_en = {
            "copper_mark": "copper mark",
            "bezel": "bezel",
            "stamped_mark": "stamped mark",
            "pwb": "PWB",
            "part": ""  # Removed fallback word
        }

        # Default/Fallback message (includes side)
        default_reason_vi = f"Lỗi không xác định bên {side_vi}"
        default_reason_en = f"Unknown error on {side_en}"

        # --- MODIFICATION: Handle OK cases first, including empty string ---
        if reason is None or reason == "" or reason == "OK" or reason.startswith("OK:"):
            # Handle specific OK reason from evaluate if PWB check was skipped
            if isinstance(reason, str) and "[Stamped Mark Skipped]" in reason:
                # Return the specific message including the side
                return f"Kiểm PWB bị tắt bên {side_vi}" if lang == "vi" else f"PWB Check Disabled on {side_en}"
            # Standard OK case or empty reason - return empty string
            return ""
        # --- END MODIFICATION ---

        # Handle non-string reasons (already covered by the check above, but keep for safety)
        if not isinstance(reason, str):
            print(f"[Warning][Reason Parse] Reason is not a string: {reason}")
            return default_reason_vi if lang == "vi" else default_reason_en

        # --- PWB/Internal Geometry Errors ---
        if reason.startswith("InternalGeom_NG:") or reason.startswith("InternalGeom_Error:"):
            pwb_term_vi = object_translations_vi.get("pwb", "PWB")
            # Ensure side is included
            return f"Lỗi vị trí dán {pwb_term_vi} bên {side_vi}" if lang == "vi" else f"PWB position error on {side_en}"

        # --- Evaluation Errors (from checker.evaluate) ---
        if reason.startswith("Eval_NG:"):
            # Count Check Error
            if "Count_NG(Cfg): Expected" in reason:
                match = re.search(r"Expected \d+ '(\w+)'", reason)
                obj_type_en = match.group(1) if match else "part"
                if lang == "vi":
                    obj_type_display = object_translations_vi.get(obj_type_en, obj_type_en)
                    # Ensure side is included, removed "đối tượng"
                    return f"Bị thiếu {obj_type_display} bên {side_vi}"
                else:
                    obj_type_display_en = object_translations_en.get(obj_type_en, obj_type_en)
                    # Ensure side is included, removed "part"
                    return f"Missing {obj_type_display_en} on {side_en}".replace("  ",
                                                                                 " ").strip()  # Avoid double spaces if obj_type is empty
            # Overlap Check Error
            elif "Overlap_NG:" in reason:
                bezel_term_vi = object_translations_vi.get("bezel", "Bezel")
                # Ensure side is included
                return f"{bezel_term_vi} bị chồng lấn bên {side_vi}" if lang == "vi" else f"Bezel overlaps on {side_en}"
            # Classification/Filtering Errors
            elif "No masks passed" in reason or "Unexpected error during integrated classification" in reason:
                # Ensure side is included
                return f"Lỗi phân loại bên {side_vi}" if lang == "vi" else f"Classification error on {side_en}"
            # Fallback for unhandled Eval_NG
            else:
                print(f"[Warning][Reason Parse] Unhandled Eval_NG reason: {reason}")
                # Ensure side is included
                return f"Lỗi đánh giá bên {side_vi}" if lang == "vi" else f"Evaluation error on {side_en}"

        # --- Initial Processing Errors (from ai_model.process_image) ---
        elif "Model not initialized" in reason or "Invalid input image" in reason or "Overall Processing Error" in reason:
            # Ensure side is included
            return f"Lỗi xử lý ảnh bên {side_vi}" if lang == "vi" else f"Image processing error on {side_en}"

        # --- Fallback for any other unexpected string ---
        else:
            # This block should ideally not be reached now for empty strings
            print(f"[Warning][Reason Parse] Unhandled reason format: {reason}")
            # Default message already includes side
            return default_reason_vi if lang == "vi" else default_reason_en

    # --- UPDATED SLOT METHOD ---
    @Slot(object, object, object, object, str, str, float, float, str, str)  # Keep 10 arguments
    def on_processing_finished(self, left_mask_img, right_mask_img, left_annotated_pwb_img, right_annotated_pwb_img,
                               left_result, right_result, left_time, right_time, final_result,
                               defect_reason_raw):  # Receive combined reason
        """
        Handle the processing results, parse the combined reason, translate, update UI text and colors.
        """
        print("[BezelPWB] Processing finished signal received.")
        print(f"  Left Result Raw: {left_result}, Right Result Raw: {right_result}, Final Raw: {final_result}")
        print(f"  Defect Reason Raw (Combined): {defect_reason_raw}")
        print(f"  Left Time: {left_time:.3f}s, Right Time: {right_time:.3f}s")
        # ... (log image types) ...

        # --- Update Time Labels with Language ---
        if hasattr(self, 'lang_manager') and self.lang_manager is not None:
            time_unit = " giây" if self.lang_manager.current_language == "vi" else " seconds"
        else:
            time_unit = " seconds"
            print("[Warning][UI Update] Language manager not found, defaulting time unit to 'seconds'.")
        self.ui.label_166.setText(f"{left_time:.2f}{time_unit}")  # Left time
        self.ui.label_168.setText(f"{right_time:.2f}{time_unit}")  # Right time

        # --- Update Status Labels and Styles ---
        self.ui.label_174.setText(left_result)
        self.ui.label_167.setText(right_result)
        self.ui.label_171.setText(final_result)
        ok_style = "color: #25be7c;"  # Green
        ng_style = "color: #be2b25;"  # Red
        self.ui.label_174.setStyleSheet(ok_style if left_result == "OK" else ng_style)
        self.ui.label_167.setStyleSheet(ok_style if right_result == "OK" else ng_style)
        self.ui.label_171.setStyleSheet(ok_style if final_result == "OK" else ng_style)

        # --- Parse and Translate Defect Reason ---
        final_defect_reason_display = ""
        # *** Still using simulation for individual reasons - needs fixing upstream ***
        # This simulation logic needs to be replaced if individual reasons are passed
        simulated_left_reason = None
        simulated_right_reason = None
        if isinstance(defect_reason_raw, str):
            parts = defect_reason_raw.split(";")
            for part in parts:
                part = part.strip()
                if part.startswith("Left:"):
                    simulated_left_reason = part.replace("Left:", "").strip()
                elif part.startswith("Right:"):
                    simulated_right_reason = part.replace("Right:", "").strip()
            # Handle case where only one reason is present without prefix
            if len(parts) == 1 and not parts[0].startswith("Left:") and not parts[0].startswith("Right:"):
                if left_result == "NG":
                    simulated_left_reason = parts[0]
                elif right_result == "NG":
                    simulated_right_reason = parts[0]
                else:
                    simulated_left_reason = parts[0]  # Assign to left if both OK but reason exists

        left_reason_parsed = self._parse_and_translate_reason(simulated_left_reason, "left")
        right_reason_parsed = self._parse_and_translate_reason(simulated_right_reason, "right")

        # Combine parsed reasons for display
        reasons_list = [r for r in [left_reason_parsed, right_reason_parsed] if r]  # Filter out empty strings

        if not reasons_list:  # If both sides were OK and no special OK reason (like skipped PWB)
            if hasattr(self, 'lang_manager') and self.lang_manager is not None:
                final_defect_reason_display = "Không phát hiện bất thường vị trí" if self.lang_manager.current_language == "vi" else "No position defect found"
            else:
                final_defect_reason_display = "No position defect found"
        else:
            final_defect_reason_display = "; ".join(reasons_list)

        self.ui.label_103.setText(final_defect_reason_display)
        # Set defect reason label color based on the overall final_result
        self.ui.label_103.setStyleSheet(ok_style if final_result == "OK" else ng_style)
        # --- END MODIFICATION ---

        # --- Display Images (Mask and PWB/Blank) ---
        # (Image display logic remains the same)
        # Display Left Mask Image
        if left_mask_img is not None and isinstance(left_mask_img, np.ndarray):
            try:
                if len(left_mask_img.shape) == 2:
                    display_img_left = cv2.cvtColor(left_mask_img, cv2.COLOR_GRAY2RGB)
                elif left_mask_img.shape[2] == 4:
                    display_img_left = cv2.cvtColor(left_mask_img, cv2.COLOR_BGRA2RGB)
                elif left_mask_img.shape[2] == 3:
                    display_img_left = cv2.cvtColor(left_mask_img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image channel count")
                h, w, c = display_img_left.shape
                q_img = QImage(display_img_left.data, w, h, c * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(220, 220, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.ui.label_311.setGeometry(720, 105, 220, 220)
                self.ui.label_311.setPixmap(pixmap)
                self.ui.label_311.setScaledContents(True)
                self.ui.label_311.setText("")
            except Exception as e:
                print(f"[Error][UI] Failed to display left mask image: {e}"); self.ui.label_311.setText(
                    "Display Error"); self.ui.label_311.setPixmap(QPixmap())
        else:
            print("[Warning] Left mask image is None"); self.ui.label_311.setText(
                "No Image"); self.ui.label_311.setPixmap(QPixmap())

        # Display Right Mask Image
        if right_mask_img is not None and isinstance(right_mask_img, np.ndarray):
            try:
                if len(right_mask_img.shape) == 2:
                    display_img_right = cv2.cvtColor(right_mask_img, cv2.COLOR_GRAY2RGB)
                elif right_mask_img.shape[2] == 4:
                    display_img_right = cv2.cvtColor(right_mask_img, cv2.COLOR_BGRA2RGB)
                elif right_mask_img.shape[2] == 3:
                    display_img_right = cv2.cvtColor(right_mask_img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image channel count")
                h, w, c = display_img_right.shape
                q_img = QImage(display_img_right.data, w, h, c * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(220, 220, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.ui.label_310.setGeometry(1070, 105, 220, 220)
                self.ui.label_310.setPixmap(pixmap)
                self.ui.label_310.setScaledContents(True)
                self.ui.label_310.setText("")
            except Exception as e:
                print(f"[Error][UI] Failed to display right mask image: {e}"); self.ui.label_310.setText(
                    "Display Error"); self.ui.label_310.setPixmap(QPixmap())
        else:
            print("[Warning] Right mask image is None"); self.ui.label_310.setText(
                "No Image"); self.ui.label_310.setPixmap(QPixmap())

        # Display Left Annotated PWB Image or Blank
        target_label_left_pwb = self.ui.label_319
        target_label_left_pwb.setGeometry(720, 335, 220, 220)
        if left_annotated_pwb_img is not None and isinstance(left_annotated_pwb_img, np.ndarray):
            try:
                if len(left_annotated_pwb_img.shape) == 2:
                    display_pwb_left = cv2.cvtColor(left_annotated_pwb_img, cv2.COLOR_GRAY2RGB)
                elif left_annotated_pwb_img.shape[2] == 4:
                    display_pwb_left = cv2.cvtColor(left_annotated_pwb_img, cv2.COLOR_BGRA2RGB)
                elif left_annotated_pwb_img.shape[2] == 3:
                    display_pwb_left = cv2.cvtColor(left_annotated_pwb_img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image channel count")
                h, w, c = display_pwb_left.shape
                q_img = QImage(display_pwb_left.data, w, h, c * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(220, 220, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                target_label_left_pwb.setPixmap(pixmap)
                target_label_left_pwb.setScaledContents(True)
                target_label_left_pwb.setText("")
            except Exception as e:
                print(f"[Error][UI] Failed to display left PWB image: {e}"); target_label_left_pwb.setText(
                    "PWB Error"); target_label_left_pwb.setPixmap(QPixmap())
        else:
            print("[Info][UI] Left annotated PWB image is None. Displaying blank.")
            blank_pixmap = QPixmap(self.BLANK_IMAGE_PATH)
            if not blank_pixmap.isNull():
                target_label_left_pwb.setPixmap(blank_pixmap.scaled(220, 220, Qt.IgnoreAspectRatio,
                                                                    Qt.SmoothTransformation)); target_label_left_pwb.setScaledContents(
                    True); target_label_left_pwb.setText("")
            else:
                print(
                    f"[Error][UI] Failed to load blank image: {self.BLANK_IMAGE_PATH}"); target_label_left_pwb.setText(
                    "Blank Missing"); target_label_left_pwb.setPixmap(QPixmap())

        # Display Right Annotated PWB Image or Blank
        target_label_right_pwb = self.ui.label_317
        target_label_right_pwb.setGeometry(1070, 335, 220, 220)
        if right_annotated_pwb_img is not None and isinstance(right_annotated_pwb_img, np.ndarray):
            try:
                if len(right_annotated_pwb_img.shape) == 2:
                    display_pwb_right = cv2.cvtColor(right_annotated_pwb_img, cv2.COLOR_GRAY2RGB)
                elif right_annotated_pwb_img.shape[2] == 4:
                    display_pwb_right = cv2.cvtColor(right_annotated_pwb_img, cv2.COLOR_BGRA2RGB)
                elif right_annotated_pwb_img.shape[2] == 3:
                    display_pwb_right = cv2.cvtColor(right_annotated_pwb_img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError("Unsupported image channel count")
                h, w, c = display_pwb_right.shape
                q_img = QImage(display_pwb_right.data, w, h, c * w, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img).scaled(220, 220, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                target_label_right_pwb.setPixmap(pixmap)
                target_label_right_pwb.setScaledContents(True)
                target_label_right_pwb.setText("")
            except Exception as e:
                print(f"[Error][UI] Failed to display right PWB image: {e}"); target_label_right_pwb.setText(
                    "PWB Error"); target_label_right_pwb.setPixmap(QPixmap())
        else:
            print("[Info][UI] Right annotated PWB image is None. Displaying blank.")
            blank_pixmap = QPixmap(self.BLANK_IMAGE_PATH)
            if not blank_pixmap.isNull():
                target_label_right_pwb.setPixmap(blank_pixmap.scaled(220, 220, Qt.IgnoreAspectRatio,
                                                                     Qt.SmoothTransformation)); target_label_right_pwb.setScaledContents(
                    True); target_label_right_pwb.setText("")
            else:
                print(
                    f"[Error][UI] Failed to load blank image: {self.BLANK_IMAGE_PATH}"); target_label_right_pwb.setText(
                    "Blank Missing"); target_label_right_pwb.setPixmap(QPixmap())

        # Explicitly update the counter display after processing results are handled
        self.update_detected_result_count()
        print("[BezelPWB] UI updated with processing results.")

    # Counter Methods
    def update_detected_result_count(self):
        """Update count display labels (105, 161, 81)."""
        if self.is_counter_turned_on and self.detection_log_worker:
            self.ui.label_105.setText(str(self.detection_log_worker.detected_part_count))
            self.ui.label_161.setText(str(self.detection_log_worker.detected_ng_count))
            self.ui.label_81.setText(str(self.detection_log_worker.detected_ok_count))
            print(
                f"[BezelPWB] Counter updated - Total: {self.detection_log_worker.detected_part_count}, NG: {self.detection_log_worker.detected_ng_count}, OK: {self.detection_log_worker.detected_ok_count}")
        else:
            self.ui.label_105.setText("0")
            self.ui.label_161.setText("0")
            self.ui.label_81.setText("0")
            print("[BezelPWB] Counter not updated - turned off or detection_log_worker is None")

    @Slot(str, int, int, int)
    def on_log_result(self, log_message, total_count, ng_count, ok_count):
        """Handle detection log results and update counters."""
        if self.is_critical:
            print("[BezelPWB] Log update skipped due to critical disk space")
            return
        print(
            f"[BezelPWB] Received log_result_signal - Log: {log_message}, Total: {total_count}, NG: {ng_count}, OK: {ok_count}")
        if self.is_counter_turned_on:
            self.ui.label_105.setText(str(total_count))
            self.ui.label_161.setText(str(ng_count))
            self.ui.label_81.setText(str(ok_count))
            print(f"[BezelPWB] Counter updated via signal - Total: {total_count}, NG: {ng_count}, OK: {ok_count}")

    # Connection Status
    def update_connection_status(self, is_connected):
        self.current_connection_state = is_connected
        if is_connected:
            self.ui.label_158.setText(self.lang_manager.get_text("label_158_connected"))
            self.ui.label_158.setStyleSheet("color: #25be7c;")
            print("[BezelPWB] Camera Status Updated: Connected")
        else:
            self.ui.label_158.setText(self.lang_manager.get_text("label_158_disconnected"))
            self.ui.label_158.setStyleSheet("color: #be2b25;")
            print("[BezelPWB] Camera Status Updated: Disconnected")

    def __del__(self):
        if hasattr(self, 'disk_monitor'):
            self.disk_monitor.stop()
        if hasattr(self, 'disk_blink_timer'):
            self.disk_blink_timer.stop()
        if self.snap_worker_left and self.snap_worker_left.isRunning():
            self.stop_video_streaming_left()
        if self.snap_worker_right and self.snap_worker_right.isRunning():
            self.stop_video_streaming_right()

    def save_raw_image(self, raw_image, camera_id, original_image_path, show_image=False):
        """
        Saves a raw image from a specified camera to a dated subdirectory.
        """
        if raw_image is None:
            print("[Error][Processor] Cannot save None image.")
            return None

        # Print current working directory
        import os
        print(f"[Debug] Current working directory: {os.getcwd()}")

        # Force absolute path
        base_save_dir = os.path.abspath(r"C:\BoardDefectChecker\images\raw-images")
        print(f"[Debug] Using absolute base dir: {base_save_dir}")

        date_str = datetime.now().strftime("%d-%m-%Y")
        sub_dir = f"bezel-pwb-position-{date_str}"
        save_dir = os.path.join(base_save_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ext = os.path.splitext(original_image_path)[1] if original_image_path else ".png"
        new_filename = f"raw_{camera_id}_{time_str}{ext}"
        saved_path = os.path.join(save_dir, new_filename)

        # Print the final path
        print(f"[Debug] Attempting to save to absolute path: {os.path.abspath(saved_path)}")

        try:
            cv2.imwrite(saved_path, raw_image)
            print(f"[Info][Processor] Raw {camera_id} saved: {saved_path}")
        except Exception as e:
            print(f"[Error][Processor] Failed to save raw image to {saved_path}: {e}")
            return None

        # Always return the absolute path
        return os.path.abspath(saved_path)
