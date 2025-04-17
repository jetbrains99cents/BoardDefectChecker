from datetime import datetime
import os
import cv2
import psutil
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt, QTimer, QThread
from PySide6.QtWidgets import QMessageBox
from PySide6.QtCore import Slot

# Import the specialized Token FPC image-processing class
from modules.token_fpc_image_processing import TokenFPCImageProcessor
from modules.ai_models import TokenFPCFastImageSegmenter
from modules.language import SimpleLanguageManager
from modules.processing_working import ProcessingWorker, ProcessingDialog
from modules.disk_space_monitoring import DiskSpaceMonitor


class TokenFPCTabController:
    """
    Controller for the 'Token FPC insertion' tab.

    UI Elements used:
      - label_82: Displays camera connection status.
      - label_61: Displays config loading status.
      - label_62: Displays the part name.
      - label_90: Defect note header.
      - label_75: Defect check result header.
      - label_76: Processing time header.
      - pushButton_5: Toggles video streaming (Start/Stop).
      - pushButton_6: Opens camera settings.
      - label_87: Displays the video stream (450x450 at 180, 105).
      - lineEdit_6: QC Staff ID input.
      - lineEdit_4: Part number input.
      - lineEdit_5: Pin count input.
      - checkBox_7: Toggles counter.
      - checkBox_8: Toggles image display.
      - checkBox_9: Toggles histogram equalizer.
      - checkBox_10: Lock-options checkbox; also disables pushButton_5 & pushButton_6.
      - radioButton: For selecting English.
      - radioButton_2: For selecting Vietnamese.
      - label_58: Displays main camera connection status.
      - label_84: Displays config file loaded text.
      - label_79: Displays detected part count.
      - label_35: Displays detected NG part count.
      - label_72: Displays detected OK part count.
      - label_83: Displays final processed image (450x450 at 900, 105).
      - label_89: Displays defect note.
      - label_69: Displays final result.
      - label_67: Displays processing time.
      - label_63: Position varies by language (1070, 65 for en; 1000, 65 for vi).
      - label_98: Warning icon (blinks with detection result).
      - label_175: Disk space status.
      - label_187: Critical disk space warning (replaces dialog).
    """

    def __init__(self, ui, camera_worker, camera_interaction, snap_worker,
                 detection_log_worker=None, image_processor=None, parent=None):
        self.ui = ui
        self.parent = parent  # Store the parent widget (e.g., QMainWindow)
        self.camera_worker = camera_worker
        self.camera_interaction = camera_interaction
        self.snap_worker = snap_worker
        self.detection_log_worker = detection_log_worker
        self.old_image_processor = image_processor  # optional / legacy

        if self.detection_log_worker:
            self.detection_log_worker.log_result_signal.connect(self.on_log_result)

        # -------------------- Language Manager --------------------
        lang_path = os.path.join(os.path.dirname(__file__), "language.json")
        self.lang_manager = SimpleLanguageManager(lang_path)
        self.lang_manager.set_tab("token_fpc_tab")
        self.lang_manager.set_language("en")
        self.apply_translations()

        # Initialize disk space monitor
        self.disk_monitor = DiskSpaceMonitor(
            interval=5,
            language=self.lang_manager.current_language,
            low_space_threshold=5  # Updated threshold for testing
        )
        self.disk_monitor.disk_space_updated.connect(self.update_disk_space_label)
        self.disk_monitor.low_space_warning.connect(self.handle_low_space_warning)
        self.disk_monitor.critical_space_stop.connect(self.handle_critical_space)
        self.disk_monitor.start()

        # Set initial text for disk space and critical warning labels
        initial_text = "Dung lượng ổ đĩa: \nĐang tính toán..." if self.lang_manager.current_language == "vi" else "Free disk size: \nCalculating..."
        self.ui.label_175.setText(initial_text)
        self.ui.label_187.setText("--")  # Initialize label_187 to normal state
        self.ui.label_187.setStyleSheet("color: white;")  # Set to white in normal state

        # Blinking timer for label_175 and label_187 (disk space and critical warning)
        self.disk_blink_timer = QTimer()
        self.disk_blink_timer.timeout.connect(self.toggle_blink)
        self.is_blinking = False
        self.blink_visible = True
        self.is_critical = False
        # State tracking for logging
        self.was_low_space = False  # Track previous low space state
        self.was_critical = False  # Track previous critical state

        # Dedicated blinking timer for result labels
        self.result_blink_timer = QTimer()

        # Connect language change signal
        if hasattr(self.lang_manager, 'language_changed'):
            self.lang_manager.language_changed.connect(self.update_language)

        # -------------------- Camera Initialization --------------------
        # Connect snap_worker's signals first
        if self.snap_worker:
            self.snap_worker.image_captured.connect(self.update_video_frame)
            self.snap_worker.devices_available.connect(self.populate_combo_box)

        # Start camera_worker and ensure it's running before enumerating devices
        if hasattr(self.camera_worker, "connection_status_changed"):
            self.camera_worker.connection_status_changed.connect(self.update_connection_status)
            if not self.camera_worker.isRunning():
                self.camera_worker.start()

        # Force device enumeration on startup and wait briefly for it to complete
        if self.snap_worker:
            print("[TokenFPC] Forcing device enumeration on startup...")
            self.snap_worker.enumerate_devices()  # Trigger enumeration immediately
            QThread.msleep(500)  # Wait 500ms for device list to populate

        # Disable checkbox 11 - Option for auto deleting files when full disk
        self.ui.checkBox_11.setEnabled(False)

        # Disable checkbox 9 - We needn't equalize histogram this AOI type
        self.ui.checkBox_9.setEnabled(False)

        # Our specialized Token FPC image processor
        self.token_fpc_image_processor = TokenFPCImageProcessor()

        # Internal state
        self.is_counter_turned_on = False
        self.is_processed_images_shown = False
        self.is_histogram_equalized = False
        self.is_options_locked = False
        self.target_pin_count = 12
        self.current_connection_state = None
        self.last_selected_index = None  # Track the last used camera index

        # Load a Token FPC AI model without conf and iou
        self.token_fpc_ai_model_runner = TokenFPCFastImageSegmenter(
            model_type="x",
            model_path="ai-models/",
            output_directory=r"C:\BoardDefectChecker\ai-outputs",
            is_image_shown=False,
            left_right_offset=50,
            connector_offset=30,
            fpc_two_marks_height_diff=9
        )

        # Set up log directory for detection_log_worker
        if self.detection_log_worker:
            self.update_log_directory()

        # -------------------- Camera & Streaming --------------------
        self.ui.pushButton_6.clicked.connect(self.open_camera_settings)
        self.ui.pushButton_5.clicked.connect(self.toggle_video_streaming)
        self.ui.comboBox_2.currentIndexChanged.connect(self.on_camera_selection_changed)

        if self.snap_worker:
            self.snap_worker.image_captured.connect(self.update_video_frame)

        if hasattr(self.camera_worker, "connection_status_changed"):
            self.camera_worker.connection_status_changed.connect(self.update_connection_status)
            if not self.camera_worker.isRunning():
                self.camera_worker.start()

        # If you want the old image_processor’s config:
        if self.old_image_processor and hasattr(self.old_image_processor, "config_loaded"):
            self.old_image_processor.config_loaded.connect(self.load_config)

        # -------------------- UI Controls --------------------
        self.ui.lineEdit_6.returnPressed.connect(self.update_staff_id)
        self.ui.lineEdit_4.returnPressed.connect(self.handle_part_number_return_pressed)
        self.ui.lineEdit_4.textChanged.connect(self.handle_text_changed)

        self.ui.checkBox_7.stateChanged.connect(self.toggle_counter)
        self.ui.checkBox_8.stateChanged.connect(self.toggle_image_displayed)
        self.ui.checkBox_9.stateChanged.connect(self.toggle_histogram_equalizer)
        self.ui.checkBox_10.stateChanged.connect(self.toggle_options_lock)

        # -------------------- Language Selection --------------------
        self.ui.radioButton.toggled.connect(self.on_english_selected)
        self.ui.radioButton_2.toggled.connect(self.on_vietnamese_selected)

        # -------------------- Initialize Status Labels --------------------
        self.ui.label_82.setText("Disconnected")
        self.ui.label_82.setStyleSheet("color: #be2b25;")
        self.ui.label_61.setText("Failed")
        self.ui.label_62.setText("--")
        self.update_detected_result_count()

        # By default, enable counter, auto deleting on low disk size mechanism and checking without JIG options
        self.ui.checkBox_7.setChecked(True)
        self.ui.checkBox_11.setChecked(True)
        self.ui.checkBox_13.setChecked(True)

        self.toggle_counter(2)

    # -------------------- Translation --------------------
    def apply_translations(self):
        """Update translatable UI elements for this tab using keys from 'token_fpc_tab' in language.json."""
        translatable_widgets = {
            "label_87": self.ui.label_87,
            "label_83": self.ui.label_83,
            "label_73": self.ui.label_73,
            "label_63": self.ui.label_63,
            "label_90": self.ui.label_90,
            "label_75": self.ui.label_75,
            "label_76": self.ui.label_76,
            "label_61": self.ui.label_61,
            "label_77": self.ui.label_77,
            "label_58": self.ui.label_58,
            "label_84": self.ui.label_84,
            "label_79": self.ui.label_79,
            "label_35": self.ui.label_35,
            "label_72": self.ui.label_72,
            "pushButton_5": self.ui.pushButton_5,
            "pushButton_6": self.ui.pushButton_6,
            "checkBox_7": self.ui.checkBox_7,
            "checkBox_8": self.ui.checkBox_8,
            "checkBox_9": self.ui.checkBox_9,
            "checkBox_10": self.ui.checkBox_10,
            "label_78": self.ui.label_78,
            "label_74": self.ui.label_74,
            "checkBox_11": self.ui.checkBox_11,
            "label_175": self.ui.label_175,
            "label_178": self.ui.label_178,
            "label_180": self.ui.label_180,
            "checkBox_13": self.ui.checkBox_13
        }
        for key, widget in translatable_widgets.items():
            widget.setText(self.lang_manager.get_text(key))
        print("Translations applied for tab:", self.lang_manager.current_tab,
              "language:", self.lang_manager.current_language)

    def on_english_selected(self, checked):
        if checked:
            print("English selected")
            self.lang_manager.set_language("en")
            self.apply_translations()
            self.ui.label_63.setGeometry(1070, 65, 271, 30)  # Position for English
            self.update_connection_status(
                self.current_connection_state if hasattr(self, 'current_connection_state') else False
            )
            # Update tab names for English
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_2), "Settings")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_3), "Inspection log")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_4), "AOI machine")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_5_token_fpc_insertion),
                                         "TP FPC checker")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_6_mounting), "FPC mounting machine")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_benzel_pwb_position),
                                         "Bezel - PWB position checker")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_small_fpc_insertion), "LED FPC checker")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_optional_aoi), "Optional AOI checker")

    def on_vietnamese_selected(self, checked):
        if checked:
            print("Vietnamese selected")
            self.lang_manager.set_language("vi")
            self.apply_translations()
            self.ui.label_63.setGeometry(1000, 65, 271, 30)  # Position for Vietnamese
            self.update_connection_status(
                self.current_connection_state if hasattr(self, 'current_connection_state') else False
            )
            # Update tab names for Vietnamese
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_2), "Cài đặt")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_3), "Nhật ký kiểm tra")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_4), "Máy kiểm AOI tự động")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_5_token_fpc_insertion), "Kiểm TP FPC")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_6_mounting), "Máy gắn FPC tự động")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_benzel_pwb_position),
                                         "Kiểm bẻ ngàm - dán PWB")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_small_fpc_insertion), "Kiểm LED FPC")
            self.ui.tabWidget.setTabText(self.ui.tabWidget.indexOf(self.ui.tab_optional_aoi), "Kiểm AOI tùy chọn")

    # -------------------- Camera & Streaming --------------------
    @Slot(list)
    def populate_combo_box(self, device_info_list):
        self.ui.comboBox_2.clear()
        if not device_info_list:
            self.ui.comboBox_2.addItem("No devices available")
            print("No devices available to populate comboBox_2")
        else:
            for display_text, dev_info in device_info_list:
                self.ui.comboBox_2.addItem(display_text, dev_info)
            print(f"Populated comboBox_2 with {len(device_info_list)} devices")
            # Auto-select the first device if available
            if len(device_info_list) > 0:
                self.ui.comboBox_2.setCurrentIndex(0)
                print(f"Auto-selected first device: {device_info_list[0][0]}")

    def open_camera_settings(self):
        if self.camera_interaction:
            self.camera_interaction.select_device()
        else:
            QMessageBox.warning(None, "Camera Settings", "Camera interaction module not available.")

    def toggle_video_streaming(self):
        """Toggle video streaming on/off based on current state and selected camera."""
        if self.snap_worker:
            current_index = self.ui.comboBox_2.currentIndex()
            if self.snap_worker.isRunning():
                # If streaming, stop it regardless of selection change
                self.stop_video_streaming()
                # If selection changed, start the new camera
                if current_index != self.last_selected_index:
                    self.start_video_streaming()
            else:
                # If not streaming, start with the current selection
                self.start_video_streaming()
            self.last_selected_index = current_index

    def start_video_streaming(self):
        """Start video streaming with the selected camera and update button text."""
        if self.snap_worker and not self.snap_worker.isRunning():
            selected_index = self.ui.comboBox_2.currentIndex()
            selected_text = self.ui.comboBox_2.currentText()

            # Check if no device is selected or the selection is invalid
            if selected_index == -1 or selected_text == "No devices available":
                title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
                message = ("Vui lòng chọn một thiết bị camera từ danh sách trước khi bắt đầu streaming.\n"
                           "Please select a camera device from the list before starting streaming.")
                QMessageBox.warning(None, title, message)
                print("Streaming attempt failed: No device selected in comboBox_2")
                return

            print(f"Starting video streaming with device at index {selected_index}")
            try:
                self.snap_worker.open_device(selected_index)
                self.snap_worker.pre_process()
                self.snap_worker.start()
                # Update button text and UI state only on success
                self.ui.pushButton_5.setText(
                    "Dừng truyền video" if self.lang_manager.current_language == "vi" else "Stop video streaming"
                )
                self.ui.comboBox_2.setEnabled(False)  # Lock selection while streaming
                print("Video streaming started successfully.")
            except Exception as e:
                title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
                message = (f"Không thể bắt đầu streaming: {str(e)}\n"
                           f"Failed to start streaming: {str(e)}")
                QMessageBox.critical(None, title, message)
                print(f"Failed to start video streaming: {e}")
        else:
            print("Video streaming already running or snap_worker is None.")

    def stop_video_streaming(self):
        """Stop video streaming and update button text."""
        if self.snap_worker:
            was_running = self.snap_worker.isRunning()
            try:
                if was_running:
                    self.snap_worker.stop()
                    print("Video streaming stopped successfully.")
            except Exception as e:
                title = "Lỗi / Error" if self.lang_manager.current_language == "vi" else "Error"
                message = (f"Không thể dừng streaming: {str(e)}\n"
                           f"Failed to stop streaming: {str(e)}")
                QMessageBox.critical(None, title, message)
                print(f"Failed to stop video streaming: {e}")
            finally:
                # Ensure button text reflects stopped state if worker is not running
                if not self.snap_worker.isRunning():
                    self.ui.pushButton_5.setText(self.lang_manager.get_text("pushButton_5"))
                    self.ui.comboBox_2.setEnabled(True)  # Unlock selection after stopping
        else:
            print("Snap_worker is None, nothing to stop.")

    @Slot(int)
    def on_camera_selection_changed(self, index):
        """Handle camera selection change while streaming."""
        if self.snap_worker and self.snap_worker.isRunning():
            print(f"Camera selection changed to index {index} while streaming, restarting with new camera.")
            self.toggle_video_streaming()

    def update_video_frame(self, frame):
        """Receives a frame (NumPy array in RGB) and updates label_87 with a 450x450 image at (180, 105)."""
        if frame is None:
            return
        h, w, c = frame.shape
        bytes_per_line = c * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(450, 450, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        self.ui.label_87.setGeometry(180, 105, 450, 450)
        self.ui.label_87.setScaledContents(True)
        self.ui.label_87.setPixmap(scaled_pixmap)

    def update_connection_status(self, is_connected):
        self.current_connection_state = is_connected
        if is_connected:
            self.ui.label_82.setText(self.lang_manager.get_text("label_82_connected"))
            self.ui.label_82.setStyleSheet("color: #25be7c;")
            print("Camera Connected - label_82 updated")
        else:
            self.ui.label_82.setText(self.lang_manager.get_text("label_82_disconnected"))
            self.ui.label_82.setStyleSheet("color: #be2b25;")
            print("Camera Disconnected - label_82 updated")
            if self.snap_worker and self.snap_worker.isRunning():
                self.stop_video_streaming()
            # Force button text to "Start" when disconnected, even if stop fails
            if self.snap_worker and not self.snap_worker.isRunning():
                self.ui.pushButton_5.setText(self.lang_manager.get_text("pushButton_5"))
                self.ui.comboBox_2.setEnabled(True)

        # Update device list regardless of connection state change
        if self.snap_worker:
            self.snap_worker.enumerate_devices()

    # -------------------- Config & Count Methods --------------------
    def load_config(self, config_data):
        """If the old image_processor fires config_loaded, we forward to TokenFPCImageProcessing."""
        self.ui.label_61.setText("Successful")
        part_name = config_data.get("part-name", "Unknown Part")
        self.ui.label_62.setText(part_name)
        print(f"Config loaded - label_61 updated, part name: {part_name}")

        # Forward config to specialized processor
        self.token_fpc_image_processor.load_config(config_data)

    def update_detected_result_count(self):
        """Update the UI labels with the current detection counts."""
        if self.is_counter_turned_on and self.detection_log_worker:
            self.ui.label_59.setText(str(self.detection_log_worker.detected_part_count))  # Total count
            self.ui.label_66.setText(str(self.detection_log_worker.detected_ng_count))  # NG count
            self.ui.label_71.setText(str(self.detection_log_worker.detected_ok_count))  # OK count
        else:
            self.ui.label_59.setText("0")
            self.ui.label_66.setText("0")
            self.ui.label_71.setText("0")

    # -------------------- ID, Part Number, Pin Count --------------------
    def update_staff_id(self):
        staff_id = self.ui.lineEdit_6.text().strip()
        if not staff_id:
            QMessageBox.warning(None, "Input Error", "Staff ID cannot be empty.")
            return
        print(f"Updating staff ID to: {staff_id}")
        if self.detection_log_worker:
            self.detection_log_worker.set_staff_id(staff_id)
        QMessageBox.information(None, "Staff ID Updated", f"Staff ID updated to: {staff_id}")

    def handle_part_number_return_pressed(self):
        part_number = self.ui.lineEdit_4.text().strip()
        if part_number:
            print(f"Part number entered: {part_number}")
            self.handle_capture_and_log()

    def handle_text_changed(self):
        part_number = self.ui.lineEdit_4.text().strip()
        if '\n' in part_number:
            self.handle_capture_and_log()

    # -------------------- Helper for showing images at 640×480 --------------------
    def show_640x480(self, window_title: str, image):
        """Resize 'image' to 640×480 then show using OpenCV."""
        if image is None:
            return
        # Force 640x480
        display_img = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
        cv2.imshow(window_title, display_img)
        cv2.waitKey(0)

    # -------------------- Disk Space Management --------------------
    def toggle_blink(self):
        """Toggle visibility of label_175 and label_187 for disk space blinking effect."""
        self.blink_visible = not self.blink_visible
        # Blink both label_175 and label_187 when critical or low space
        if self.is_critical or self.is_blinking:
            color = "#be2b25"  # Red for critical or low space
            self.ui.label_175.setStyleSheet(f"color: {'transparent' if not self.blink_visible else color};")
            self.ui.label_187.setStyleSheet(f"color: {'transparent' if not self.blink_visible else color};")
        else:
            self.ui.label_175.setStyleSheet("color: #25be7c;")  # Green for normal state
            self.ui.label_187.setStyleSheet("color: white;")  # White for normal state

    @Slot(str)
    def update_disk_space_label(self, text):
        """Update label_175 with disk space info."""
        self.ui.label_175.setText(text)
        if not self.is_blinking and not self.is_critical:
            self.ui.label_175.setStyleSheet("color: #25be7c;")  # Green when normal

    @Slot(bool)
    def handle_low_space_warning(self, is_low):
        """Handle low space warning: blink labels red, log only on state change."""
        if is_low and not self.is_blinking:
            self.is_blinking = True
            self.disk_blink_timer.start(500)
            if not self.was_low_space:  # Log only on transition to low space
                print("[TokenFPC] Low disk space warning activated")
                self.was_low_space = True
        elif not is_low and not self.is_critical:
            self.is_blinking = False
            self.disk_blink_timer.stop()
            self.ui.label_175.setStyleSheet("color: #25be7c;")  # Green when normal
            self.ui.label_187.setStyleSheet("color: white;")  # White when normal
            if self.was_low_space:  # Log only on transition out of low space
                print("[TokenFPC] Low disk space warning deactivated")
                self.was_low_space = False

    @Slot(bool)
    def handle_critical_space(self, is_critical):
        """Handle critical space: stop software, update label_187, log only on state change."""
        self.is_critical = is_critical
        if is_critical:
            self.is_blinking = True
            self.disk_blink_timer.start(500)
            # Disable UI
            self.ui.tabWidget.setEnabled(False)  # Assuming tabWidget is the container
            if not self.was_critical:  # Log only on transition to critical
                print("[TokenFPC] Critical disk space: Software fully stopped")
                self.was_critical = True
            # Update label_187 with the critical message based on language
            if self.lang_manager.current_language == "vi":
                self.ui.label_187.setText("Xóa bớt dung lượng \nổ cứng")
            else:
                self.ui.label_187.setText("Delete some files \nto free disk size")

        elif not is_critical:
            total_free = sum(psutil.disk_usage(drive).free for drive in self.disk_monitor.get_all_drives())
            if total_free < self.disk_monitor.warning_threshold:
                self.is_blinking = True  # Keep blinking if in warning range
            else:
                self.is_blinking = False
                self.disk_blink_timer.stop()
                self.ui.label_175.setStyleSheet("color: #25be7c;")  # Green when normal
                self.ui.label_187.setStyleSheet("color: white;")  # White when normal
            # Re-enable UI
            self.ui.tabWidget.setEnabled(True)
            if self.was_critical:  # Log only on transition out of critical
                print("[TokenFPC] Critical disk space resolved, software resumed")
                self.was_critical = False
            # Reset label_187 to normal state
            self.ui.label_187.setText("--")
            self.ui.label_187.setStyleSheet("color: white;")

    @Slot(str)
    def update_language(self, new_language):
        """Update language in DiskSpaceMonitor and UI."""
        self.disk_monitor.language = new_language
        drives = self.disk_monitor.get_all_drives()
        total_free = sum(psutil.disk_usage(drive).free for drive in drives if psutil.disk_usage(drive))
        total_capacity = sum(psutil.disk_usage(drive).total for drive in drives)
        total_free_gb = total_free / (1024 ** 3)
        total_capacity_gb = total_capacity / (1024 ** 3)
        text = (f"Dung lượng ổ đĩa: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB" if new_language == "vi"
                else f"Free disk size: \n{total_free_gb:.0f} GB/{total_capacity_gb:.0f} GB")
        self.update_disk_space_label(text)

    # -------------------- Log Directory Management --------------------
    def update_log_directory(self):
        """Set the log directory to token-fpc-dd-mm-yyyy format inside operation-logs."""
        if not self.detection_log_worker:
            return

        # Base directory updated to operation-logs
        base_dir = r"C:\BoardDefectChecker\operation-logs"
        current_date = datetime.now().strftime("%d-%m-%Y")
        log_dir = os.path.join(base_dir, f"token-fpc-{current_date}")

        print(f"[TokenFPC] Log directory set to: {log_dir}")

        # Use the setter method instead of direct attribute access
        self.detection_log_worker.set_log_directory(log_dir)

    # -------------------- Main Capture/Log --------------------
    def handle_capture_and_log(self):
        """Disable processing if critical."""
        if self.is_critical:
            print("[TokenFPC] Cannot process: Insufficient disk space")
            return

        print("[TokenFPC] handle_capture_and_log triggered.")

        # Update log directory before processing to ensure logs go to the correct folder
        self.update_log_directory()

        # Reset result labels and ensure they are visible
        self.ui.label_69.setText("--")
        self.ui.label_67.setText("--")
        self.ui.label_89.setText("--")
        self.ui.label_69.setVisible(True)
        self.ui.label_89.setVisible(True)
        self.ui.label_98.setVisible(True)

        # Display blank image for processed result
        blank_path = r"C:\BoardDefectChecker\resources\blank.png"
        self.display_processed_image(blank_path)

        # Retrieve and trim input values for part number and staff ID
        part_number = self.ui.lineEdit_4.text().strip()
        staff_id = self.ui.lineEdit_6.text().strip()

        # Check for newline characters in part_number
        if "\n" in part_number:
            part_number = part_number.split("\n")[0].strip()
            if not part_number:
                QMessageBox.warning(None, "Input Error", "Part number cannot be empty.")
                return

        # Validate inputs
        if not part_number:
            QMessageBox.warning(None, "Input Error", "Part number cannot be empty.")
            return
        if not staff_id:
            QMessageBox.warning(None, "Input Error", "Staff ID cannot be empty.")
            return

        # Update staff ID if changed
        if self.detection_log_worker.get_staff_id() != staff_id:
            self.detection_log_worker.set_staff_id(staff_id)
            if self.is_counter_turned_on:
                self.update_detected_result_count()
            print(f"Updated staff ID to: {staff_id}")

        # 1) Capture image via snap_worker
        raw_image_path = None
        if self.snap_worker:
            raw_image_path = self.snap_worker.capture_single_image(save_image=True)
        if not raw_image_path:
            QMessageBox.warning(None, "Capture Error", "Failed to capture image.")
            return

        # 2) Save raw image into a dated subdirectory and delete original file
        raw_img = cv2.imread(raw_image_path)
        if raw_img is None:
            QMessageBox.warning(None, "Capture Error", "Failed to load captured image.")
            return
        saved_raw_path = self.token_fpc_image_processor.save_raw_image(
            raw_img, original_image_path=raw_image_path, show_image=False
        )
        if saved_raw_path:
            try:
                os.remove(raw_image_path)
                print(f"[TokenFPC] Original raw image deleted: {raw_image_path}")
            except Exception as e:
                print(f"[TokenFPC] Failed to delete raw image: {e}")
            raw_image_path = saved_raw_path

        # 3) Start processing in a separate thread with a loading dialog
        self.processing_worker = ProcessingWorker(
            self.token_fpc_ai_model_runner,
            raw_image_path,
            self.detection_log_worker,
            part_number,
            staff_id,
            self.is_counter_turned_on
        )
        self.processing_dialog = ProcessingDialog(None)
        self.processing_worker.processing_finished.connect(self.on_processing_finished)
        self.processing_worker.finished.connect(self.processing_dialog.accept)
        self.processing_worker.start()
        self.processing_dialog.exec_()

        print("[TokenFPC] handle_capture_and_log finished for part:", part_number)

    def on_processing_finished(self, visualized_image, bb_path, processing_time, detection_result, defect_reason,
                               raw_image_path):
        if self.is_critical:
            print("[TokenFPC] Cannot process results: Insufficient disk space")
            return

        if visualized_image is None or bb_path is None:
            print("[TokenFPC] AI processing failed.")
            return

        print(
            f"[TokenFPC] Received from worker - Detection result: '{detection_result}', Defect reason: '{defect_reason}'")

        time_suffix = " seconds" if self.lang_manager.current_language == "en" else " giây"
        self.ui.label_67.setText(f"{processing_time:.2f}{time_suffix}")

        detection_result = str(detection_result).strip()

        if self.lang_manager.current_language == "vi":
            if detection_result == "OK":
                defect_note = "Không có lỗi thao tác gắn TOKEN FPC"
            elif defect_reason:
                if ("Connector mask not found" in defect_reason or
                        "Connector mask not found for distance check" in defect_reason or
                        "Connector height" in defect_reason):
                    defect_note = "Nắp khóa chưa đóng"
                elif ("Left or right FPC mask not found" in defect_reason or
                      "Unbalanced FPC" in defect_reason or
                      "Distance to connector exceeds" in defect_reason):
                    defect_note = "Cáp FPC bị lệch hoặc chưa được gắn chặt"
                else:
                    defect_note = defect_reason
            else:
                defect_note = "Unknown defect"
        else:
            defect_note = "Matched insertion" if detection_result == "OK" else (defect_reason or "Unknown defect")

        if detection_result == "OK":
            self.ui.label_69.setText("OK")
            self.ui.label_69.setStyleSheet("color: #25be7c;")
            self.ui.label_89.setText(defect_note)
            self.ui.label_89.setStyleSheet("color: #25be7c;")
            self.ui.label_98.setStyleSheet("color: #25be7c;")
        else:
            self.ui.label_69.setText("NG")
            self.ui.label_69.setStyleSheet("color: #be2b25;")
            self.ui.label_89.setText(defect_note)
            self.ui.label_89.setStyleSheet("color: #be2b25;")
            self.ui.label_98.setStyleSheet("color: #be2b25;")

        # Ensure labels are visible before blinking
        self.ui.label_69.setVisible(True)
        self.ui.label_89.setVisible(True)
        self.ui.label_98.setVisible(True)

        self.blink_labels()
        self.display_processed_image(bb_path)

        # Ensure labels remain visible after blinking
        self.ui.label_69.setVisible(True)
        self.ui.label_89.setVisible(True)
        self.ui.label_98.setVisible(True)

    @Slot(str, int, int, int)
    def on_log_result(self, log_message, total_count, ng_count, ok_count):
        if self.is_critical:
            return
        if self.is_counter_turned_on:
            self.ui.label_59.setText(str(total_count))
            self.ui.label_66.setText(str(ng_count))
            self.ui.label_71.setText(str(ok_count))
            print(f"[TokenFPC] Counter updated - Total: {total_count}, OK: {ok_count}, NG: {ng_count}")

    def __del__(self):
        """Clean up the disk monitor and timers."""
        if hasattr(self, 'disk_monitor'):
            self.disk_monitor.stop()
        if hasattr(self, 'disk_blink_timer'):
            self.disk_blink_timer.stop()
        if hasattr(self, 'result_blink_timer'):
            self.result_blink_timer.stop()
        if self.snap_worker and self.snap_worker.isRunning():
            self.stop_video_streaming()

    # -------------------- Blinking Animation Method --------------------
    def blink_labels(self):
        """Blink label_69, label_89, and label_98 (warning icon) for 2 seconds."""
        labels = [self.ui.label_69, self.ui.label_89, self.ui.label_98]
        max_blinks = 4  # 2 seconds / 0.5 seconds per toggle = 4 toggles

        # Stop any existing timer to avoid overlap
        if self.result_blink_timer.isActive():
            self.result_blink_timer.stop()
            print("[TokenFPC] Stopped existing result blink timer.")

        # Local blink state
        blink_count = 0
        visible = True

        def toggle():
            nonlocal blink_count, visible
            visible = not visible
            for label in labels:
                label.setVisible(visible)
            blink_count += 1
            if blink_count >= max_blinks:
                self.result_blink_timer.stop()
                self.result_blink_timer.timeout.disconnect()  # Clean up connection
                for label in labels:
                    label.setVisible(True)  # Ensure final state
                print("[TokenFPC] Result label blinking completed.")

        # Connect and start the dedicated timer
        self.result_blink_timer.timeout.connect(toggle)
        self.result_blink_timer.start(500)
        print("[TokenFPC] Started blinking result labels.")

    # -------------------- Toggle Methods --------------------
    def toggle_counter(self, state):
        self.is_counter_turned_on = (state == 2)
        print(f"[TokenFPC] Counter turned {'ON' if self.is_counter_turned_on else 'OFF'}")
        if self.is_options_locked:
            print("Counter toggle denied - options locked")
            QMessageBox.information(None, "Message", "Options are locked.")

    def toggle_image_displayed(self, state):
        self.is_processed_images_shown = (state == 2)
        self.token_fpc_image_processor.is_images_shown = (state == 2)
        print(f"[TokenFPC] Show processed images: {self.is_processed_images_shown}")
        if self.is_options_locked:
            QMessageBox.information(None, "Message", "Options are locked.")

    def toggle_histogram_equalizer(self, state):
        self.is_histogram_equalized = (state == 2)
        if self.is_options_locked:
            print("Histogram equalizer toggle denied - options locked")
            QMessageBox.information(None, "Message", "Options are locked.")
        else:
            if self.is_histogram_equalized:
                print("[TokenFPC] Using histogram equalizer now.")
            else:
                print("[TokenFPC] Not using histogram equalizer.")

    def toggle_options_lock(self, state):
        self.is_options_locked = (state == 2)
        if self.is_options_locked:
            print("[TokenFPC] Options lock turned on.")
            self.ui.checkBox_7.setEnabled(False)
            self.ui.checkBox_8.setEnabled(False)
            self.ui.checkBox_9.setEnabled(False)
            self.ui.checkBox_10.setEnabled(False)
            self.ui.pushButton_5.setEnabled(False)
            self.ui.pushButton_6.setEnabled(False)
            QMessageBox.information(None, "Message", "Options are locked.")

    # -------------------- Display Processed Images Methods --------------------
    def display_processed_image(self, image_path):
        """Display the final processed image from the provided image path at (900, 105) with size 450x450."""
        pixmap = QPixmap(image_path)
        self.ui.label_83.setPixmap(pixmap)
        self.ui.label_83.setScaledContents(True)
        self.ui.label_83.setGeometry(900, 105, 450, 450)
        self.ui.label_83.show()