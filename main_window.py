import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import QMainWindow, QMessageBox
from modules.camera_connection import CameraWorker  # Import the camera worker
from modules.camera_interaction import CameraInteraction, SnapSinkWorker  # Import the camera interaction
from modules.image_processing import ImageProcessor
from modules.main_ui_design import Ui_MainWindow
from modules.ai_models import SmallFPCFastImageSegmenter, ModelManager
from modules.token_fpc_tab import TokenFPCTabController  # Your separate tab logic
from modules.bezel_pwb_position_tab import BezelPWBTabController
from modules.language import SimpleLanguageManager
from datetime import datetime
import os
import imagingcontrol4 as ic4


class MainWindow(QMainWindow):
    def __init__(self, detection_log_worker):
        super().__init__()

        # --- >> ADDED BLOCK for Window Icon (Using Absolute Path) << ---
        self.setWindowTitle("Board Defect Checker - v.1.0.1904")  # Set your desired window title here
        try:
            # --- Use the specific absolute path provided ---
            absolute_icon_path = r"C:\BoardDefectChecker\resources\software_icon.ico"  # Raw string for Windows path

            if os.path.exists(absolute_icon_path):
                # QIcon is already imported at the top of the file
                app_icon = QIcon(absolute_icon_path)
                self.setWindowIcon(app_icon)
                print(f"Window icon set successfully from: {absolute_icon_path}")
            else:
                print(f"Warning: Icon file not found at the specified path: {absolute_icon_path}")

        except Exception as e:
            print(f"Error setting window icon: {e}")
        # --- End of Added Block ---

        self.display_widget = None
        self.snap_worker = None
        self.current_connection_state = None
        self.ui = Ui_MainWindow()  # Assuming you have a setup method for the UI
        self.ui.setupUi(self)

        self.ui.label_14.setStyleSheet("color: #be2b25;")  # Color for disconnected
        self.ui.label_14.setText("Disconnected")  # Initialize label

        # Disable checkbox 5 - Option for auto deleting file when full disk
        self.ui.checkBox_5.setEnabled(False)

        """Display company logo"""
        sharp_logo_pixmap = QPixmap(r"C:\BoardDefectChecker\resources\sharp_logo.png")
        self.ui.label_28.setPixmap(sharp_logo_pixmap)
        self.ui.label_28.setGeometry(1500, 0, 91, 71)
        self.ui.label_28.show()

        self.ui.label_36.setPixmap(sharp_logo_pixmap)
        self.ui.label_36.setGeometry(1500, 0, 91, 71)
        self.ui.label_36.show()

        self.ui.label_68.setPixmap(sharp_logo_pixmap)
        self.ui.label_68.setGeometry(1500, 0, 91, 71)
        self.ui.label_68.show()

        self.ui.label_309.setPixmap(sharp_logo_pixmap)
        self.ui.label_309.setGeometry(1500, 0, 91, 71)
        self.ui.label_309.show()

        """Display other logos"""
        # arrow_up_pixmap = QPixmap(r"C:\BoardDefectChecker\resources\arrow-up.png")
        # self.ui.label_124.setPixmap(arrow_up_pixmap)
        # self.ui.label_124.setGeometry(640, 140, 65, 65)
        # self.ui.label_124.show()
        #
        # arrow_down_pixmap = QPixmap(r"C:\BoardDefectChecker\resources\arrow-down.png")
        # self.ui.label_126.setPixmap(arrow_down_pixmap)
        # self.ui.label_126.setGeometry(640, 320, 65, 65)
        # self.ui.label_126.show()
        #
        # arrow_left_pixmap = QPixmap(r"C:\BoardDefectChecker\resources\arrow-left.png")
        # self.ui.label_127.setPixmap(arrow_left_pixmap)
        # self.ui.label_127.setGeometry(770, 375, 65, 65)
        # self.ui.label_127.show()
        #
        # arrow_right_pixmap = QPixmap(r"C:\BoardDefectChecker\resources\arrow-right.png")
        # self.ui.label_131.setPixmap(arrow_right_pixmap)
        # self.ui.label_131.setGeometry(940, 375, 65, 65)
        # self.ui.label_131.show()

        # Display widget position for displaying video stream from camera
        # self.display_widget = ic4.pyside6.DisplayWidget()  # Create the display widget
        # Set the geometry so that x=520, y=100, width=250, height=250
        # self.display_widget.setGeometry(520, 100, 250, 250)
        # Fix the widget size so that it doesn't resize
        # self.display_widget.setFixedSize(250, 250)
        # self.display_widget.show()
        # self.display_handle = None
        # Delay setting up the display handle until after the event loop starts.
        # QTimer.singleShot(100, self.setup_display)

        """Display button icon"""
        self.ui.pushButton_12.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-up.png"))
        self.ui.pushButton_13.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-down.png"))
        self.ui.pushButton_14.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-left.png"))
        self.ui.pushButton_15.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-right.png"))
        self.ui.pushButton_17.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-up.png"))
        self.ui.pushButton_18.setIcon(QIcon(r"C:\BoardDefectChecker\resources\arrow-down.png"))

        # Target pin count
        self.target_pin_count = 12

        # Using histogram equalizer
        self.is_histogram_equalized = False
        self.current_checkbox_3_state = False
        self.ui.checkBox_3.stateChanged.connect(self.toggle_histogram_equalizer)

        # Counter turn on check box
        self.ui.checkBox.stateChanged.connect(self.toggle_counter)
        self.is_counter_turned_on = False
        self.current_checkbox_state = False

        # Image displaying check box
        self.ui.checkBox_2.stateChanged.connect(self.toggle_image_displayed)
        self.is_processed_images_shown = False
        self.current_checkbox_2_state = False

        # Lock options to prevent interference from user
        self.is_options_locked = False
        self.ui.checkBox_4.stateChanged.connect(self.toggle_options_lock)

        # Store the detection log worker reference
        self.detection_log_worker = detection_log_worker

        # Update detected part count, ng part count, ok part count on the UI
        self.update_detected_result_count()

        # Initialize camera worker
        self.camera_worker = CameraWorker()
        self.camera_worker.connection_status_changed.connect(self.update_connection_status)
        self.camera_worker.start()  # Start the camera worker

        # Initialize camera interaction
        self.camera_interaction = CameraInteraction()

        # Initialize image processing worker
        self.image_processor = ImageProcessor()

        # Initialize AI model runner
        self.small_fpc_ai_model_runner = SmallFPCFastImageSegmenter(model_type='x',
                                                                    model_path="D:\\Working\\BoardDefectChecker\\ai-models\\",
                                                                    angle_difference_threshold=1.2)
        self.model_manager = ModelManager()

        # Create the display widget and set it to the position of graphicsView
        # self.display_widget = ic4.pyside6.DisplayWidget()  # Create the display widget
        # self.display_widget.setGeometry(self.ui.graphicsView.geometry())  # Set the geometry to match graphicsView
        # self.setCentralWidget(self.display_widget)  # Set the display widget as the central widget

        # Connect the returnPressed signal
        self.ui.lineEdit.returnPressed.connect(self.update_staff_id)  # Connect to update staff ID
        self.ui.lineEdit_2.returnPressed.connect(self.handle_return_pressed)
        self.ui.lineEdit_3.returnPressed.connect(self.update_target_pin_count)
        self.ui.lineEdit_2.textChanged.connect(self.handle_text_changed)

        # Connect signals for image capture
        # self.ui.lineEdit_2.returnPressed.connect(self.handle_return_pressed())
        # self.ui.lineEdit_2.textChanged.connect(self.handle_text_changed())

        # Create separate SnapSinkWorker instances for each tab
        self.snap_worker_token = SnapSinkWorker()  # For Token FPC tab
        self.snap_worker_left = SnapSinkWorker()  # For Bezel-PWB left camera
        self.snap_worker_right = SnapSinkWorker()  # For Bezel-PWB right camera

        # Button for opening camera setting
        self.ui.pushButton.clicked.connect(self.camera_interaction.select_device)

        # Button for starting video streaming
        self.ui.pushButton_2.clicked.connect(self.start_video_streaming)

        # Create the Token FPC tab controller with references
        self.token_fpc_tab_controller = TokenFPCTabController(
            self.ui,
            camera_worker=self.camera_worker,  # from your main window
            camera_interaction=self.camera_interaction,  # from your main window
            snap_worker=self.snap_worker_token,  # Dedicated worker
            detection_log_worker=self.detection_log_worker,  # if available
            image_processor=self.image_processor,  # if available
            parent=Ui_MainWindow
        )

        # Create the Bezel - PWB position tab controller with two distinct workers
        self.bezel_pwb_tab_controller = BezelPWBTabController(
            self.ui,
            camera_worker=self.camera_worker,
            camera_interaction=self.camera_interaction,
            snap_worker_left=self.snap_worker_left,  # Left camera worker
            snap_worker_right=self.snap_worker_right,  # Right camera worker
            detection_log_worker=self.detection_log_worker,
            parent=self
        )

        # Just after creation, manually call the camera state:
        if hasattr(self.camera_worker, "last_status") and self.camera_worker.last_status is not None:
            # Suppose camera_worker stores last_status as True/False
            self.token_fpc_tab_controller.update_connection_status(self.camera_worker.last_status)

        # -------------------- Main Window Translation Setup --------------------
        # We'll use the "small_fpc_tab" section from the combined JSON file for main window translations.
        lang_path = os.path.join(os.path.dirname(__file__), "modules", "language.json")
        self.main_lang_manager = SimpleLanguageManager(lang_path)
        print("Calling set_tab('small_fpc_tab')")
        self.main_lang_manager.set_tab("small_fpc_tab")
        print("Calling set_language('en')")
        self.main_lang_manager.set_language("en")
        print("Tab is now:", self.main_lang_manager.current_tab)
        print("Language is now:", self.main_lang_manager.current_language)
        # self.apply_main_translations()

        # Radio buttons for language switching (radioButton_3 for ENG, radioButton_4 for VIE)
        self.ui.radioButton_3.toggled.connect(self.on_main_english_selected)
        self.ui.radioButton_4.toggled.connect(self.on_main_vietnamese_selected)

    # -------------------- Main Window Translation Method --------------------
    def apply_main_translations(self):
        """
        Update main window translatable UI elements using keys from the small_fpc_tab section.
        This method maps every key from the small_fpc_tab section to corresponding UI widgets.
        """
        translatable_widgets = {
            "label_29": self.ui.label_29,
            "label_26": self.ui.label_26,
            "label_27": self.ui.label_27,
            "label_3": self.ui.label_3,
            "label_5": self.ui.label_5,
            "label_6": self.ui.label_6,
            "label": self.ui.label,  # QC Staff ID label
            "label_2": self.ui.label_2,  # Part ID label
            "label_30": self.ui.label_30,  # Pin count label
            "checkBox": self.ui.checkBox,  # Turn on counter checkbox
            "checkBox_2": self.ui.checkBox_2,  # Display processed images checkbox
            "checkBox_3": self.ui.checkBox_3,  # Use histogram checkbox
            "checkBox_4": self.ui.checkBox_4,  # Lock options checkbox
            "label_11": self.ui.label_11,  # Main camera connection status label
            "label_14_connected": self.ui.label_14,  # Connection status (connected) – updated separately
            "label_17": self.ui.label_17,  # Part name label
            "label_12": self.ui.label_12,  # Config file loaded label
            "label_18": self.ui.label_18,  # Config status label
            "pushButton": self.ui.pushButton,  # Open Camera Setting
            "pushButton_2": self.ui.pushButton_2,  # Start Video Streaming
            "label_13": self.ui.label_13,  # Detected part count label
            "label_15": self.ui.label_15,  # Detected NG part count label
            "label_16": self.ui.label_16,  # Detected OK part count label
            "label_91": self.ui.label_91,  # Defect note label
            "label_7": self.ui.label_7,
            "label_9": self.ui.label_9,
            "label_31": self.ui.label_31,
            "label_33": self.ui.label_33,
            "label_23": self.ui.label_23,
            "checkBox_5": self.ui.checkBox_5,
            "label_176": self.ui.label_176,
            "label_64": self.ui.label_64,
            "label_181": self.ui.label_181
        }
        for key, widget in translatable_widgets.items():
            widget.setText(self.main_lang_manager.get_text(key))
        print("Main window translations applied for small_fpc_tab section:", self.main_lang_manager.current_language)

    # -------------------- Main Window Language Switch Handlers --------------------
    def on_main_english_selected(self, checked):
        if checked:
            self.main_lang_manager.set_language("en")
            self.apply_main_translations()
            # Re-apply camera connection status in the new language
            self.update_connection_status(
                self.current_connection_state if hasattr(self, 'current_connection_state') else False
            )
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

    def on_main_vietnamese_selected(self, checked):
        if checked:
            self.main_lang_manager.set_language("vi")
            self.apply_main_translations()
            self.update_connection_status(
                self.current_connection_state if hasattr(self, 'current_connection_state') else False
            )
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

    def start_video_streaming(self):
        if not self.snap_worker.running:
            print("Starting video streaming")
            # Start snap sink worker thread
            self.snap_worker.pre_process()
            self.snap_worker.start()
        else:
            print("Video steaming is starting. Won't restart")

    def update_video_frame(self, frame):
        """
        Slot to receive a frame (as a NumPy array in RGB format) and update label_29.
        Adjusts label_29's geometry to x=180, y=105 with a size of 350x350.
        """
        if frame is None:
            return

        height, width, channels = frame.shape
        bytes_per_line = channels * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale the pixmap to exactly 350x350 with a smooth transformation.
        scaled_pixmap = pixmap.scaled(350, 350, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        # Set the geometry of label_29 to the desired position and size.
        self.ui.label_29.setGeometry(180, 105, 350, 350)
        self.ui.label_29.setScaledContents(True)
        self.ui.label_29.setPixmap(scaled_pixmap)

    def setup_display(self):
        """Set up the display widget after the library is initialized."""
        try:
            # Obtain the display handle from the display widget
            self.display_handle = self.display_widget.as_display()
            # Set the render position so that the image is stretched to fit the widget area
            self.display_handle.set_render_position(ic4.DisplayRenderPosition.STRETCH_CENTER)
        except Exception as e:
            QMessageBox.critical(self, "Display Error", str(e))

    def update_connection_status(self, is_connected):
        """Update the UI based on camera connection status."""
        # Keep track of current connection state for language switching
        self.current_connection_state = is_connected

        if is_connected:
            # Use the translated "label_14_connected" key from small_fpc_tab
            self.ui.label_14.setText(self.main_lang_manager.get_text("label_14_connected"))
            self.ui.label_14.setStyleSheet("color: #25be7c;")  # Green for connected
            print("UI updated: Camera Connected")
        else:
            # Use the translated "label_14_disconnected" key from small_fpc_tab
            self.ui.label_14.setText(self.main_lang_manager.get_text("label_14_disconnected"))
            self.ui.label_14.setStyleSheet("color: #be2b25;")  # Red for disconnected
            print("UI updated: Camera Disconnected")

            # Stop streaming if the camera is disconnected
            if self.snap_worker is not None and self.snap_worker is not None and self.snap_worker.running:
                self.snap_worker.stop()
                print("Snap sink worker stopped")

    def update_staff_id(self):
        """Update the staff ID in the detection log worker."""
        staff_id = self.ui.lineEdit.text()  # Get the staff ID from lineEdit
        if not staff_id.strip():  # Check if the input is empty
            QMessageBox.warning(self, "Input Error", "Staff ID cannot be empty.")
            return  # Exit the method if input is empty

        print(f"Updating staff ID to: {staff_id}")  # Debug print statement
        self.detection_log_worker.set_staff_id(staff_id)  # Update the worker with the new staff ID

        # Show a confirmation dialog
        QMessageBox.information(self, "Staff ID Updated", f"The staff ID has been updated to: {staff_id}")

    def handle_return_pressed(self):
        part_number = self.ui.lineEdit_2.text().strip()
        if part_number:
            self.handle_capture_and_log()

    def handle_text_changed(self):
        part_number = self.ui.lineEdit_2.text().strip()
        if '\n' in part_number:
            self.handle_capture_and_log()

    def update_log_status(self, message):
        """Slot to update the log status in the UI."""
        # print(f"Log Status Update: {message}")  # Debug print statement

    def load_config(self, config_data):
        # Assign config data to image processor
        self.image_processor.config_dict = config_data
        self.image_processor.load_config()

        self.small_fpc_ai_model_runner.image_processor.config_dict = config_data
        self.small_fpc_ai_model_runner.image_processor.load_config()

        """Update the UI with the loaded configuration data."""
        part_name = config_data.get("part-name", "Unknown Part")  # Default to "Unknown Part"
        self.ui.label_20.setText(part_name)  # Update the UI label
        print(f"label_20 updated with part name: {part_name}")  # Debug print statement

    def handle_capture_and_log(self):
        """Handle image capture and logging for both Enter key and text changes."""
        print("Handle capture and log triggered")

        # Reset result
        self.ui.label_24.setStyleSheet("color: #25be7c;")
        self.ui.label_25.setStyleSheet("color: #25be7c;")
        self.ui.label_10.setStyleSheet("color: #25be7c;")

        self.ui.label_24.setText("--")
        self.ui.label_25.setText("--")
        self.ui.label_10.setText("--")
        self.ui.label_32.setText("--")
        self.ui.label_34.setText("--")

        self.display_processed_connector_lock_image(r"C:\BoardDefectChecker\resources\blank.png")
        self.display_processed_jack_fit_image(r"C:\BoardDefectChecker\resources\blank.png")

        part_number = self.ui.lineEdit_2.text().strip()
        staff_id = self.ui.lineEdit.text().strip()

        # Check if the input contains newlines
        if '\n' in part_number:
            part_number = part_number.split('\n')[0].strip()  # Take the first non-empty part
            if not part_number:
                return

        # Skip empty inputs
        if not part_number:
            return

        # Validate inputs
        if not staff_id:
            QMessageBox.warning(self, "Input Error", "Staff ID cannot be empty.")
            return

        # Update staff ID if needed
        if self.detection_log_worker.get_staff_id() != staff_id:
            self.detection_log_worker.set_staff_id(staff_id)
            if self.is_counter_turned_on:
                self.update_detected_result_count()
            print(f"Updated staff ID to: {staff_id}")

        # Capture image
        print(f"Capturing image for part number: {part_number}")
        # raw_image_path = self.camera_interaction.capture_single_image(save_image=True)
        raw_image_path = self.snap_worker.capture_single_image(save_image=True)

        if raw_image_path:
            print("Image captured successfully")
            print(f"Image saved at: {raw_image_path}")

            # Load the image from the file path
            pixmap = QPixmap(raw_image_path)
            if not pixmap.isNull():  # Check if the image loaded successfully
                raw_image = cv2.imread(raw_image_path)

                # Check if image is full of white, black or (52,52,52) color
                if self.image_processor.is_image_full_custom_color(raw_image, (
                        0, 0, 0)) or self.image_processor.is_image_full_custom_color(raw_image, (
                        255, 255, 255)) or self.image_processor.is_image_full_custom_color(raw_image, (52, 52, 52)):
                    print("Image color is full of black/white/other color. Return now")
                    return

                self.ui.label_29.setPixmap(pixmap)
                self.ui.label_29.setScaledContents(True)
                self.ui.label_29.setGeometry(180, 100, 320, 320)  # Set position and fixed size
                self.ui.label_29.show()  # Show the label

                # Initialize model manager
                self.model_manager.scan_model_files()

                # Preprocessing, resize to 640x480
                # resized_raw_image_path, resized_raw_image = self.image_processor.preprocessing(
                #     raw_image_path,
                #     show_image=False,
                #     apply_median_blur=True,
                #     median_kernel_size=7,  # Larger kernel for stronger noise reduction
                #     apply_gaussian_blur=True,
                #     gaussian_kernel_size=3,  # Larger kernel for more smoothing
                #     gaussian_sigma=0.5  # Larger sigma for stronger blur
                # )

                blurred_raw_image, blurred_raw_image_path = self.image_processor.apply_blur_filters(
                    raw_image_path,
                    show_image=False,  # Display intermediate and final images
                    apply_median_blur=True,  # Apply median blur
                    median_kernel_size=7,  # Kernel size for median blur
                    apply_gaussian_blur=True,  # Apply Gaussian blur
                    gaussian_kernel_size=3,  # Kernel size for Gaussian blur
                    gaussian_sigma=0.5,  # Sigma value for Gaussian blur
                )

                # Check if preprocessing was successful
                if blurred_raw_image_path is None:
                    print("Blurring raw image failed.")
                else:
                    # Analyze histogram
                    # self.image_processor.analyze_histogram(resized_image_path, show_plot=False)

                    # Adjust histogram for resized image
                    # adjusted_histogram_image, adjusted_histogram_image_path = self.image_processor.adjust_histogram(
                    #    resized_image_path, show_result=False)
                    # self.image_processor.adjusted_histogram_image = adjusted_histogram_image

                    # Adjust histogram for raw image
                    raw_adjusted_histogram_image, raw_adjusted_histogram_image_path = self.image_processor.adjust_histogram(
                        blurred_raw_image_path, show_result=False)
                    self.image_processor.adjusted_histogram_image = raw_adjusted_histogram_image

                    detection_result = "OK"
                    part_1_detection_result = "OK"
                    part_2_detection_result = "OK"

                    # Start processing captured image using image processor
                    self.image_processor.set_image_path(raw_image_path)

                    # Process using AI models
                    # Get current timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                    # Create output path with timestamp
                    if self.is_histogram_equalized:
                        print("Use histogram equalized image")
                        raw_hist_extracted_connector_image_path, raw_hist_extracted_fpc_image_path, raw_hist_extracted_connector_image, raw_hist_extracted_fpc_image = self.image_processor.extract_raw_parts(
                            raw_adjusted_histogram_image, show_images=False)
                        connector_path_to_use = raw_hist_extracted_connector_image_path
                        connector_image_to_use = raw_hist_extracted_connector_image
                        fpc_path_to_use = raw_hist_extracted_fpc_image_path
                    else:
                        print("No use histogram equalized image")
                        raw_extracted_connector_image_path, raw_extracted_fpc_image_path, raw_extracted_connector_image, raw_extracted_fpc_image = self.image_processor.extract_raw_parts(
                            blurred_raw_image, show_images=False)
                        connector_path_to_use = raw_extracted_connector_image_path
                        connector_image_to_use = raw_extracted_connector_image
                        fpc_path_to_use = raw_extracted_fpc_image_path

                    print(f"FPC path to use: {fpc_path_to_use}")
                    fpc_filename = os.path.basename(fpc_path_to_use)
                    fpc_base_name, ext = os.path.splitext(fpc_filename)
                    fpc_prefix = "fast"
                    fpc_output_path = os.path.join("C:\\BoardDefectChecker\\ai-outputs",
                                                   f"{fpc_prefix}_segmented_{fpc_base_name}_{timestamp}{ext}")

                    print(f"Connector path to use: {connector_path_to_use}")
                    connector_filename = os.path.basename(connector_path_to_use)
                    connector_base_name, ext = os.path.splitext(connector_filename)
                    connector_prefix = "fast"
                    connector_output_path = os.path.join("C:\\BoardDefectChecker\\ai-outputs",
                                                         f"{connector_prefix}_segmented_{connector_base_name}_{timestamp}{ext}")

                    # Process connector image and get outputs
                    connector_result_image, connector_result_mask, connector_output_path, connector_process_time = self.small_fpc_ai_model_runner.process_image(
                        connector_path_to_use, connector_output_path, pin_count=self.target_pin_count,
                        visualize_all_masks=False,
                        input_image_type='connector')

                    # Start evaluating NG/OK for connector part
                    # Without histogram
                    # 2941
                    # 4264
                    # 2949
                    # 3290
                    # 3057
                    # 5779

                    # With histogram - Dark
                    # 3090, 4772
                    # 3166, 4720
                    # 2902, 4778
                    # 2798, 4460
                    # 2717, 4732
                    # 2724, 4488
                    # 2907, 4542
                    # 2655, 4613
                    # 2739, 4448

                    # With histogram - Lighter
                    # 3124, 4704
                    # 3172, 4656
                    # 3144, 4652
                    # 3201, 4654
                    # 3142, 4623

                    # With histogram - Much lighter
                    # 3172, 4629
                    # 3215, 4571
                    # 3199, 4599
                    # 3136, 4597
                    # 3078, 4625

                    # With histogram - Light
                    # 3045, 4630
                    # 3051, 4619
                    # 3053, 4590
                    # 3050, 4579
                    # 3112, 4596

                    min_left_density = 2650
                    min_right_density = 2450
                    max_right_density = 4200

                    if self.target_pin_count == 12:
                        # For 12 pins model
                        max_left_density = 3930
                        # For small 12 pins model
                        # max_left_density = 5000
                        max_right_density = 5500
                    else:
                        # For 10 pins model
                        max_left_density = 4100
                        min_right_density = 4530
                    left_offset = 0
                    right_offset = 0
                    if connector_result_mask is not None:
                        if self.small_fpc_ai_model_runner.check_connector_lock_defect(self.is_processed_images_shown,
                                                                                      connector_image_to_use,
                                                                                      connector_result_mask[0],
                                                                                      self.target_pin_count,
                                                                                      min_left_density,
                                                                                      max_left_density,
                                                                                      min_right_density,
                                                                                      max_right_density,
                                                                                      left_offset, right_offset):

                            self.ui.label_25.setStyleSheet("color: #23b5d8;")
                        else:
                            self.ui.label_25.setStyleSheet("color: #be2b25;")
                            part_1_detection_result = "NG"
                        if self.main_lang_manager.current_language == "en":
                            self.ui.label_32.setText(str(connector_process_time) + " seconds")
                        else:
                            self.ui.label_32.setText(str(connector_process_time) + " giây")
                    else:
                        self.ui.label_25.setStyleSheet("color: #be2b25;")
                        part_1_detection_result = "NG"

                    # Process fpc image and get outputs
                    fpc_result_image, fpc_output_path, fpc_angle_difference, fpc_skeleton_image, fpc_skeleton_output_path, fpc_process_time = self.small_fpc_ai_model_runner.process_image(
                        fpc_path_to_use, fpc_output_path, pin_count=self.target_pin_count, visualize_all_masks=False,
                        input_image_type='fpc')
                    # Check FPC lead balance
                    if self.target_pin_count == 12:
                        # For 12 pins model
                        lower_width_threshold = 610
                        higher_width_threshold = 860
                        lower_height_threshold = 245
                        higher_height_threshold = 298  # Old: 290
                    else:
                        # For 10 pins model
                        lower_width_threshold = 550
                        higher_width_threshold = 700
                        lower_height_threshold = 245
                        higher_height_threshold = 270

                    is_balanced = None
                    balance_output_path = fpc_output_path
                    if fpc_angle_difference is not None:
                        is_balanced, balance_output_path = self.small_fpc_ai_model_runner.check_fpc_lead_balance(
                            fpc_result_image,
                            fpc_output_path,
                            fpc_angle_difference,
                            lower_width_threshold,
                            higher_width_threshold,
                            lower_height_threshold,
                            higher_height_threshold,
                            self.small_fpc_ai_model_runner.current_result_mask)
                    else:
                        part_2_detection_result = "NG"
                        # QMessageBox.warning(self, "AI model error", "AI model error. Try again")
                        # return

                    if connector_result_image is not None:
                        self.display_processed_connector_lock_image(
                            self.small_fpc_ai_model_runner.image_processor.pin_detected_image_path)
                    else:
                        self.display_processed_connector_lock_image(connector_path_to_use)

                    if balance_output_path is not None:
                        self.display_processed_jack_fit_image(balance_output_path)
                    else:
                        # QMessageBox.warning(self, "AI model error", "AI model error. Try again")
                        # return
                        self.display_processed_jack_fit_image(fpc_path_to_use)

                    # Start evaluating NG/OK for FPC Lead/Jack part
                    if is_balanced is not None and is_balanced:
                        self.ui.label_10.setStyleSheet("color: #23b5d8;")
                    else:
                        self.ui.label_10.setStyleSheet("color: #be2b25;")
                        part_2_detection_result = "NG"
                    if self.main_lang_manager.current_language == "en":
                        self.ui.label_34.setText(str(fpc_process_time) + " seconds")
                    else:
                        self.ui.label_34.setText(str(fpc_process_time) + " giây")

                    if part_1_detection_result == "NG" or part_2_detection_result == "NG":
                        detection_result = "NG"

                    # Log the detection result
                    if self.is_counter_turned_on:
                        self.detection_log_worker.log_detection_result(
                            part_serial_number=part_number,
                            detection_result=detection_result,
                            jpeg_image_path=raw_adjusted_histogram_image_path
                        )

                    # Set detection result to UI
                    self.ui.label_25.setText(part_1_detection_result)
                    self.ui.label_10.setText(part_2_detection_result)

                    if detection_result == "OK":
                        self.ui.label_24.setStyleSheet("color: #23b5d8;")
                    else:
                        self.ui.label_24.setStyleSheet("color: #be2b25;")

                    self.ui.label_24.setText(detection_result)

            else:
                QMessageBox.warning(self, "Image Load Error", "Failed to load the image.")
            # QMessageBox.information(
            #    self,
            #    "Detection Result Logged",
            #    f"Image captured and detection result logged for part number '{part_number}'"
            # )

            self.ui.lineEdit_2.clear()

            if self.is_counter_turned_on:
                self.update_detected_result_count()
        else:
            QMessageBox.warning(self, "Capture Error", "Failed to capture image")

    def update_detected_result_count(self):
        if self.is_counter_turned_on:
            # Update detected part count, ng part count, ok part count on the UI
            self.ui.label_19.setText(str(self.detection_log_worker.detected_part_count))
            self.ui.label_21.setText(str(self.detection_log_worker.detected_ng_count))
            self.ui.label_22.setText(str(self.detection_log_worker.detected_ok_count))

    def display_processed_connector_lock_image(self, image_path):
        """Display the processed connector lock image from the provided image path."""
        pixmap = QPixmap(image_path)

        self.ui.label_26.setPixmap(pixmap)
        # self.ui.label_26.setScaledContents(True)
        self.ui.label_26.setGeometry(700, 175, 250, 100)  # Set position and size
        self.ui.label_26.show()  # Show the label

    def display_processed_jack_fit_image(self, image_path):
        """Display the processed jack fit image from the provided image path."""
        pixmap = QPixmap(image_path)

        self.ui.label_27.setPixmap(pixmap)
        self.ui.label_27.setScaledContents(True)
        self.ui.label_27.setGeometry(1000, 175, 250, 100)  # Set position and size
        self.ui.label_27.show()  # Show the label

    def toggle_counter(self, state):
        self.is_counter_turned_on = (state == 2)
        print(f"State: {self.is_counter_turned_on}")
        if not self.is_options_locked:
            print("Allowed")
            if self.is_counter_turned_on:
                print("User turned on counter")
            else:
                print("User turned off counter")
            self.current_checkbox_state = self.is_counter_turned_on
        else:
            print("Denied")
            QMessageBox.information(self, "Message", "Denied")

    def toggle_image_displayed(self, state):
        self.is_processed_images_shown = (state == 2)
        self.image_processor.is_images_shown = (state == 2)
        self.small_fpc_ai_model_runner.is_image_shown = (state == 2)

        if not self.is_options_locked:
            if self.is_processed_images_shown:
                print("User choose to show processed images")
            else:
                print("User choose to not show processed image")
        else:
            print("Denied")
            QMessageBox.information(self, "Message", "Denied")

    def update_target_pin_count(self):
        if not self.is_options_locked:
            self.target_pin_count = int(self.ui.lineEdit_3.text().strip())
            self.image_processor.target_pin_count = self.target_pin_count

            print("Target pin count has been changed to: " + str(self.target_pin_count))
            QMessageBox.information(self, "Message",
                                    f"Target pin count has been changed to: {str(self.target_pin_count)}")
        else:
            print("Denied")
            QMessageBox.information(self, "Message", "Denied")

    def toggle_histogram_equalizer(self, state):
        self.is_histogram_equalized = (state == 2)

        if not self.is_options_locked:
            if self.is_histogram_equalized:
                print("User choose to use histogram equalizer")
            else:
                print("User choose to not use histogram equalizer")
        else:
            print("Denied")
            QMessageBox.information(self, "Message", "Denied")

    def toggle_options_lock(self, state):
        self.is_options_locked = (state == 2)

        if self.is_options_locked:
            print("Options lock is turned on")
            self.ui.checkBox.setEnabled(False)
            self.ui.checkBox_2.setEnabled(False)
            self.ui.checkBox_3.setEnabled(False)
            self.ui.lineEdit_3.setEnabled(False)
            self.ui.pushButton.setEnabled(False)
            self.ui.pushButton_2.setEnabled(False)
            QMessageBox.information(self, "Message", "Options lock is turned on")
            self.ui.checkBox_4.setEnabled(False)
