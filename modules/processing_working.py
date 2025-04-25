import os
import cv2
from datetime import datetime
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from modules.bezel_pwb_position_image_processing import BezelPWBPositionImageProcessor


class ProcessingWorker(QThread):
    processing_finished = Signal(object, str, float, str, str, str)

    def __init__(self, token_fpc_ai_model_runner, input_path, detection_log_worker, part_number, staff_id,
                 is_counter_turned_on):
        super().__init__()
        self.token_fpc_ai_model_runner = token_fpc_ai_model_runner
        self.input_path = input_path
        self.detection_log_worker = detection_log_worker
        self.part_number = part_number
        self.staff_id = staff_id
        self.is_counter_turned_on = is_counter_turned_on

    def run(self):
        visualized_image, bb_path, processing_time, detection_result, defect_reason = self.token_fpc_ai_model_runner.process_image(
            input_path=self.input_path, visualize_all_masks=False, label_color=(255, 0, 0)
        )
        self.processing_finished.emit(visualized_image, bb_path, processing_time, detection_result, defect_reason,
                                      self.input_path)
        if self.is_counter_turned_on and self.detection_log_worker:
            self.detection_log_worker.log_token_fpc_detection_result(
                part_serial_number=self.part_number,
                detection_result=detection_result,
                bmp_image_path=self.input_path
            )


class BezelPWBProcessingWorker(QThread):
    processing_finished = Signal(object, object, object, object, str, str, float, float, str, str)

    def __init__(self, bezel_pwb_ai_model, unused_model, left_input_path, right_input_path,
                 detection_log_worker, part_number, staff_id, is_counter_turned_on, is_show_images=False, is_pwb_check_enabled=True):
        super().__init__()
        self.bezel_pwb_ai_model = bezel_pwb_ai_model
        self.left_input_path = left_input_path
        self.right_input_path = right_input_path
        self.detection_log_worker = detection_log_worker
        self.part_number = part_number
        self.staff_id = staff_id
        self.is_counter_turned_on = is_counter_turned_on
        self.is_show_images = is_show_images
        self.processor = BezelPWBPositionImageProcessor()
        self.processor.is_images_shown = self.is_show_images
        self.is_pwb_check_enabled = is_pwb_check_enabled

    def run(self):
        # MODIFIED: Removed the visualize_all_masks keyword argument.
        # The processor instance (self.processor) already has its
        # is_images_shown attribute set correctly in the worker's __init__,
        # and the processor's process_image method uses that internal attribute.
        result = self.processor.process_image(
            self.bezel_pwb_ai_model,
            None,  # unused_model placeholder
            self.left_input_path,
            self.right_input_path,
            self.is_pwb_check_enabled
            # No explicit keyword args needed here anymore,
            # as the processor handles passing relevant params down via **kwargs
            # and uses its own self.is_images_shown state.
        )

        # Check if the result is valid (should be a tuple of 10 elements)
        if result is None or not isinstance(result, tuple) or len(result) != 10:
            # Emit None for all image objects and NG status if processing failed
            self.processing_finished.emit(None, None, None, None, "NG", "NG", 0.0, 0.0, "NG",
                                          "Processing failed or returned invalid data")
            print("[Worker] Result is None or length/type is incorrect. Emitting failure signal.")
            return

        # Unpack the 10 results correctly
        (left_mask_img, right_mask_img, left_annotated_pwb_img, right_annotated_pwb_img,
         left_result, right_result, left_time, right_time, final_result, defect_reason) = result

        # Emit the results via the signal
        self.processing_finished.emit(
            left_mask_img, right_mask_img, left_annotated_pwb_img, right_annotated_pwb_img,
            left_result, right_result, left_time, right_time, final_result, defect_reason
        )

        # Log the result if the counter is enabled
        if self.is_counter_turned_on and self.detection_log_worker:
            print(f"[Worker] Logging result - Final: {final_result}, Left: {left_result}, Right: {right_result}")
            self.detection_log_worker.log_bezel_pwb_detection_result(
                part_serial_number=self.part_number,
                left_result=left_result,
                right_result=right_result,
                left_image_path=self.left_input_path,
                right_image_path=self.right_input_path
            )
        elif not self.is_counter_turned_on:
            print("[Worker] Counter is off. Skipping logging.")
        elif self.detection_log_worker is None:
            print("[Worker] Detection log worker is None. Skipping logging.")


class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setModal(True)
        self.setFixedSize(300, 100)
        layout = QVBoxLayout(self)
        label = QLabel("Processing...", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        progress_bar = QProgressBar(self)
        progress_bar.setRange(0, 0)
        layout.addWidget(progress_bar)
        self.setLayout(layout)
