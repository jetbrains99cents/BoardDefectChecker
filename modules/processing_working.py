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
                 detection_log_worker, part_number, staff_id, is_counter_turned_on, is_show_images=False):
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

    def run(self):
        result = self.processor.process_image(
            self.bezel_pwb_ai_model, None, self.left_input_path, self.right_input_path,
            visualize_all_masks=self.is_show_images
        )
        if result is None or len(result) != 10:
            self.processing_finished.emit(None, None, None, None, "NG", "NG", 0.0, 0.0, "NG", "Processing failed")
            print("Result is None or length of result is not equal to 10. Processing failed")
            return

        (left_bezel_vis, right_bezel_vis, left_pwb_vis, right_pwb_vis,
         left_result, right_result, left_time, right_time, final_result, defect_reason) = result

        self.processing_finished.emit(
            left_bezel_vis, right_bezel_vis, left_pwb_vis, right_pwb_vis,
            left_result, right_result, left_time, right_time, final_result, defect_reason
        )

        if self.is_counter_turned_on and self.detection_log_worker:
            self.detection_log_worker.log_bezel_pwb_detection_result(
                part_serial_number=self.part_number,
                left_result=left_result,
                right_result=right_result,
                left_image_path=self.left_input_path,
                right_image_path=self.right_input_path
            )


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
