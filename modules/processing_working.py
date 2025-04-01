from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar


class ProcessingWorker(QThread):
    processing_finished = Signal(object, str, float, str, str, str)  # Adjusted to float for processing_time

    def __init__(self, token_fpc_ai_model_runner, input_path, detection_log_worker, part_number, staff_id, is_counter_turned_on):
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
        self.processing_finished.emit(visualized_image, bb_path, processing_time, detection_result, defect_reason, self.input_path)

        if self.is_counter_turned_on and self.detection_log_worker:
            self.detection_log_worker.log_token_fpc_detection_result(
                part_serial_number=self.part_number,
                detection_result=detection_result,
                bmp_image_path=self.input_path
            )


class ProcessingDialog(QDialog):
    """Dialog to show a loading animation during processing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing")
        self.setModal(True)  # Make it block interaction with the main window
        self.setFixedSize(300, 100)

        # Layout
        layout = QVBoxLayout(self)

        # "Processing" label
        label = QLabel("Processing...", self)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)

        # Indeterminate progress bar for animation
        progress_bar = QProgressBar(self)
        progress_bar.setRange(0, 0)  # Indeterminate mode (continuous animation)
        layout.addWidget(progress_bar)

        self.setLayout(layout)