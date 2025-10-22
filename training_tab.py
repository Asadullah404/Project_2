import os
import threading
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox
from PyQt5.QtCore import pyqtSignal, QObject, QThread

from model import train_model

# --- Training Worker Thread ---
class TrainingWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, dataset_path, epochs, lr, batch_size, save_path, model_type):
        super().__init__()
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path
        self.model_type = model_type
        self._stop_event = threading.Event()

    def run(self):
        """Executes the training function from model.py."""
        train_model(
            dataset_path=self.dataset_path, epochs=self.epochs, learning_rate=self.lr,
            batch_size=self.batch_size, model_save_path=self.save_path,
            progress_callback=self.progress, stop_event=self._stop_event, model_type=self.model_type
        )
        self.finished.emit()

    def stop(self):
        """Signals the training loop to stop."""
        self._stop_event.set()

class TrainingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.training_thread = None
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Dataset Selection
        dataset_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit(); self.dataset_path_edit.setPlaceholderText("Path to audio dataset directory...")
        self.browse_button = QPushButton("Browse..."); self.browse_button.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(QLabel("Dataset Path:")); dataset_layout.addWidget(self.dataset_path_edit); dataset_layout.addWidget(self.browse_button)
        layout.addLayout(dataset_layout)

        # Model Selection
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model Architecture:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "GRU_Codec (16kbps, Fast)", 
            "TS3_Codec (16kbps, Transformer)",
            "ScoreDec Post-Filter (on GRU Codec)"
        ])
        self.model_type_combo.currentTextChanged.connect(self.update_save_path) # Connect signal
        model_select_layout.addWidget(self.model_type_combo)
        layout.addLayout(model_select_layout)

        # Hyperparameters
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Epochs:")); self.epochs_spinbox = QSpinBox(); self.epochs_spinbox.setRange(1, 1000); self.epochs_spinbox.setValue(50)
        params_layout.addWidget(self.epochs_spinbox)
        params_layout.addWidget(QLabel("Learning Rate:")); self.lr_edit = QLineEdit("0.0002"); params_layout.addWidget(self.lr_edit)
        params_layout.addWidget(QLabel("Batch Size:")); self.batch_size_spinbox = QSpinBox(); self.batch_size_spinbox.setRange(1, 128); self.batch_size_spinbox.setValue(16)
        params_layout.addWidget(self.batch_size_spinbox)
        layout.addLayout(params_layout)

        # Save Path
        save_layout = QHBoxLayout()
        self.model_save_path_edit = QLineEdit(); self.model_save_path_edit.setPlaceholderText("Path to save model...")
        save_layout.addWidget(QLabel("Save Model As:")); save_layout.addWidget(self.model_save_path_edit)
        layout.addLayout(save_layout)
        
        # Controls
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training"); self.start_button.setStyleSheet("background-color: #4CAF50; color: white;"); self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training"); self.stop_button.setStyleSheet("background-color: #f44336; color: white;"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.start_button); button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # Log Area
        layout.addWidget(QLabel("Training Log:"))
        self.log_text_edit = QTextEdit(); self.log_text_edit.setReadOnly(True)
        layout.addWidget(self.log_text_edit)

        # Set initial save path
        self.update_save_path(self.model_type_combo.currentText())

    def update_save_path(self, model_name):
        """Automatically updates the model save path based on selection."""
        if "GRU_Codec" in model_name:
            sanitized_name = "gru"
        elif "TS3_Codec" in model_name:
            sanitized_name = "ts3_transformer"
        elif "ScoreDec" in model_name:
            sanitized_name = "scoredec_post_filter"
        self.model_save_path_edit.setText(f"low_latency_codec_{sanitized_name}.pth")

    def browse_dataset(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if directory: self.dataset_path_edit.setText(directory)
            
    def update_log(self, message):
        self.log_text_edit.append(message)

    def start_training(self):
        dataset_path = self.dataset_path_edit.text()
        save_path = self.model_save_path_edit.text()
        
        if not os.path.isdir(dataset_path): self.update_log("ERROR: Please provide a valid dataset directory."); return
        if not save_path: self.update_log("ERROR: Please provide a path to save the model."); return

        try:
            epochs = self.epochs_spinbox.value()
            lr = float(self.lr_edit.text())
            batch_size = self.batch_size_spinbox.value()
        except ValueError:
            self.update_log("ERROR: Hyperparameters must be valid numbers."); return

        model_type_map = {
            "GRU_Codec (16kbps, Fast)": "gru", 
            "TS3_Codec (16kbps, Transformer)": "transformer",
            "ScoreDec Post-Filter (on GRU Codec)": "scoredec"
        }
        model_type = model_type_map.get(self.model_type_combo.currentText())

        # Check for ScoreDec dependency
        if model_type == 'scoredec' and not os.path.exists('low_latency_codec_gru.pth'):
            self.update_log("--------------------------------------------------")
            self.update_log("ERROR: Dependency not found!")
            self.update_log("Training 'ScoreDec' requires a pre-trained 'low_latency_codec_gru.pth' file.")
            self.update_log("Please train and save the 'GRU_Codec' model first.")
            self.update_log("--------------------------------------------------")
            return

        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.log_text_edit.clear(); self.update_log(f"Starting training for {model_type} model...")

        self.training_thread = QThread()
        self.worker = TrainingWorker(dataset_path, epochs, lr, batch_size, save_path, model_type)
        self.worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.training_finished)
        self.worker.progress.connect(self.update_log)
        self.training_thread.start()

    def stop_training(self):
        if self.worker:
            self.update_log("Sending stop signal to training process...")
            self.worker.stop()
            self.stop_button.setEnabled(False)
            
    def training_finished(self):
        self.update_log("Training process has finished.")
        if self.training_thread:
            self.training_thread.quit()
            self.training_thread.wait()
        self.worker = None; self.training_thread = None
        self.start_button.setEnabled(True); self.stop_button.setEnabled(False)

