import os
import threading
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, 
    QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import pyqtSignal, QObject, QThread

from model import train_model

# --- Training Worker Thread ---
class TrainingWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, dataset_path, epochs, lr, batch_size, save_path, model_type, tfm_history_chunks, disc_warmup_steps, lr_decay_rate, lambda_stft, lambda_fm, d_g_ratio):
        super().__init__()
        self.dataset_path = dataset_path
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.save_path = save_path
        self.model_type = model_type
        self.tfm_history_chunks = tfm_history_chunks
        self.disc_warmup_steps = disc_warmup_steps
        self.lr_decay_rate = lr_decay_rate
        self.lambda_stft = lambda_stft
        self.lambda_fm = lambda_fm
        self.d_g_ratio = d_g_ratio
        self._stop_event = threading.Event()

    def run(self):
        """Executes the training function from model.py."""
        train_model(
            dataset_path=self.dataset_path, 
            epochs=self.epochs, 
            learning_rate=self.lr,
            batch_size=self.batch_size, 
            model_save_path=self.save_path,
            progress_callback=self.progress, 
            stop_event=self._stop_event, 
            model_type=self.model_type,
            tfm_history_chunks=self.tfm_history_chunks,
            disc_warmup_steps=self.disc_warmup_steps,
            lr_decay_rate=self.lr_decay_rate,
            lambda_stft=self.lambda_stft,
            lambda_fm=self.lambda_fm,
            d_g_ratio=self.d_g_ratio
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

        # --- Dataset Selection ---
        dataset_layout = QHBoxLayout()
        self.dataset_path_edit = QLineEdit(); self.dataset_path_edit.setPlaceholderText("Path to audio dataset directory...")
        self.browse_button = QPushButton("Browse..."); self.browse_button.clicked.connect(self.browse_dataset)
        dataset_layout.addWidget(QLabel("Dataset Path:")); dataset_layout.addWidget(self.dataset_path_edit); dataset_layout.addWidget(self.browse_button)
        layout.addLayout(dataset_layout)

        # --- Model Selection ---
        model_select_layout = QHBoxLayout()
        model_select_layout.addWidget(QLabel("Model Architecture:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["TS3_Codec (16kbps, Transformer GAN)"]) 
        self.model_type_combo.currentTextChanged.connect(self.update_save_path)
        model_select_layout.addWidget(self.model_type_combo)
        layout.addLayout(model_select_layout)

        # --- Hyperparameters ---
        params_layout = QVBoxLayout()
        params_layout.addWidget(QLabel("<b>Core Hyperparameters:</b>"))
        
        core_h_layout = QHBoxLayout()
        self.epochs_spinbox = QSpinBox(); self.epochs_spinbox.setRange(1, 1000); self.epochs_spinbox.setValue(1000)
        
        # Using QLineEdit for floats is simpler for the user to input specific precision, but QDoubleSpinBox is safer. 
        # Since the user provided LRs as strings, I will keep QLineEdit and ensure parsing is safe.
        self.lr_edit = QLineEdit("0.0001"); 
        self.batch_size_spinbox = QSpinBox(); self.batch_size_spinbox.setRange(1, 32); self.batch_size_spinbox.setValue(8)
        
        core_h_layout.addWidget(QLabel("Epochs:")); core_h_layout.addWidget(self.epochs_spinbox)
        core_h_layout.addWidget(QLabel("Learning Rate:")); core_h_layout.addWidget(self.lr_edit)
        core_h_layout.addWidget(QLabel("Batch Size:")); core_h_layout.addWidget(self.batch_size_spinbox)
        params_layout.addLayout(core_h_layout)

        params_layout.addWidget(QLabel("<b>GAN & Loss Controls:</b>"))
        gan_h_layout = QHBoxLayout()
        # Increased STFT weight for better reconstruction quality (PESQ/STOI)
        self.lambda_stft_edit = QLineEdit("60.0"); 
        self.lambda_fm_edit = QLineEdit("4.0") 
        self.disc_warmup_spinbox = QSpinBox(); self.disc_warmup_spinbox.setRange(1, 1000); self.disc_warmup_spinbox.setValue(100)
        self.d_g_ratio_spinbox = QSpinBox(); self.d_g_ratio_spinbox.setRange(1, 10); self.d_g_ratio_spinbox.setValue(2)

        gan_h_layout.addWidget(QLabel("STFT ($\lambda$):")); gan_h_layout.addWidget(self.lambda_stft_edit)
        gan_h_layout.addWidget(QLabel("FM ($\lambda$):")); gan_h_layout.addWidget(self.lambda_fm_edit)
        gan_h_layout.addWidget(QLabel("Warmup Steps:")); gan_h_layout.addWidget(self.disc_warmup_spinbox)
        gan_h_layout.addWidget(QLabel("D/G Ratio:")); gan_h_layout.addWidget(self.d_g_ratio_spinbox)
        params_layout.addLayout(gan_h_layout)

        params_layout.addWidget(QLabel("<b>Latency & Schedule:</b>"))
        lat_sch_layout = QHBoxLayout()
        self.tfm_history_spinbox = QSpinBox(); self.tfm_history_spinbox.setRange(0, 5); self.tfm_history_spinbox.setValue(0)
        self.lr_decay_edit = QLineEdit("0.9999")

        lat_sch_layout.addWidget(QLabel("TFM Hist. Chunks:")); lat_sch_layout.addWidget(self.tfm_history_spinbox)
        lat_sch_layout.addWidget(QLabel("LR Decay:")); lat_sch_layout.addWidget(self.lr_decay_edit)
        params_layout.addLayout(lat_sch_layout)

        layout.addLayout(params_layout)

        # --- Save Path ---
        save_layout = QHBoxLayout()
        self.model_save_path_edit = QLineEdit(); self.model_save_path_edit.setPlaceholderText("Path to save model...")
        save_layout.addWidget(QLabel("Save Model As:")); save_layout.addWidget(self.model_save_path_edit)
        layout.addLayout(save_layout)
        
        # --- Controls ---
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Training (TS3 GACodec)"); self.start_button.setStyleSheet("background-color: #4CAF50; color: white;"); self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stop Training"); self.stop_button.setStyleSheet("background-color: #f44336; color: white;"); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop_training)
        button_layout.addWidget(self.start_button); button_layout.addWidget(self.stop_button)
        layout.addLayout(button_layout)

        # --- Log Area ---
        layout.addWidget(QLabel("Training Log:"))
        self.log_text_edit = QTextEdit(); self.log_text_edit.setReadOnly(True)
        layout.addWidget(self.log_text_edit)

        self.update_save_path(self.model_type_combo.currentText())

    def update_save_path(self, model_name):
        self.model_save_path_edit.setText(f"low_latency_codec_ts3_gacodec.pth")

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
            # Core
            epochs = self.epochs_spinbox.value(); lr = float(self.lr_edit.text()); batch_size = self.batch_size_spinbox.value()
            model_type = "transformer"
            # GAN
            lambda_stft = float(self.lambda_stft_edit.text()); lambda_fm = float(self.lambda_fm_edit.text())
            disc_warmup_steps = self.disc_warmup_spinbox.value(); d_g_ratio = self.d_g_ratio_spinbox.value()
            # Schedule/Latency
            lr_decay_rate = float(self.lr_decay_edit.text()); tfm_history_chunks = self.tfm_history_spinbox.value()
        except ValueError as e:
            self.update_log(f"ERROR: Hyperparameters must be valid numbers: {e}"); return

        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.log_text_edit.clear(); self.update_log(f"Starting TS3 GACodec training...")

        self.training_thread = QThread()
        self.worker = TrainingWorker(
            dataset_path, epochs, lr, batch_size, save_path, model_type,
            tfm_history_chunks, disc_warmup_steps, lr_decay_rate, lambda_stft, lambda_fm, d_g_ratio
        )
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
