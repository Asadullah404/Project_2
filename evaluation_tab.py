# import os # <-- ADDED IMPORT
# import numpy as np
# import librosa
# import soundfile as sf
# import torch
# from pesq import pesq
# from pystoi import stoi
# import pyaudio
# import threading
# import time

# from PyQt5.QtWidgets import (
#     QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
#     QLabel, QFileDialog, QComboBox
# )
# from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex

# import matplotlib
# matplotlib.use('Qt5Agg')
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# from model import (
#     MuLawCodec, ALawCodec, DACCodec, EnCodecModel, TinyTransformerCodec, 
#     HOP_SIZE, DAC_AVAILABLE, ENCODEC_AVAILABLE
# )

# # --- Matplotlib Canvas Widget ---
# class MplCanvas(FigureCanvas):
#     def __init__(self, parent=None, width=5, height=4, dpi=100):
#         self.fig = Figure(figsize=(width, height), dpi=dpi)
#         self.axes = self.fig.add_subplot(111)
#         super(MplCanvas, self).__init__(self.fig)

# # --- Evaluation Worker Thread ---
# class EvaluationWorker(QObject):
#     finished = pyqtSignal(dict)
    
#     def __init__(self, model, original_file_path, model_type_str):
#         super().__init__()
#         self.model = model
#         self.original_file_path = original_file_path
#         self.model_type_str = model_type_str
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     def run(self):
#         results = {}
#         try:
#             sr = 16000
#             original_wav, _ = librosa.load(self.original_file_path, sr=sr, mono=True)
#             original_wav = original_wav.astype(np.float32)

#             start_time = time.time()
            
#             # --- CODEC PROCESSING LOGIC ---
#             reconstructed_wav = np.copy(original_wav)
            
#             if self.model:
#                 audio_tensor = torch.from_numpy(original_wav).unsqueeze(0).to(self.device, dtype=torch.float32)
                
#                 # Check for all neural codecs
#                 if isinstance(self.model, (DACCodec, EnCodecModel, TinyTransformerCodec)): 
#                     # Neural codec processing
#                     if audio_tensor.dim() == 2:
#                         audio_tensor = audio_tensor.unsqueeze(1)
                    
#                     codes, orig_len = self.model.encode(audio_tensor)
#                     reconstructed_tensor = self.model.decode(codes, orig_len)
                    
#                     reconstructed_wav = reconstructed_tensor.squeeze().cpu().numpy()
                    
#                 else:
#                     # Traditional codecs
#                     audio_tensor = torch.from_numpy(original_wav).unsqueeze(0).unsqueeze(0)
#                     encoded = self.model.encode(audio_tensor)
#                     reconstructed_tensor = self.model.decode(encoded)
#                     reconstructed_wav = reconstructed_tensor.squeeze().cpu().numpy()

#             end_time = time.time()
#             processing_time = end_time - start_time
#             audio_duration = len(original_wav) / sr
#             real_time_factor = processing_time / audio_duration

#             # Ensure lengths match for metrics
#             min_len = min(len(original_wav), len(reconstructed_wav))
#             original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
#             original_wav = original_wav.astype(np.float32)
#             reconstructed_wav = reconstructed_wav.astype(np.float32)

#             pesq_score = pesq(sr, original_wav, reconstructed_wav, 'wb')
#             stoi_score = stoi(original_wav, reconstructed_wav, sr, extended=False)

#             results = {
#                 'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
#                 'sr': sr, 'pesq': pesq_score, 'stoi': stoi_score, 
#                 'rtf': real_time_factor, 'error': None
#             }
#         except Exception as e:
#             results['error'] = str(e)
#             import traceback
#             print(f"Error in evaluation: {e}")
#             print(traceback.format_exc())

#         self.finished.emit(results)

# class EvaluationTab(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.model = None
#         self.original_wav = None
#         self.reconstructed_wav = None
        
#         # Audio Playback Control
#         self.audio_thread = None
#         self.stop_audio_event = threading.Event()
#         self.audio_mutex = QMutex()
        
#         self._setup_ui()

#     def _setup_ui(self):
#         layout = QVBoxLayout(self)

#         # File and Model Selection
#         file_layout = QHBoxLayout()
#         self.audio_file_edit = QLineEdit()
#         self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
#         self.browse_audio_button = QPushButton("Browse Audio...")
#         self.browse_audio_button.clicked.connect(self.browse_audio)
#         file_layout.addWidget(QLabel("Audio File:"))
#         file_layout.addWidget(self.audio_file_edit)
#         file_layout.addWidget(self.browse_audio_button)
#         layout.addLayout(file_layout)

#         model_layout = QHBoxLayout()
#         self.model_path_edit = QLineEdit()
#         self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth). Mandatory for Tiny Transformer Codec.") 
#         self.browse_model_button = QPushButton("Browse Model...")
#         self.browse_model_button.clicked.connect(self.browse_model)
#         self.model_type_combo = QComboBox()
        
#         # Add all available models
#         model_items = [
#             "Uncompressed",
#             "μ-Law Codec (Baseline)", 
#             "A-Law Codec (Baseline)",
#             "Tiny Transformer Codec (Custom, 16kHz, ~10ms)", # Your custom codec
#         ]
#         if DAC_AVAILABLE:
#             model_items.append("DAC Codec (16kHz, 20ms)")
#         if ENCODEC_AVAILABLE:
#             model_items.append("EnCodec (24kHz, 6kbps)")
        
#         self.model_type_combo.addItems(model_items)
#         self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
#         model_layout.addWidget(QLabel("Codec Model:"))
#         model_layout.addWidget(self.model_type_combo)
#         model_layout.addWidget(self.model_path_edit)
#         model_layout.addWidget(self.browse_model_button)
#         layout.addLayout(model_layout)
        
#         # Controls and Results
#         self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
#         self.run_eval_button.clicked.connect(self.run_evaluation)
#         layout.addWidget(self.run_eval_button)

#         results_layout = QHBoxLayout()
#         self.pesq_label = QLabel("PESQ: --")
#         self.stoi_label = QLabel("STOI: --")
#         self.rtf_label = QLabel("Real-Time Factor: --")
#         self.bitrate_label = QLabel(f"Bitrate: N/A")
#         results_layout.addWidget(self.pesq_label)
#         results_layout.addWidget(self.stoi_label)
#         results_layout.addWidget(self.rtf_label)
#         results_layout.addWidget(self.bitrate_label)
#         layout.addLayout(results_layout)
        
#         # Playback Controls
#         playback_layout = QHBoxLayout()
#         self.play_original_button = QPushButton("▶ Play Original")
#         self.play_original_button.setEnabled(False)
#         self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
#         self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
#         self.play_reconstructed_button.setEnabled(False)
#         self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

#         self.stop_playback_button = QPushButton("■ Stop Playback")
#         self.stop_playback_button.setEnabled(False)
#         self.stop_playback_button.clicked.connect(self.stop_audio)
        
#         playback_layout.addWidget(self.play_original_button) 
#         playback_layout.addWidget(self.play_reconstructed_button)
#         playback_layout.addWidget(self.stop_playback_button)
#         layout.addLayout(playback_layout)

#         self.status_label = QLabel("Status: Ready")
#         layout.addWidget(self.status_label)

#         # Plots
#         plot_layout = QHBoxLayout()
#         self.canvas_original = MplCanvas(self)
#         self.canvas_reconstructed = MplCanvas(self)
#         plot_layout.addWidget(self.canvas_original)
#         plot_layout.addWidget(self.canvas_reconstructed)
#         layout.addLayout(plot_layout)

#         self.on_model_type_changed(self.model_type_combo.currentText())

#     def on_model_type_changed(self, model_name):
#         is_neural = any(x in model_name for x in ["DAC", "EnCodec", "Transformer"]) 
#         is_custom_transformer = "Tiny Transformer Codec" in model_name
        
#         self.model_path_edit.setEnabled(is_neural)
#         self.browse_model_button.setEnabled(is_neural)
        
#         if not is_neural:
#             self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
#             bitrate = '256 kbps' if 'Uncompressed' in model_name else '128 kbps'
#             self.bitrate_label.setText(f"Bitrate: {bitrate}")
#         elif "DAC" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms")
#         elif "EnCodec" in model_name:
#             self.model_path_edit.setText("(auto-download if not provided)")
#             self.bitrate_label.setText(f"Bitrate: 6 kbps | Latency: ~10ms")
#         elif is_custom_transformer:
#              self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pth)")
#              self.bitrate_label.setText(f"Bitrate: ~8 kbps (Estimated) | Latency: ~10ms")

#     def browse_audio(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
#         if filepath:
#             self.audio_file_edit.setText(filepath)

#     def browse_model(self):
#         filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
#         if filepath:
#             self.model_path_edit.setText(filepath)
            
#     def load_model(self):
#         model_type_str = self.model_type_combo.currentText()
#         self.model = None

#         if "Uncompressed" in model_type_str:
#             self.status_label.setText("Status: Using Uncompressed Passthrough.")
#             return True
#         elif "μ-Law" in model_type_str: 
#             self.model = MuLawCodec()
#             self.status_label.setText("Status: Loaded μ-Law Codec.")
#             return True
#         elif "A-Law" in model_type_str: 
#             self.model = ALawCodec()
#             self.status_label.setText("Status: Loaded A-Law Codec.")
#             return True
#         elif "DAC" in model_type_str:
#             if not DAC_AVAILABLE:
#                 self.status_label.setText("Status: ERROR - DAC not installed.")
#                 return False
#             try:
#                 path = self.model_path_edit.text()
#                 if "auto-download" in path or not path or "N/A" in path:
#                     path = None
#                 self.model = DACCodec(model_path=path, model_type="16khz")
#                 self.status_label.setText("Status: Loaded DAC Codec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
#                 return False
#         elif "EnCodec" in model_type_str:
#             if not ENCODEC_AVAILABLE:
#                 self.status_label.setText("Status: ERROR - EnCodec not installed.")
#                 return False
#             try:
#                 path = self.model_path_edit.text()
#                 if "auto-download" in path or not path or "N/A" in path:
#                     path = None
#                 self.model = EnCodecModel(model_path=path)
#                 self.status_label.setText("Status: Loaded EnCodec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load EnCodec: {e}")
#                 return False
#         elif "Tiny Transformer Codec" in model_type_str: # Custom Codec Loading
#             model_path = self.model_path_edit.text()
#             # FIX: os.path.exists requires 'os' module to be imported
#             if "MANDATORY" in model_path or not os.path.exists(model_path): 
#                 self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pth) model.")
#                 return False
#             try:
#                 self.model = TinyTransformerCodec.load_model(model_path)
#                 self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
#                 return True
#             except Exception as e:
#                 self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
#                 return False
            
#         return False

#     def run_evaluation(self):
#         # Stop any audio before running evaluation
#         self.stop_audio()
        
#         if not self.audio_file_edit.text():
#             self.status_label.setText("Status: Please select an audio file.")
#             return
#         if not self.load_model():
#             return

#         self.status_label.setText("Status: Evaluating... Please wait.")
#         self.run_eval_button.setEnabled(False)
#         self.play_original_button.setEnabled(False)
#         self.play_reconstructed_button.setEnabled(False)

#         self.eval_thread = QThread()
#         self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
#         self.eval_worker.moveToThread(self.eval_thread)
#         self.eval_worker.finished.connect(self.on_evaluation_complete)
#         self.eval_thread.started.connect(self.eval_worker.run)
#         self.eval_thread.start()

#     def on_evaluation_complete(self, results):
#         if results.get('error'):
#             self.status_label.setText(f"Status: ERROR - {results['error']}")
#         else:
#             pesq_score = results['pesq']
#             stoi_score = results['stoi']
#             rtf_score = results['rtf']
            
#             # Highlight scores based on targets
#             pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
#             stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
#             rtf_color = 'green' if rtf_score < 1.0 else 'red'
            
#             self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
#             self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
#             self.rtf_label.setText(f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>")
            
#             self.original_wav = results['original_wav']
#             self.reconstructed_wav = results['reconstructed_wav']
#             self.play_original_button.setEnabled(True)
#             self.play_reconstructed_button.setEnabled(True)
#             self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
#             self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
#             if rtf_score > 1.0:
#                 self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
#             else:
#                 self.status_label.setText("Status: Evaluation complete. (RTF < 1.0, real-time capable)")
            
#         self.run_eval_button.setEnabled(True)
#         if self.eval_thread:
#             self.eval_thread.quit()
#             self.eval_thread.wait()
    
#     def stop_audio(self):
#         """Stops the currently playing audio thread."""
#         if self.audio_thread and self.audio_thread.is_alive():
#             self.stop_audio_event.set()
#             self.audio_thread.join(timeout=0.1)
#             self.audio_thread = None
#             self.status_label.setText("Status: Playback stopped.")
        
#         self.play_original_button.setText("▶ Play Original")
#         self.play_reconstructed_button.setText("▶ Play Reconstructed")
#         self.stop_playback_button.setEnabled(False)

#     def play_audio(self, wav_data, button_clicked):
#         if wav_data is None:
#             self.status_label.setText("Status: No audio data to play.")
#             return
        
#         self.stop_audio()
        
#         self.stop_playback_button.setEnabled(True)
#         button_clicked.setText("Playing...")

#         self.stop_audio_event.clear()
#         self.status_label.setText("Status: Playing audio...")
#         self.audio_thread = threading.Thread(
#             target=self._play_audio_thread, 
#             args=(wav_data, 16000, button_clicked), 
#             daemon=True
#         )
#         self.audio_thread.start()

#     def _play_audio_thread(self, wav_data, sr, button):
#         p = None
#         stream = None
#         try:
#             p = pyaudio.PyAudio()
#             stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
#             chunk_size = 1024
#             data_to_play = wav_data.astype(np.float32)
            
#             for i in range(0, len(data_to_play), chunk_size):
#                 if self.stop_audio_event.is_set():
#                     break
                
#                 chunk = data_to_play[i:i + chunk_size].tobytes()
#                 stream.write(chunk)
                
#             self.audio_mutex.lock()
#             if not self.stop_audio_event.is_set():
#                 self.status_label.setText("Status: Playback finished.")
#             self.audio_mutex.unlock()
            
#         except Exception as e:
#             print(f"Error playing audio: {e}")
#             self.status_label.setText(f"Status: Playback error: {e}")
#         finally:
#             if stream:
#                 stream.stop_stream()
#                 stream.close()
#             if p:
#                 p.terminate()
            
#             button.setText(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}")
#             if not self.stop_audio_event.is_set():
#                 self.stop_playback_button.setEnabled(False)

#     def plot_spectrogram(self, canvas, wav, sr, title):
#         try:
#             canvas.axes.cla()
#             import librosa.display
#             S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
#             librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
#             canvas.axes.set_title(title)
#             canvas.fig.tight_layout()
#             canvas.draw()
#         except Exception as e:
#             print(f"Error plotting spectrogram: {e}")

#     def closeEvent(self, event):
#         self.stop_audio()
#         super().closeEvent(event)


import os 
import numpy as np
import librosa
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
import pyaudio
import threading
import time

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QFileDialog, QComboBox
)
from PyQt5.QtCore import QObject, pyqtSignal, QThread, QMutex

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from model import (
    MuLawCodec, ALawCodec, DACCodec, EnCodecModel, TinyTransformerCodec, 
    HOP_SIZE, DAC_AVAILABLE, ENCODEC_AVAILABLE
)

# --- Matplotlib Canvas Widget ---
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

# --- Evaluation Worker Thread ---
class EvaluationWorker(QObject):
    finished = pyqtSignal(dict)
    
    def __init__(self, model, original_file_path, model_type_str):
        super().__init__()
        self.model = model
        self.original_file_path = original_file_path
        self.model_type_str = model_type_str
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        results = {}
        try:
            sr = 16000
            original_wav, _ = librosa.load(self.original_file_path, sr=sr, mono=True)
            original_wav = original_wav.astype(np.float32)

            start_time = time.time()
            
            # --- CODEC PROCESSING LOGIC ---
            reconstructed_wav = np.copy(original_wav)
            
            # We explicitly use torch.no_grad() here for safety, though model.eval() is often sufficient, 
            # detaching the tensor before numpy conversion is the key fix.
            with torch.no_grad():
                if self.model:
                    audio_tensor = torch.from_numpy(original_wav).unsqueeze(0).to(self.device, dtype=torch.float32)
                    
                    # Check for all neural codecs
                    if isinstance(self.model, (DACCodec, EnCodecModel)): 
                        # DAC/EnCodec processing
                        if audio_tensor.dim() == 2:
                            audio_tensor = audio_tensor.unsqueeze(1)
                        
                        codes, orig_len = self.model.encode(audio_tensor)
                        reconstructed_tensor = self.model.decode(codes, orig_len)
                        
                        # FIX: Add .detach()
                        reconstructed_wav = reconstructed_tensor.squeeze().detach().cpu().numpy()
                        
                    elif isinstance(self.model, TinyTransformerCodec):
                        # --- NEW VQ-CODEC PROCESSING ---
                        # 1. Encode: Get the quantized float codes and all necessary artifacts
                        codes, indices_list, orig_len, encoder_outputs = self.model.encode(audio_tensor)
                        
                        # 2. Decode using the full reconstruction path (quantized float codes + skip features)
                        reconstructed_tensor = self.model.decode(codes, orig_len, encoder_outputs)
                        
                        # FIX: Add .detach()
                        reconstructed_wav = reconstructed_tensor.squeeze().detach().cpu().numpy()
                        
                    else:
                        # Traditional codecs
                        audio_tensor = torch.from_numpy(original_wav).unsqueeze(0).unsqueeze(0)
                        encoded = self.model.encode(audio_tensor)
                        reconstructed_tensor = self.model.decode(encoded)
                        # FIX: Add .detach()
                        reconstructed_wav = reconstructed_tensor.squeeze().detach().cpu().numpy()

            end_time = time.time()
            processing_time = end_time - start_time
            audio_duration = len(original_wav) / sr
            real_time_factor = processing_time / audio_duration

            # Ensure lengths match for metrics
            min_len = min(len(original_wav), len(reconstructed_wav))
            original_wav, reconstructed_wav = original_wav[:min_len], reconstructed_wav[:min_len]
            
            original_wav = original_wav.astype(np.float32)
            reconstructed_wav = reconstructed_wav.astype(np.float32)

            pesq_score = pesq(sr, original_wav, reconstructed_wav, 'wb')
            stoi_score = stoi(original_wav, reconstructed_wav, sr, extended=False)

            results = {
                'original_wav': original_wav, 'reconstructed_wav': reconstructed_wav,
                'sr': sr, 'pesq': pesq_score, 'stoi': stoi_score, 
                'rtf': real_time_factor, 'error': None
            }
        except Exception as e:
            results['error'] = str(e)
            import traceback
            print(f"Error in evaluation: {e}")
            print(traceback.format_exc())

        self.finished.emit(results)

class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.original_wav = None
        self.reconstructed_wav = None
        
        # Audio Playback Control
        self.audio_thread = None
        self.stop_audio_event = threading.Event()
        self.audio_mutex = QMutex()
        
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File and Model Selection
        file_layout = QHBoxLayout()
        self.audio_file_edit = QLineEdit()
        self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
        self.browse_audio_button = QPushButton("Browse Audio...")
        self.browse_audio_button.clicked.connect(self.browse_audio)
        file_layout.addWidget(QLabel("Audio File:"))
        file_layout.addWidget(self.audio_file_edit)
        file_layout.addWidget(self.browse_audio_button)
        layout.addLayout(file_layout)

        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Path to TRAINED model (.pth). Mandatory for Tiny Transformer Codec.") 
        self.browse_model_button = QPushButton("Browse Model...")
        self.browse_model_button.clicked.connect(self.browse_model)
        self.model_type_combo = QComboBox()
        
        # Add all available models
        model_items = [
            "Uncompressed",
            "μ-Law Codec (Baseline)", 
            "A-Law Codec (Baseline)",
            "Tiny Transformer Codec (Custom, 16kHz, ~10ms)", # Your custom codec
        ]
        if DAC_AVAILABLE:
            model_items.append("DAC Codec (16kHz, 20ms)")
        if ENCODEC_AVAILABLE:
            model_items.append("EnCodec (24kHz, 6kbps)")
        
        self.model_type_combo.addItems(model_items)
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(QLabel("Codec Model:"))
        model_layout.addWidget(self.model_type_combo)
        model_layout.addWidget(self.model_path_edit)
        model_layout.addWidget(self.browse_model_button)
        layout.addLayout(model_layout)
        
        # Controls and Results
        self.run_eval_button = QPushButton("Run Evaluation (Calculate PESQ, STOI, RTF)")
        self.run_eval_button.clicked.connect(self.run_evaluation)
        layout.addWidget(self.run_eval_button)

        results_layout = QHBoxLayout()
        self.pesq_label = QLabel("PESQ: --")
        self.stoi_label = QLabel("STOI: --")
        self.rtf_label = QLabel("Real-Time Factor: --")
        self.bitrate_label = QLabel(f"Bitrate: N/A")
        results_layout.addWidget(self.pesq_label)
        results_layout.addWidget(self.stoi_label)
        results_layout.addWidget(self.rtf_label)
        results_layout.addWidget(self.bitrate_label)
        layout.addLayout(results_layout)
        
        # Playback Controls
        playback_layout = QHBoxLayout()
        self.play_original_button = QPushButton("▶ Play Original")
        self.play_original_button.setEnabled(False)
        self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav, self.play_original_button))
        
        self.play_reconstructed_button = QPushButton("▶ Play Reconstructed")
        self.play_reconstructed_button.setEnabled(False)
        self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav, self.play_reconstructed_button))

        self.stop_playback_button = QPushButton("■ Stop Playback")
        self.stop_playback_button.setEnabled(False)
        self.stop_playback_button.clicked.connect(self.stop_audio)
        
        playback_layout.addWidget(self.play_original_button) 
        playback_layout.addWidget(self.play_reconstructed_button)
        playback_layout.addWidget(self.stop_playback_button)
        layout.addLayout(playback_layout)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        # Plots
        plot_layout = QHBoxLayout()
        self.canvas_original = MplCanvas(self)
        self.canvas_reconstructed = MplCanvas(self)
        plot_layout.addWidget(self.canvas_original)
        plot_layout.addWidget(self.canvas_reconstructed)
        layout.addLayout(plot_layout)

        self.on_model_type_changed(self.model_type_combo.currentText())

    def on_model_type_changed(self, model_name):
        is_neural = any(x in model_name for x in ["DAC", "EnCodec", "Transformer"]) 
        is_custom_transformer = "Tiny Transformer Codec" in model_name
        
        self.model_path_edit.setEnabled(is_neural)
        self.browse_model_button.setEnabled(is_neural)
        
        if not is_neural:
            self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
            bitrate = '256 kbps' if 'Uncompressed' in model_name else '128 kbps'
            self.bitrate_label.setText(f"Bitrate: {bitrate}")
        elif "DAC" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            self.bitrate_label.setText(f"Bitrate: ~8-12 kbps | Latency: 20ms")
        elif "EnCodec" in model_name:
            self.model_path_edit.setText("(auto-download if not provided)")
            self.bitrate_label.setText(f"Bitrate: 6 kbps | Latency: ~10ms")
        elif is_custom_transformer:
            self.model_path_edit.setText("MANDATORY: Path to your trained checkpoint file (.pth)")
            self.bitrate_label.setText(f"Bitrate: 9.00 kbps (Calculated) | Latency: 20ms")

    def browse_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
        if filepath:
            self.audio_file_edit.setText(filepath)

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth *.pt)")
        if filepath:
            self.model_path_edit.setText(filepath)
            
    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None

        if "Uncompressed" in model_type_str:
            self.status_label.setText("Status: Using Uncompressed Passthrough.")
            return True
        elif "μ-Law" in model_type_str: 
            self.model = MuLawCodec()
            self.status_label.setText("Status: Loaded μ-Law Codec.")
            return True
        elif "A-Law" in model_type_str: 
            self.model = ALawCodec()
            self.status_label.setText("Status: Loaded A-Law Codec.")
            return True
        elif "DAC" in model_type_str:
            if not DAC_AVAILABLE:
                self.status_label.setText("Status: ERROR - DAC not installed.")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                self.model = DACCodec(model_path=path, model_type="16khz")
                self.status_label.setText("Status: Loaded DAC Codec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load DAC: {e}")
                return False
        elif "EnCodec" in model_type_str:
            if not ENCODEC_AVAILABLE:
                self.status_label.setText("Status: ERROR - EnCodec not installed.")
                return False
            try:
                path = self.model_path_edit.text()
                if "auto-download" in path or not path or "N/A" in path:
                    path = None
                self.model = EnCodecModel(model_path=path)
                self.status_label.setText("Status: Loaded EnCodec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load EnCodec: {e}")
                return False
        elif "Tiny Transformer Codec" in model_type_str: # Custom Codec Loading
            model_path = self.model_path_edit.text()
            if "MANDATORY" in model_path or not os.path.exists(model_path): 
                self.status_label.setText("Status: ERROR - Please provide a valid path to your trained Tiny Transformer Codec (.pth) model.")
                return False
            try:
                self.model = TinyTransformerCodec.load_model(model_path)
                self.status_label.setText("Status: Loaded Tiny Transformer Codec.")
                return True
            except Exception as e:
                self.status_label.setText(f"Status: ERROR - Failed to load Tiny Transformer Codec: {e}")
                return False
            
        return False

    def run_evaluation(self):
        # Stop any audio before running evaluation
        self.stop_audio()
        
        if not self.audio_file_edit.text():
            self.status_label.setText("Status: Please select an audio file.")
            return
        if not self.load_model():
            return

        self.status_label.setText("Status: Evaluating... Please wait.")
        self.run_eval_button.setEnabled(False)
        self.play_original_button.setEnabled(False)
        self.play_reconstructed_button.setEnabled(False)

        self.eval_thread = QThread()
        self.eval_worker = EvaluationWorker(self.model, self.audio_file_edit.text(), self.model_type_combo.currentText())
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_worker.finished.connect(self.on_evaluation_complete)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_thread.start()

    def on_evaluation_complete(self, results):
        if results.get('error'):
            self.status_label.setText(f"Status: ERROR - {results['error']}")
        else:
            pesq_score = results['pesq']
            stoi_score = results['stoi']
            rtf_score = results['rtf']
            
            # Highlight scores based on targets
            pesq_color = 'green' if pesq_score >= 3.5 else 'orange'
            stoi_color = 'green' if stoi_score >= 0.9 else 'orange'
            rtf_color = 'green' if rtf_score < 1.0 else 'red'
            
            self.pesq_label.setText(f"PESQ: <font color='{pesq_color}'>{pesq_score:.4f}</font>")
            self.stoi_label.setText(f"STOI: <font color='{stoi_color}'>{stoi_score:.4f}</font>")
            self.rtf_label.setText(f"Real-Time Factor: <font color='{rtf_color}'>{rtf_score:.3f}</font>")
            
            self.original_wav = results['original_wav']
            self.reconstructed_wav = results['reconstructed_wav']
            self.play_original_button.setEnabled(True)
            self.play_reconstructed_button.setEnabled(True)
            self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
            self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
            if rtf_score > 1.0:
                self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, NOT real-time capable)")
            else:
                self.status_label.setText("Status: Evaluation complete. (RTF < 1.0, real-time capable)")
            
        self.run_eval_button.setEnabled(True)
        if self.eval_thread:
            self.eval_thread.quit()
            self.eval_thread.wait()
    
    def stop_audio(self):
        """Stops the currently playing audio thread."""
        if self.audio_thread and self.audio_thread.is_alive():
            self.stop_audio_event.set()
            self.audio_thread.join(timeout=0.1)
            self.audio_thread = None
            self.status_label.setText("Status: Playback stopped.")
        
        self.play_original_button.setText("▶ Play Original")
        self.play_reconstructed_button.setText("▶ Play Reconstructed")
        self.stop_playback_button.setEnabled(False)

    def play_audio(self, wav_data, button_clicked):
        if wav_data is None:
            self.status_label.setText("Status: No audio data to play.")
            return
        
        self.stop_audio()
        
        self.stop_playback_button.setEnabled(True)
        button_clicked.setText("Playing...")

        self.stop_audio_event.clear()
        self.status_label.setText("Status: Playing audio...")
        self.audio_thread = threading.Thread(
            target=self._play_audio_thread, 
            args=(wav_data, 16000, button_clicked), 
            daemon=True
        )
        self.audio_thread.start()

    def _play_audio_thread(self, wav_data, sr, button):
        p = None
        stream = None
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            
            chunk_size = 1024
            data_to_play = wav_data.astype(np.float32)
            
            for i in range(0, len(data_to_play), chunk_size):
                if self.stop_audio_event.is_set():
                    break
                
                chunk = data_to_play[i:i + chunk_size].tobytes()
                stream.write(chunk)
                
            self.audio_mutex.lock()
            if not self.stop_audio_event.is_set():
                self.status_label.setText("Status: Playback finished.")
            self.audio_mutex.unlock()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.status_label.setText(f"Status: Playback error: {e}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            if p:
                p.terminate()
            
            button.setText(f"▶ Play {'Original' if button == self.play_original_button else 'Reconstructed'}")
            if not self.stop_audio_event.is_set():
                self.stop_playback_button.setEnabled(False)

    def plot_spectrogram(self, canvas, wav, sr, title):
        try:
            canvas.axes.cla()
            import librosa.display
            S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
            canvas.axes.set_title(title)
            canvas.fig.tight_layout()
            canvas.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")

    def closeEvent(self, event):
        self.stop_audio()
        super().closeEvent(event)
