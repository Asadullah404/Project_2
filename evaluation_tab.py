import numpy as np
import librosa
import soundfile as sf
import torch
from pesq import pesq
from pystoi import stoi
import pyaudio
import threading
import time

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QComboBox
from PyQt5.QtCore import QObject, pyqtSignal, QThread

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import all models, including the new optimized ones
from model import (
    GRU_Codec, TS3_Codec, MuLawCodec, ALawCodec, ScoreDecPostFilter, HOP_SIZE
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
    
    def __init__(self, model, post_filter, original_file_path):
        super().__init__()
        self.model = model
        self.post_filter = post_filter # This can be None
        self.original_file_path = original_file_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        results = {}
        try:
            sr = 16000
            original_wav, _ = librosa.load(self.original_file_path, sr=sr, mono=True)
            
            start_time = time.time()
            
            if self.model:
                if isinstance(self.model, torch.nn.Module):
                    self.model.to(self.device)
                    self.model.eval()
                    
                    chunk_size = HOP_SIZE # 320 samples
                    reconstructed_parts = []
                    original_len = len(original_wav)

                    if original_len % chunk_size != 0:
                        padding = chunk_size - (original_len % chunk_size)
                        padded_wav = np.pad(original_wav, (0, padding), 'constant')
                    else:
                        padded_wav = original_wav
                    
                    h_enc, h_dec = None, None

                    with torch.no_grad():
                        for i in range(0, len(padded_wav), chunk_size):
                            chunk = padded_wav[i:i+chunk_size]
                            audio_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                            
                            indices, h_enc = self.model.encode(audio_tensor, h_enc)
                            reconstructed_tensor, h_dec = self.model.decode(indices, h_dec)
                            
                            reconstructed_parts.append(reconstructed_tensor.squeeze().cpu().numpy())

                    reconstructed_wav = np.concatenate(reconstructed_parts)
                    reconstructed_wav = reconstructed_wav[:original_len]

                else: # Handle traditional codecs (μ-Law, A-Law)
                    audio_tensor = torch.from_numpy(original_wav).unsqueeze(0).unsqueeze(0)
                    encoded = self.model.encode(audio_tensor)
                    reconstructed_tensor = self.model.decode(encoded)
                    reconstructed_wav = reconstructed_tensor.squeeze().cpu().numpy()
            else:
                reconstructed_wav = np.copy(original_wav)

            # --- Apply Post-Filter if it exists ---
            if self.post_filter:
                self.post_filter.to(self.device)
                self.post_filter.eval()
                
                enhanced_parts = []
                # Process in chunks to avoid OOM
                with torch.no_grad():
                    for i in range(0, len(reconstructed_wav), chunk_size):
                        if i + chunk_size > len(reconstructed_wav):
                            chunk = reconstructed_wav[i:]
                            pad = chunk_size - len(chunk)
                            chunk = np.pad(chunk, (0, pad), 'constant')
                        else:
                            chunk = reconstructed_wav[i:i+chunk_size]
                        
                        chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self.device, dtype=torch.float32)
                        
                        # Enhance with 10 diffusion steps (slow!)
                        enhanced_chunk = self.post_filter.enhance(chunk_tensor, timesteps=10) 
                        enhanced_parts.append(enhanced_chunk.squeeze().cpu().numpy())

                reconstructed_wav = np.concatenate(enhanced_parts)
                reconstructed_wav = reconstructed_wav[:original_len]
            # --- End Post-Filter ---

            end_time = time.time()
            processing_time = end_time - start_time
            audio_duration = len(original_wav) / sr
            real_time_factor = processing_time / audio_duration

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
            print(f"Error in evaluation: {e}") 

        self.finished.emit(results)

class EvaluationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.post_filter = None # For ScoreDec
        self.original_wav = None
        self.reconstructed_wav = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # File and Model Selection
        file_layout = QHBoxLayout()
        self.audio_file_edit = QLineEdit(); self.audio_file_edit.setPlaceholderText("Path to original audio file (.wav)...")
        self.browse_audio_button = QPushButton("Browse Audio..."); self.browse_audio_button.clicked.connect(self.browse_audio)
        file_layout.addWidget(QLabel("Audio File:")); file_layout.addWidget(self.audio_file_edit); file_layout.addWidget(self.browse_audio_button)
        layout.addLayout(file_layout)

        model_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit(); self.model_path_edit.setPlaceholderText("Path to trained model (.pth)...")
        self.browse_model_button = QPushButton("Browse Model..."); self.browse_model_button.clicked.connect(self.browse_model)
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "GRU_Codec (16kbps, Fast)", 
            "TS3_Codec (16kbps, Transformer)",
            "GRU_Codec + ScoreDec (High Quality)",
            "μ-Law Codec", 
            "A-Law Codec"
        ])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(QLabel("Codec Model:")); model_layout.addWidget(self.model_type_combo); model_layout.addWidget(self.model_path_edit); model_layout.addWidget(self.browse_model_button)
        layout.addLayout(model_layout)
        
        # Controls and Results
        self.run_eval_button = QPushButton("Run Evaluation"); self.run_eval_button.clicked.connect(self.run_evaluation)
        layout.addWidget(self.run_eval_button)

        results_layout = QHBoxLayout()
        self.pesq_label = QLabel("PESQ: --"); self.stoi_label = QLabel("STOI: --")
        self.rtf_label = QLabel("Real-Time Factor: --")
        results_layout.addWidget(self.pesq_label); results_layout.addWidget(self.stoi_label); results_layout.addWidget(self.rtf_label)
        layout.addLayout(results_layout)
        
        playback_layout = QHBoxLayout()
        self.play_original_button = QPushButton("▶ Play Original"); self.play_original_button.setEnabled(False)
        self.play_original_button.clicked.connect(lambda: self.play_audio(self.original_wav))
        self.play_reconstructed_button = QPushButton("▶ Play Reconstructed"); self.play_reconstructed_button.setEnabled(False)
        self.play_reconstructed_button.clicked.connect(lambda: self.play_audio(self.reconstructed_wav))
        playback_layout.addWidget(self.play_original_button); playback_layout.addWidget(self.play_reconstructed_button)
        layout.addLayout(playback_layout)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        # Plots
        plot_layout = QHBoxLayout()
        self.canvas_original = MplCanvas(self); self.canvas_reconstructed = MplCanvas(self)
        plot_layout.addWidget(self.canvas_original); plot_layout.addWidget(self.canvas_reconstructed)
        layout.addLayout(plot_layout)

        self.on_model_type_changed(self.model_type_combo.currentText())

    def on_model_type_changed(self, model_name):
        is_neural = "Codec" in model_name
        is_combo = "ScoreDec" in model_name
        
        self.model_path_edit.setEnabled(is_neural or is_combo)
        self.browse_model_button.setEnabled(is_neural or is_combo)
        
        if not (is_neural or is_combo):
            self.model_path_edit.setText("N/A (Traditional Codec)")
        elif is_combo:
            self.model_path_edit.setText("Loads '..._gru.pth' and '..._scoredec_...pth'")
            self.model_path_edit.setEnabled(False)
            self.browse_model_button.setEnabled(False)
        else:
            self.model_path_edit.setText("")
            self.model_path_edit.setEnabled(True)
            self.browse_model_button.setEnabled(True)


    def browse_audio(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.flac)")
        if filepath: self.audio_file_edit.setText(filepath)

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if filepath: self.model_path_edit.setText(filepath)
            
    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None
        self.post_filter = None # Reset post-filter

        if "μ-Law" in model_type_str: 
            self.model = MuLawCodec(); self.status_label.setText("Status: Loaded μ-Law Codec."); return True
        if "A-Law" in model_type_str: 
            self.model = ALawCodec(); self.status_label.setText("Status: Loaded A-Law Codec."); return True

        try:
            if "GRU_Codec + ScoreDec" in model_type_str:
                self.model = GRU_Codec()
                self.model.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=torch.device('cpu')))
                self.model.eval()
                
                self.post_filter = ScoreDecPostFilter()
                self.post_filter.load_state_dict(torch.load("low_latency_codec_scoredec_post_filter.pth", map_location=torch.device('cpu')))
                self.post_filter.eval()
                
                self.status_label.setText("Status: Loaded GRU_Codec + ScoreDec Post-Filter.")
                return True

            path = self.model_path_edit.text()
            if not path or "N/A" in path:
                self.status_label.setText("Status: ERROR - Please provide a path for the neural model."); return False

            if "GRU_Codec" in model_type_str: 
                self.model = GRU_Codec()
            elif "TS3_Codec" in model_type_str: 
                self.model = TS3_Codec()
            else: 
                self.model = None; return False

            if self.model:
                self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
                self.model.eval()
                self.status_label.setText(f"Status: Model {model_type_str} loaded successfully.")
                return True
        except Exception as e:
            self.status_label.setText(f"Status: ERROR - Failed to load model: {e}"); self.model = None; self.post_filter = None; return False
        return False

    def run_evaluation(self):
        if not self.audio_file_edit.text():
            self.status_label.setText("Status: Please select an audio file."); return
        if not self.load_model(): return

        self.status_label.setText("Status: Evaluating... Please wait."); self.run_eval_button.setEnabled(False)

        self.eval_thread = QThread()
        # Pass both model and post-filter (which might be None)
        self.eval_worker = EvaluationWorker(self.model, self.post_filter, self.audio_file_edit.text())
        self.eval_worker.moveToThread(self.eval_thread)
        self.eval_worker.finished.connect(self.on_evaluation_complete)
        self.eval_thread.started.connect(self.eval_worker.run)
        self.eval_thread.start()

    def on_evaluation_complete(self, results):
        if results.get('error'):
            self.status_label.setText(f"Status: ERROR - {results['error']}")
        else:
            self.pesq_label.setText(f"PESQ: {results['pesq']:.4f}")
            self.stoi_label.setText(f"STOI: {results['stoi']:.4f}")
            # RTF = 1.0 means it runs at exactly real-time. < 1.0 is faster. > 1.0 is slower.
            self.rtf_label.setText(f"Real-Time Factor: {results['rtf']:.3f}")
            
            self.original_wav, self.reconstructed_wav = results['original_wav'], results['reconstructed_wav']
            self.play_original_button.setEnabled(True); self.play_reconstructed_button.setEnabled(True)
            self.plot_spectrogram(self.canvas_original, results['original_wav'], results['sr'], "Original Spectrogram")
            self.plot_spectrogram(self.canvas_reconstructed, results['reconstructed_wav'], results['sr'], "Reconstructed Spectrogram")
            
            if results['rtf'] > 1.0 and "ScoreDec" in self.model_type_combo.currentText():
                 self.status_label.setText("Status: Evaluation complete. (Note: RTF > 1.0, not real-time)")
            elif results['rtf'] > 1.0:
                self.status_label.setText("Status: Evaluation complete. (Warning: RTF > 1.0, model is too slow)")
            else:
                self.status_label.setText("Status: Evaluation complete. (RTF < 1.0, real-time capable)")
        
        self.run_eval_button.setEnabled(True)
        if self.eval_thread:
            self.eval_thread.quit(); self.eval_thread.wait()
    
    def play_audio(self, wav_data):
        if wav_data is not None:
            self.status_label.setText("Status: Playing audio...")
            threading.Thread(target=self._play_audio_thread, args=(wav_data, 16000), daemon=True).start()
        else:
            self.status_label.setText("Status: No audio data to play.")

    def _play_audio_thread(self, wav_data, sr):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sr, output=True)
            stream.write(wav_data.astype(np.float32).tobytes())
            stream.stop_stream(); stream.close(); p.terminate()
        except Exception as e:
            print(f"Error playing audio: {e}")
            
    def plot_spectrogram(self, canvas, wav, sr, title):
        try:
            canvas.axes.cla()
            S_db = librosa.amplitude_to_db(np.abs(librosa.stft(wav)), ref=np.max)
            librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', ax=canvas.axes)
            canvas.axes.set_title(title); canvas.fig.tight_layout(); canvas.draw()
        except Exception as e:
            print(f"Error plotting spectrogram: {e}")

