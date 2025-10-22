import os
import threading
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QTextEdit, QSpinBox, QComboBox, QGroupBox, QCheckBox, QListWidget
from PyQt5.QtCore import pyqtSignal, QObject, QThread, Qt
from PyQt5.QtGui import QFont

import socket
import pyaudio
import numpy as np
import torch
import time
import librosa

# Import all necessary models
from model import (
    MuLawCodec, ALawCodec, 
    GRU_Codec, TS3_Codec, ScoreDecPostFilter,
    HOP_SIZE, NUM_QUANTIZERS
)

# --- Configuration ---
BROADCAST_PORT = 37020
STREAM_PORT = 37021
DEVICE_ID = "NeuralCodecPC"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = HOP_SIZE # 320 samples = 20ms. This is the new latency target.

# --- Network Discovery Worker ---
class DiscoveryWorker(QObject):
    peer_found = pyqtSignal(str, str)
    def __init__(self): super().__init__(); self._running = True
    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('', BROADCAST_PORT))
            except OSError as e:
                print(f"Discovery: Could not bind to port {BROADCAST_PORT}. {e}")
                return
            s.settimeout(1.0)
            while self._running:
                try:
                    data, addr = s.recvfrom(1024)
                    message = data.decode()
                    if message.startswith(DEVICE_ID):
                        self.peer_found.emit(message.split(':')[1], addr[0])
                except socket.timeout: continue
                except Exception as e: print(f"Discovery error: {e}")
    def stop(self): self._running = False

# --- Audio Streaming Worker ---
class StreamerWorker(QObject):
    log_message = pyqtSignal(str)
    def __init__(self, target_ip, model, post_filter, model_type_str):
        super().__init__()
        self.target_ip, self.model, self.post_filter, self.model_type_str = target_ip, model, post_filter, model_type_str
        self.p = pyaudio.PyAudio()
        self._running = True
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_muted = False
        self.file_playback_path = None
        self.file_playback_event = threading.Event()
        
        # --- STATEFUL HIDDEN STATES ---
        self.h_enc, self.h_dec = None, None
        
        # --- FIX: Check if models are PyTorch modules before moving to device ---
        if self.model and isinstance(self.model, torch.nn.Module):
            self.model.to(self.device)
            self.model.eval()
        if self.post_filter and isinstance(self.post_filter, torch.nn.Module):
            self.post_filter.to(self.device)
            self.post_filter.eval()
        # --- End Fix ---

    def run(self):
        self.sender_thread = threading.Thread(target=self.send_audio, daemon=True)
        self.receiver_thread = threading.Thread(target=self.receive_audio, daemon=True)
        self.sender_thread.start(); self.receiver_thread.start()
        self.sender_thread.join(); self.receiver_thread.join()

    def set_mute(self, muted):
        self.is_muted = muted
        self.log_message.emit(f"Microphone {'muted' if muted else 'unmuted'}.")

    def start_file_playback(self, filepath):
        self.file_playback_path = filepath
        self.file_playback_event.set()
        self.log_message.emit(f"Queued file for playback: {filepath}")

    def stop_file_playback(self): self.file_playback_path = None; self.file_playback_event.clear()

    def encode_data(self, data_bytes):
        """Encodes raw audio bytes based on the selected model."""
        try:
            # Convert raw bytes to float tensor
            audio_np = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0) # Keep on CPU for traditional codecs
            
            if not self.model:
                # Uncompressed: just return the raw int16 bytes
                return data_bytes 
            
            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                # Traditional codecs are stateless and run on CPU
                encoded_tensor = self.model.encode(audio_tensor)
                return encoded_tensor.cpu().numpy().tobytes() # uint8

            # --- STATEFUL NEURAL ENCODING ---
            # Move tensor to GPU *only* for neural models
            audio_tensor = audio_tensor.to(self.device)
            with torch.no_grad():
                # This is the new, low-latency, high-compression path
                indices, self.h_enc = self.model.encode(audio_tensor, self.h_enc)
                
                # indices is (1, 40). We must pack this into 40 bytes.
                # Since VQ_EMBEDDINGS is 256 (8 bits), each index fits in a uint8.
                payload_bytes = indices.cpu().numpy().astype(np.uint8).tobytes()
                
                # payload_bytes should be exactly NUM_QUANTIZERS (40) bytes
                return payload_bytes

        except Exception as e:
            self.log_message.emit(f"Encoding error: {e}"); return None

    def play_file_audio(self, filepath, s):
        try:
            wav, _ = librosa.load(filepath, sr=RATE, mono=True)
            self.log_message.emit(f"Sending file: {filepath}...")
            wav_int16 = (wav * 32767.0).astype(np.int16)
            
            # Reset state for file playback
            self.h_enc = None
            
            for i in range(0, len(wav_int16), CHUNK):
                if not self._running or self.file_playback_path != filepath: break
                
                chunk_data = wav_int16[i:i+CHUNK]
                # Pad the last chunk
                if len(chunk_data) < CHUNK:
                    chunk_data = np.pad(chunk_data, (0, CHUNK - len(chunk_data)), 'constant')
                
                chunk_bytes = chunk_data.tobytes()
                
                payload = self.encode_data(chunk_bytes)
                if payload: s.sendto(payload, (self.target_ip, STREAM_PORT))
                time.sleep(float(CHUNK) / RATE) # Pace the sending
            
            self.log_message.emit("File playback finished.")
            # Reset state back to None for microphone
            self.h_enc = None 
            
        except Exception as e: self.log_message.emit(f"Error playing file: {e}")

    def send_audio(self):
        try:
            stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                while self._running:
                    if self.file_playback_event.is_set():
                        self.file_playback_event.clear()
                        if self.file_playback_path: self.play_file_audio(self.file_playback_path, s)
                        if self.file_playback_path: self.file_playback_path = None
                        continue
                    
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    payload = self.encode_data(b'\x00' * (CHUNK * 2) if self.is_muted else data)
                    if payload: s.sendto(payload, (self.target_ip, STREAM_PORT))
        except Exception as e: 
            if self._running: self.log_message.emit(f"ERROR in send_audio: {e}")
        finally:
            if hasattr(self, 'p'): self.p.terminate()

    def receive_audio(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(('', STREAM_PORT))
                s.settimeout(2.0)
                stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True, frames_per_buffer=CHUNK)
                
                # Reset decoder state
                self.h_dec = None
                
                while self._running:
                    try:
                        data, _ = s.recvfrom(65536)
                    except socket.timeout:
                        self.log_message.emit("Socket timeout, resetting decoder state.")
                        # If we lose connection, we must reset the state
                        self.h_dec = None
                        continue
                        
                    if not self.model:
                        payload = data # Uncompressed
                    else:
                        try:
                            if isinstance(self.model, (MuLawCodec, ALawCodec)):
                                # Traditional codecs are stateless and run on CPU
                                latent_tensor = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0)
                                decoded_tensor = self.model.decode(latent_tensor)
                            else:
                                # --- STATEFUL NEURAL DECODING ---
                                with torch.no_grad():
                                    # Data is 40 bytes (uint8)
                                    indices = torch.from_numpy(np.frombuffer(data, dtype=np.uint8)).unsqueeze(0).to(self.device, dtype=torch.long) # (1, 40)
                                    
                                    # Check for corrupted packet
                                    if indices.shape[1] != NUM_QUANTIZERS:
                                        self.log_message.emit(f"Corrupted packet received. Expected {NUM_QUANTIZERS} bytes, got {indices.shape[1]}. Resetting state.")
                                        self.h_dec = None
                                        continue

                                    decoded_tensor, self.h_dec = self.model.decode(indices, self.h_dec)
                                    
                                    # --- APPLY POST-FILTER (HIGH LATENCY) ---
                                    if self.post_filter:
                                        # Run enhancement. Use a small number of steps for "faster" (but still slow) streaming
                                        decoded_tensor = self.post_filter.enhance(decoded_tensor, timesteps=5)
                                
                            # Trim to CHUNK size and convert to bytes
                            decoded_tensor = decoded_tensor[..., :CHUNK]
                            payload = (decoded_tensor.squeeze().cpu().numpy() * 32767.0).astype(np.int16).tobytes()

                        except Exception as e: self.log_message.emit(f"Decoding error: {e}"); continue
                    
                    stream.write(payload)
        except Exception as e:
            if self._running: self.log_message.emit(f"ERROR in receive_audio: {e}")
        finally:
            if hasattr(self, 'p'): self.p.terminate()

    def stop(self): 
        self._running = False; self.stop_file_playback()


class StreamingTab(QWidget):
    def __init__(self):
        super().__init__()
        self.peers = {}
        self.streamer_worker = None
        self.streamer_thread = None
        self._setup_ui()
        self._start_discovery()
        self.model = None
        self.post_filter = None # For ScoreDec

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # --- Model and File Configuration Group ---
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("<b>Codec:</b>"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems([
            "Uncompressed",
            "μ-Law Codec", 
            "A-Law Codec",
            "GRU_Codec (16kbps, Fast)", 
            "TS3_Codec (16kbps, Transformer)",
            "GRU_Codec + ScoreDec (High Latency)"
        ])
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        config_layout.addLayout(model_layout)
        
        self.model_path_edit = QLineEdit(); self.model_path_edit.setPlaceholderText("Path to trained model (.pth)...")
        self.browse_model_button = QPushButton("Browse..."); self.browse_model_button.clicked.connect(self.browse_model)
        self.load_model_button = QPushButton("Load Model"); self.load_model_button.clicked.connect(self.load_model)
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(self.model_path_edit); model_path_layout.addWidget(self.browse_model_button); model_path_layout.addWidget(self.load_model_button)
        config_layout.addLayout(model_path_layout)

        self.play_file_path_edit = QLineEdit(); self.play_file_path_edit.setPlaceholderText("Path to audio file for playback...")
        self.browse_play_file_button = QPushButton("Browse..."); self.browse_play_file_button.clicked.connect(self.browse_play_file)
        file_playback_layout = QHBoxLayout()
        file_playback_layout.addWidget(self.play_file_path_edit); file_playback_layout.addWidget(self.browse_play_file_button)
        config_layout.addLayout(file_playback_layout)
        layout.addWidget(config_group)

        # --- Network Group ---
        network_group = QGroupBox("Network")
        network_layout = QVBoxLayout(network_group)
        network_layout.addWidget(QLabel("<b>Available Peers:</b>"))
        self.peer_list = QListWidget(); self.peer_list.setMaximumHeight(100)
        self.refresh_button = QPushButton("Refresh List"); self.refresh_button.clicked.connect(self.send_broadcast)
        network_layout.addWidget(self.peer_list); network_layout.addWidget(self.refresh_button, 0, Qt.AlignRight)
        layout.addWidget(network_group)

        # --- Controls Group ---
        controls_group = QGroupBox("Controls")
        controls_layout = QHBoxLayout(controls_group)
        self.connect_button = QPushButton("Start Streaming"); self.connect_button.setStyleSheet("background-color: #4CAF50; color: white; padding: 5px; border-radius: 5px;"); self.connect_button.clicked.connect(self.start_streaming)
        self.disconnect_button = QPushButton("Stop Streaming"); self.disconnect_button.setStyleSheet("background-color: #f44336; color: white; padding: 5px; border-radius: 5px;"); self.disconnect_button.setEnabled(False); self.disconnect_button.clicked.connect(self.stop_streaming)
        self.play_file_button = QPushButton("▶ Play File"); self.play_file_button.setEnabled(False); self.play_file_button.clicked.connect(self.start_file_playback)
        self.mute_mic_checkbox = QCheckBox("Mute Mic"); self.mute_mic_checkbox.stateChanged.connect(self.on_mute_changed)
        self.status_label = QLabel("<b>Status:</b> <font color='red'>Disconnected</font>")
        
        controls_layout.addWidget(self.connect_button); controls_layout.addWidget(self.disconnect_button); controls_layout.addWidget(self.play_file_button)
        controls_layout.addStretch(); controls_layout.addWidget(self.mute_mic_checkbox); controls_layout.addWidget(self.status_label)
        layout.addWidget(controls_group)

        # --- Log Group ---
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text_edit = QTextEdit(); self.log_text_edit.setReadOnly(True)
        log_layout.addWidget(self.log_text_edit)
        layout.addWidget(log_group)
        
        self.on_model_type_changed(self.model_type_combo.currentText())

    def log(self, message): 
        self.log_text_edit.append(message)
        self.log_text_edit.verticalScrollBar().setValue(self.log_text_edit.verticalScrollBar().maximum())

    def on_model_type_changed(self, model_name):
        is_neural = "Codec" in model_name and "Law" not in model_name
        is_combo = "ScoreDec" in model_name
        
        self.model_path_edit.setEnabled(is_neural and not is_combo)
        self.browse_model_button.setEnabled(is_neural and not is_combo)
        self.load_model_button.setEnabled(is_neural and not is_combo)

        if not (is_neural or is_combo):
            self.model_path_edit.setText("N/A (Traditional or Uncompressed)")
            self.load_model() # Automatically load the traditional codec
        elif is_combo:
            self.model_path_edit.setText("Loads '..._gru.pth' and '..._scoredec_...pth'")
            self.load_model() # Automatically load the combo models
        else:
            self.model_path_edit.setText("")

    def on_mute_changed(self, state):
        if self.streamer_worker: self.streamer_worker.set_mute(state == Qt.Checked)

    def browse_play_file(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if filepath: self.play_file_path_edit.setText(filepath)

    def start_file_playback(self):
        if not self.play_file_path_edit.text(): self.log("ERROR: Please select a file to play."); return
        if self.streamer_worker: self.streamer_worker.start_file_playback(self.play_file_path_edit.text())
        else: self.log("ERROR: Must be streaming to play a file.")

    def browse_model(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch Models (*.pth)")
        if filepath: self.model_path_edit.setText(filepath)

    def load_model(self):
        model_type_str = self.model_type_combo.currentText()
        self.model = None
        self.post_filter = None

        if "Uncompressed" in model_type_str: 
            self.model = None; self.log("Streaming uncompressed."); return True
        if "μ-Law" in model_type_str: 
            self.model = MuLawCodec(); self.log("Loaded μ-Law Codec."); return True
        if "A-Law" in model_type_str: 
            self.model = ALawCodec(); self.log("Loaded A-Law Codec."); return True
        
        try:
            if "GRU_Codec + ScoreDec" in model_type_str:
                self.model = GRU_Codec()
                self.model.load_state_dict(torch.load("low_latency_codec_gru.pth", map_location=torch.device('cpu')))
                self.model.eval()
                
                self.post_filter = ScoreDecPostFilter()
                self.post_filter.load_state_dict(torch.load("low_latency_codec_scoredec_post_filter.pth", map_location=torch.device('cpu')))
                self.post_filter.eval()
                
                self.log("Loaded GRU_Codec + ScoreDec Post-Filter.")
                return True

            path = self.model_path_edit.text()
            if not path: self.model = None; return False # Don't log error if path is just empty
            
            if "GRU_Codec" in model_type_str: 
                self.model = GRU_Codec()
            elif "TS3_Codec" in model_type_str: 
                self.model = TS3_Codec()
            else: 
                self.model = None; return False

            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
            self.model.eval(); self.log(f"Successfully loaded model: {path}"); return True
        except Exception as e:
            self.log(f"ERROR: Failed to load model. {e}"); self.model = None; self.post_filter = None; return False

    def add_peer(self, name, ip):
        if ip not in self.peers.values(): self.peers[name] = ip; self.peer_list.addItem(f"{name} ({ip})")

    def send_broadcast(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                s.sendto(f"{DEVICE_ID}:{socket.gethostname()}".encode(), ('<broadcast>', BROADCAST_PORT))
            self.log("Sent discovery broadcast.")
        except Exception as e:
            self.log(f"Error sending broadcast: {e}")

    def _start_discovery(self):
        self.discovery_thread = QThread()
        self.discovery_worker = DiscoveryWorker()
        self.discovery_worker.moveToThread(self.discovery_thread)
        self.discovery_worker.peer_found.connect(self.add_peer)
        self.discovery_thread.started.connect(self.discovery_worker.run)
        self.discovery_thread.start(); self.send_broadcast()

    def start_streaming(self):
        if not self.peer_list.selectedItems(): self.log("ERROR: Please select a peer."); return
        
        model_type_str = self.model_type_combo.currentText()
        target_ip = self.peer_list.selectedItems()[0].text().split('(')[-1].strip(')')
        
        if not self.load_model() and "Uncompressed" not in model_type_str and "Codec" in model_type_str:
                 self.log("ERROR: Model could not be loaded. Aborting stream."); return

        self.log(f"--- Starting stream to {target_ip} with {model_type_str} ---")
        self.log(f"Frame size: {CHUNK} samples (20ms)")
        if "Codec" in model_type_str and "Law" not in model_type_str:
             self.log(f"Bitrate: {NUM_QUANTIZERS * 8} bits / 20ms = 16.0 kbps")
        elif "Law" in model_type_str:
            self.log(f"Bitrate: 8 bits/sample * 16000 Hz = 128 kbps")
        else:
            self.log(f"Bitrate: 16 bits/sample * 16000 Hz = 256 kbps")
        
        # --- CRITICAL WARNING FOR LATENCY ---
        if "ScoreDec" in model_type_str:
            self.log("--------------------------------------------------")
            self.log("WARNING: ScoreDec (Diffusion) is active.")
            self.log("This is a HIGH-LATENCY post-filter and will")
            self.log("NOT meet the < 20ms real-time goal.")
            self.log("This mode is for quality comparison only.")
            self.log("--------------------------------------------------")


        self.streamer_thread = QThread()
        # Pass both the main model and the post_filter (which may be None)
        self.streamer_worker = StreamerWorker(target_ip, self.model, self.post_filter, model_type_str)
        self.streamer_worker.moveToThread(self.streamer_thread)
        self.streamer_worker.log_message.connect(self.log)
        self.streamer_thread.started.connect(self.streamer_worker.run)
        self.streamer_thread.start()
        
        self.connect_button.setEnabled(False); self.disconnect_button.setEnabled(True)
        self.play_file_button.setEnabled(True); self.model_type_combo.setEnabled(False)
        self.status_label.setText("<b>Status:</b> <font color='green'>Connected</font>")

    def stop_streaming(self):
        if self.streamer_worker: self.streamer_worker.stop()
        if self.streamer_thread: self.streamer_thread.quit(); self.streamer_thread.wait()
        self.streamer_worker = None; self.streamer_thread = None
        self.connect_button.setEnabled(True); self.disconnect_button.setEnabled(False)
        self.play_file_button.setEnabled(False); self.model_type_combo.setEnabled(True)
        self.log("Streaming stopped.")
        self.status_label.setText("<b>Status:</b> <font color='red'>Disconnected</font>")

    def closeEvent(self):
        self.stop_streaming()
        if hasattr(self, 'discovery_worker'): self.discovery_worker.stop()
        if hasattr(self, 'discovery_thread'): self.discovery_thread.quit(); self.discovery_thread.wait()

